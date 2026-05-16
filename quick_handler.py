"""API Gateway Lambda for creating and polling JD/resume matching jobs."""

import json
import uuid
import boto3
import os
from datetime import datetime, timezone
from typing import Dict, Any
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

from utils.s3_cloudfront import json_dumps_decimal
from utils.hash import calculate_hash

# Initialize AWS clients
# AWS clients are created at module import time so warm Lambda invocations can
# reuse connections instead of constructing new clients for every request.
dynamodb = boto3.resource('dynamodb')
sqs = boto3.client('sqs')

# Environment variables
# Required deployment configuration. The infra stack injects these values into
# the quick Lambda so it knows where to persist jobs and which queue to notify.
JOBS_TABLE_NAME = os.environ['JOBS_TABLE_NAME']
JOB_QUEUE_URL = os.environ['JOB_QUEUE_URL']

# DynamoDB table
# DynamoDB table that stores one item per matching job.
jobs_table = dynamodb.Table(JOBS_TABLE_NAME)


def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standardized API Gateway response."""

    # API Gateway expects the Lambda proxy integration shape: statusCode,
    # headers, and a JSON string body. ``json_dumps_decimal`` is used because
    # DynamoDB may return Decimal values that the standard JSON encoder cannot
    # serialize.
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
        },
        'body': json_dumps_decimal(body)
    }


def handle_post_process(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle POST /process requests - check existing job by cache_key first, then create new job and queue it."""

    # This endpoint is intentionally quick. It validates the request, creates a
    # PENDING job, and sends only the job id to SQS. The expensive LLM extraction
    # work happens later in ``worker_handler.py``.
    try:
        # Parse request body
        # API Gateway sends body as a string. Tests or direct invocation may pass
        # a dict, so support both forms.
        body_raw = event.get('body', '{}')
        body_obj = json.loads(body_raw) if isinstance(body_raw, str) else (body_raw or {})
        resume_text = (body_obj.get('resume_text') or '').strip()
        jd_text = (body_obj.get('jd_text') or '').strip()

        # Both sides are required because the matcher needs a resume and a job
        # description to compute any useful overlap metrics.
        if not resume_text or not jd_text:
            return create_response(400, {
                'message': 'Both resume_text and jd_text are required'
            })

        # Normalize whitespace-trimmed input before hashing so duplicate
        # requests map to the same cache key and job lookup.
        normalized_payload = {
            'resume_text': resume_text,
            'jd_text': jd_text,
        }
        cache_key = calculate_hash(json.dumps(normalized_payload, ensure_ascii=False))
        
        # Check if a job already exists with this cache_key using GSI
        # Check whether an equivalent request already exists using the
        # cache-key-index GSI. This prevents queueing duplicate work when the
        # same user submits the same resume/JD pair multiple times.
        try:
            response = jobs_table.query(
                IndexName='cache-key-index',
                KeyConditionExpression=Key('cache_key').eq(cache_key),
                Limit=1  # We only need to know if one exists
            )
            
            if response.get('Items'):
                # Job already exists with this cache_key
                # If a non-failed job already exists, return it. It may be
                # PENDING, PROCESSING, or SUCCEEDED; clients can poll the job id.
                existing_job = response['Items'][0]
                status = (existing_job.get('status') or '').upper()
                if status != 'FAILED':
                    return create_response(200, {
                        'message': 'Job already exists for this request',
                        'data': {
                            "job": existing_job,
                        }
                    })
                # If the existing job is FAILED, proceed to create a new job
                # A FAILED job is not reused because the user likely expects a
                # fresh retry, not the previous error state.
                
        except ClientError as e:
            print(f"DynamoDB GSI query error: {str(e)}")
            # Continue with creating new job if query fails
            # Do not fail the request only because the dedupe lookup failed.
            # Creating a new job is safer than dropping a valid request.
        
        # Generate unique job ID
        # Generate a public job id that the client can poll later.
        job_id = str(uuid.uuid4())
        
        # Create job record in DynamoDB
        # Store the full normalized payload in DynamoDB. SQS carries only the
        # job id, keeping queue messages small and avoiding duplicated payloads.
        job = {
            'job_id': job_id,
            'status': 'PENDING',
            'payload': normalized_payload,
            'cache_key': cache_key,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Store job in DynamoDB
        # Persist before enqueueing so the worker can read the job when it
        # receives the SQS message.
        jobs_table.put_item(Item=job)
        
        # Send message to SQS queue
        # Queue async processing. The worker Lambda is subscribed to this queue.
        sqs.send_message(
            QueueUrl=JOB_QUEUE_URL,
            MessageBody=json.dumps({
                'job_id': job_id
            })
        )
        
        return create_response(202, {
            'message': 'Job created and queued for processing',
            'data': {
                "job": job
            }
        })
        
    except json.JSONDecodeError:
        # Invalid request JSON is a client error, not a Lambda/server failure.
        return create_response(400, {
            'message': 'Invalid JSON in request body'
        })
    except Exception as e:
        # Keep the error visible during development. Production code would
        # normally avoid returning raw exception text to clients.
        print(f"Error processing job: {str(e)}")
        return create_response(500, {
            'message': 'Internal server error',
            'error': str(e)
        })


def handle_get_job_status(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle GET /jobs/{job_id} requests - return job status."""

    # Clients call this endpoint after ``POST /process`` to observe the async job
    # state and eventually retrieve the worker result.
    try:
        # Extract job_id from path parameters
        # API Gateway maps the {job_id} route parameter into pathParameters.
        path_parameters = event.get('pathParameters', {})
        job_id = path_parameters.get('job_id')
        
        if not job_id:
            return create_response(400, {
                'message': 'Missing job_id in path parameters'
            })
        
        # Get job from DynamoDB
        try:
            # Get job from DynamoDB
            # Look up the job by primary key. The returned item includes status,
            # payload, timestamps, and result/error once the worker updates it.
            response = jobs_table.get_item(Key={'job_id': job_id})
            
            if 'Item' not in response:
                return create_response(404, {
                    'message': 'Job not found'
                })
            
            job = response['Item']
            
            # Return job status
            return create_response(200, {
                "message": "Job status retrieved",
                "data": {
                    "job": job
                }
            })
            
        except ClientError as e:
            # DynamoDB errors are infrastructure/server failures from the
            # client's perspective.
            print(f"DynamoDB error: {str(e)}")
            return create_response(500, {
                'message': 'Database error',
                'error': str(e),
            })
            
    except Exception as e:
        print(f"Error getting job status: {str(e)}")
        return create_response(500, {
            'message': 'Internal server error',
            'error': str(e),
        })


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler function for quick processing."""

    print(f"Received event: {json.dumps(event)}")
    
    # Handle different HTTP methods and paths
    # The CDK stack wires multiple API Gateway routes to this one Lambda, so
    # routing is done by method/path here.
    http_method = event.get('httpMethod', 'GET')
    path = event.get('path', '/')
    
    try:
        # Write new job to DynamoDB
        if http_method == 'POST' and path == '/process':
            return handle_post_process(event)
        # Get status update (read) from DynamoDB
        elif http_method == 'GET' and path.startswith('/jobs/'):
            return handle_get_job_status(event)
        else:
            return create_response(405, {
                'message': 'Method not allowed',
                'error': f'HTTP method {http_method} on path {path} is not supported'
            })
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return create_response(500, {
            'message': 'Internal server error',
            'error': str(e)
        })
