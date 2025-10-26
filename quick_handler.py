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
dynamodb = boto3.resource('dynamodb')
sqs = boto3.client('sqs')

# Environment variables
JOBS_TABLE_NAME = os.environ['JOBS_TABLE_NAME']
JOB_QUEUE_URL = os.environ['JOB_QUEUE_URL']

# DynamoDB table
jobs_table = dynamodb.Table(JOBS_TABLE_NAME)


def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standardized API Gateway response."""
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
    try:
        # Parse request body
        body_raw = event.get('body', '{}')
        body_obj = json.loads(body_raw) if isinstance(body_raw, str) else (body_raw or {})
        resume_text = (body_obj.get('resume_text') or '').strip()
        jd_text = (body_obj.get('jd_text') or '').strip()
        if not resume_text or not jd_text:
            return create_response(400, {
                'message': 'Both resume_text and jd_text are required'
            })

        normalized_payload = {
            'resume_text': resume_text,
            'jd_text': jd_text,
        }
        cache_key = calculate_hash(json.dumps(normalized_payload, ensure_ascii=False))
        
        # Check if a job already exists with this cache_key using GSI
        try:
            response = jobs_table.query(
                IndexName='cache-key-index',
                KeyConditionExpression=Key('cache_key').eq(cache_key),
                Limit=1  # We only need to know if one exists
            )
            
            if response.get('Items'):
                # Job already exists with this cache_key
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
                
        except ClientError as e:
            print(f"DynamoDB GSI query error: {str(e)}")
            # Continue with creating new job if query fails
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job record in DynamoDB
        job = {
            'job_id': job_id,
            'status': 'PENDING',
            'payload': normalized_payload,
            'cache_key': cache_key,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Store job in DynamoDB
        jobs_table.put_item(Item=job)
        
        # Send message to SQS queue
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
        return create_response(400, {
            'message': 'Invalid JSON in request body'
        })
    except Exception as e:
        print(f"Error processing job: {str(e)}")
        return create_response(500, {
            'message': 'Internal server error',
            'error': str(e)
        })

def handle_get_job_status(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle GET /jobs/{job_id} requests - return job status."""
    try:
        # Extract job_id from path parameters
        path_parameters = event.get('pathParameters', {})
        job_id = path_parameters.get('job_id')
        
        if not job_id:
            return create_response(400, {
                'message': 'Missing job_id in path parameters'
            })
        
        # Get job from DynamoDB
        try:
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
    http_method = event.get('httpMethod', 'GET')
    path = event.get('path', '/')
    
    try:
        if http_method == 'POST' and path == '/process':
            return handle_post_process(event)
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