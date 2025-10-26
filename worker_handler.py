import json
import time
import boto3
import os
import random
from datetime import datetime, timezone
from typing import Dict, Any
from botocore.exceptions import ClientError

from utils.hash import calculate_hash
from utils.s3_cloudfront import get_cached_data, store_in_s3, json_dumps_decimal
from utils.decimal import convert_floats_to_decimal
from nodes.keyword_extractor_graph import run_phase1

# Initialize AWS clients
dynamodb = boto3.resource("dynamodb")

# Environment variables
JOBS_TABLE_NAME = os.environ["JOBS_TABLE_NAME"]
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.5").strip())

# DynamoDB table
jobs_table = dynamodb.Table(JOBS_TABLE_NAME)


def update_job_status(
    job_id: str, status: str, result: Dict[str, Any] = None, error: str = None
) -> None:
    """Update job status in DynamoDB."""
    try:
        update_expression = "SET #status = :status, updated_at = :updated_at"
        expression_attribute_names = {"#status": "status"}
        expression_attribute_values = {
            ":status": status,
            ":updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if result:
            update_expression += ", #result = :result"
            expression_attribute_names["#result"] = "result"
            expression_attribute_values[":result"] = convert_floats_to_decimal(result)

        if error:
            update_expression += ", #error = :error"
            expression_attribute_names["#error"] = "error"
            expression_attribute_values[":error"] = error

        jobs_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values,
        )

    except ClientError as e:
        print(f"Error updating job {job_id}: {str(e)}")
        raise


def get_job_payload(job_id: str) -> Dict[str, Any]:
    """Get job payload from DynamoDB."""
    try:
        response = jobs_table.get_item(Key={"job_id": job_id})

        if "Item" not in response:
            raise ValueError(f"Job {job_id} not found")

        payload = response["Item"].get("payload")
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except Exception:
                # Fallback to raw string payload
                return {"raw": payload}
        return payload

    except ClientError as e:
        print(f"Error getting job {job_id}: {str(e)}")
        raise


def _skill_to_dict(sk) -> Dict[str, Any]:
    return {
        "name": sk.name,
        "category": (
            sk.category.value if hasattr(sk.category, "value") else str(sk.category)
        ),
        "importance": (
            sk.importance.value
            if hasattr(sk.importance, "value")
            else str(sk.importance)
        ),
        "yoe": sk.get_yoe(),
        "proficiency": (
            sk.proficiency.value
            if hasattr(sk.proficiency, "value")
            else str(sk.proficiency)
        ),
        "referenced_sentence_ids": sk.referenced_sentence_ids or [],
    }


def process_job(job_id: str, payload: Any) -> Dict[str, Any]:
    """Process the job: run Phase 1 and Phase 2 for resume and JD, then compute matching score."""
    print(f"Processing job {job_id} with payload: {payload}")

    # Normalize payload
    if isinstance(payload, dict):
        resume_text = (payload.get("resume_text") or "").strip()
        jd_text = (payload.get("jd_text") or "").strip()
    elif isinstance(payload, str):
        try:
            obj = json.loads(payload)
            resume_text = (obj.get("resume_text") or "").strip()
            jd_text = (obj.get("jd_text") or "").strip()
        except Exception:
            resume_text = payload
            jd_text = ""
    else:
        resume_text = ""
        jd_text = ""

    if not resume_text or not jd_text:
        raise ValueError("Both resume_text and jd_text must be provided")

    # Calculate cache key consistently with quick_handler
    normalized_payload = {"resume_text": resume_text, "jd_text": jd_text}
    cache_key = calculate_hash(json.dumps(normalized_payload, ensure_ascii=False))
    print(f"Cache key for job {job_id}: {cache_key}")

    # Check CloudFront/S3 cache
    existing_cache_data = get_cached_data(cache_key)
    if existing_cache_data:
        print(f"Job {job_id} result found in cache, skipping processing")
        update_job_status(job_id, "SUCCEEDED", result=existing_cache_data)
        print(f"Job {job_id} completed from cache")
        return existing_cache_data

    # Update job status to PROCESSING
    update_job_status(job_id, "PROCESSING")

    # Run Phase 1 and Phase 2 for JD
    print(f"Running Phase 1 for JD...")
    jd_phase1_state = run_phase1(jd_text)

    # TODO: Implement Phase 2

    # Run Phase 1 and Phase 2 for Resume
    print(f"Running Phase 1 for Resume...")
    resume_phase1_state = run_phase1(resume_text)

    # TODO: Implement Phase 2

    # TODO: Implement Matcher


    # TODO: Output matching results
    processed_result = {
        "cache_key": cache_key,
        "source": "processor",
        "input_data": normalized_payload,
        "extractions": {
            "jd_sentences": jd_phase1_state.sentences,
            "resume_sentences": resume_phase1_state.sentences,
        },
        "matching": {
        },
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Store result in S3 and get CloudFront URL
    try:
        cloudfront_url = store_in_s3(cache_key, processed_result)
        processed_result["cloudfront_url"] = cloudfront_url
        print(f"Job {job_id} result stored in S3 with CloudFront URL: {cloudfront_url}")
    except Exception as e:
        print(f"Warning: Could not store result in S3 for job {job_id}: {e}")

    update_job_status(job_id, "SUCCEEDED", result=processed_result)
    print(f"Job {job_id} completed successfully")
    return processed_result


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler function for worker processing."""
    print(f"Worker received event: {json.dumps(event)}")

    # Process SQS messages
    records = event.get("Records", [])

    for record in records:
        try:
            # Parse SQS message
            message_body = json.loads(record["body"])
            job_id = message_body.get("job_id")

            if not job_id:
                print(f"No job_id found in message: {message_body}")
                continue

            print(f"Processing job: {job_id}")

            # Get job payload from DynamoDB
            try:
                payload = get_job_payload(job_id)
            except ValueError as e:
                print(f"Job not found: {str(e)}")
                continue
            except Exception as e:
                print(f"Error getting job payload: {str(e)}")
                update_job_status(
                    job_id, "FAILED", error=f"Error retrieving job: {str(e)}"
                )
                continue

            # Process the job
            try:
                result = process_job(job_id, payload)
                print(f"Job {job_id} processed successfully")

            except Exception as e:
                print(f"Error processing job {job_id}: {str(e)}")
                update_job_status(job_id, "FAILED", error=str(e))

        except json.JSONDecodeError as e:
            print(f"Error parsing SQS message: {str(e)}")
            continue
        except Exception as e:
            print(f"Unexpected error processing record: {str(e)}")
            continue

    return {
        "statusCode": 200,
        "body": json_dumps_decimal(
            {
                "message": f"Processed {len(records)} records",
                "data": {"records_processed": len(records)},
            }
        ),
    }
