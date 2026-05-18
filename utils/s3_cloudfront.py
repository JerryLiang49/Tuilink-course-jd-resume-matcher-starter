"""S3 and CloudFront helpers for cached worker results."""

import json
import time
import boto3
import os
import requests
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError
from utils.decimal import json_dumps_decimal

# Initialize AWS clients
# Module-level clients are reused by warm Lambda containers.
s3_client = boto3.client('s3')
cloudfront_client = boto3.client('cloudfront')

# Environment variables
# Cache configuration injected by the infra stack. Results are stored in S3 and
# served through CloudFront using the deterministic cache key.
CACHE_BUCKET_NAME = os.environ['CACHE_BUCKET_NAME']
CLOUDFRONT_DOMAIN = os.environ['CLOUDFRONT_DOMAIN']
CLOUDFRONT_URL = os.environ['CLOUDFRONT_URL']


def get_cached_data(hash_key: str) -> Optional[Dict[str, Any]]:
    """Check if data exists in CloudFront cache and return it if found."""

    # CloudFront is checked first because that is the public, fast path clients
    # and workers should prefer. If CloudFront has a temporary network issue,
    # the function falls back to direct S3 access.
    try:
        # Try to retrieve from CloudFront first
        # The worker writes results as <hash>.json, so cached URLs are
        # deterministic and can be reconstructed without another lookup.
        cloudfront_url = f"{CLOUDFRONT_URL}/{hash_key}.json"
        
        response = requests.get(cloudfront_url, timeout=10)
        
        if response.status_code == 200:
            # Cache hit: skip expensive LLM extraction/matching.
            data = response.json()
            print(f"Cache hit from CloudFront for key: {hash_key}")
            return data
        elif response.status_code == 404:
            # Cache miss: caller should continue normal processing.
            print(f"Cache miss from CloudFront for key: {hash_key}")
            return None
        else:
            print(f"CloudFront returned status {response.status_code} for key: {hash_key}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving from CloudFront: {e}")
        # Fallback to S3 direct access in case of CloudFront issues
        # Fallback to S3 direct access in case CloudFront is unavailable or
        # propagation has not completed yet.
        try:
            print(f"Falling back to S3 direct access for key: {hash_key}")
            response = s3_client.get_object(
                Bucket=CACHE_BUCKET_NAME,
                Key=f"{hash_key}.json"
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            return data
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise
    except Exception as e:
        print(f"Unexpected error checking cache: {e}")
        return None


def check_s3_exists(hash_key: str) -> bool:
    """Check if data exists in S3 (used by worker for cache validation)."""

    try:
        s3_client.head_object(
            Bucket=CACHE_BUCKET_NAME,
            Key=f"{hash_key}.json"
        )
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def store_in_s3(hash_key: str, data: Dict[str, Any]) -> str:
    """Store data in S3 and return the CloudFront URL."""

    s3_key = f"{hash_key}.json"
    
    # Store in S3
    # Write JSON result under a deterministic key so identical payloads can be
    # retrieved later without recomputing.
    s3_client.put_object(
        Bucket=CACHE_BUCKET_NAME,
        Key=s3_key,
        Body=json_dumps_decimal(data),
        ContentType='application/json'
    )
    
    # Create CloudFront invalidation to ensure fresh content
    # Invalidate the exact object path so CloudFront serves this new result even
    # if a previous failed/stale object existed with the same key.
    try:
        cloudfront_client.create_invalidation(
            DistributionId=os.environ.get('CLOUDFRONT_DISTRIBUTION_ID', ''),
            InvalidationBatch={
                'Paths': {
                    'Quantity': 1,
                    'Items': [f'/{s3_key}']
                },
                'CallerReference': f"cache-store-{hash_key}-{int(time.time())}"
            }
        )
        print(f"CloudFront invalidation created for key: {hash_key}")
    except Exception as e:
        print(f"Warning: Could not create CloudFront invalidation: {e}")
    
    # Return CloudFront URL
    cloudfront_url = f"{CLOUDFRONT_URL}/{s3_key}"
    print(f"Data stored in S3 with CloudFront URL: {cloudfront_url}")
    return cloudfront_url


def get_cloudfront_url(hash_key: str) -> str:
    """Get CloudFront URL for a given hash key."""

    return f"{CLOUDFRONT_URL}/{hash_key}.json"
