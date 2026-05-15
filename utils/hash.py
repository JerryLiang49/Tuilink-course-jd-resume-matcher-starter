"""Hashing helpers used for request deduplication and result caching."""

import hashlib


def calculate_hash(data: str) -> str:
    """Calculate the deterministic cache key for a request payload.

    The quick Lambda and worker Lambda both use this function. If the normalized
    resume/JD payload is identical, the same SHA-256 value is produced, allowing
    duplicate jobs to reuse DynamoDB and S3/CloudFront cache entries.
    """

    return hashlib.sha256(data.encode('utf-8')).hexdigest()
