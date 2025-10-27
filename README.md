# JD Resume Matcher - Starter

An intelligent system for extracting skills from job descriptions and resumes and computing matching metrics. This starter intentionally leaves key pieces for you to implement, and includes an optional AWS async workflow scaffold.

## Overview

This project implements an extraction and matching workflow that:

1. Preprocesses raw text into sentences (implemented)
2. Extracts and merges skills with sentence references (to be implemented)
3. Validates and modifies skills for consistency (to be implemented)
4. Computes professional matching metrics (precision, recall, F1) (to be implemented)

## Project Structure

```
.
├── docs/                      # Mermaid charts for workflows
├── examples/                  # Sample inputs (e.g., job_description.md)
├── models/                    # Core data models
├── nodes/                     # Processing nodes and graph builders
├── utils/                     # Utility functions (LLM, cache, JSON, hash)
├── .env.example               # Example environment config
├── build.sh                   # Build script for dist/
├── extract.ipynb              # Local notebook to play around
├── quick_handler.py           # Web / Quick Lambda: create jobs and query status
├── requirements-compact.txt   # Compact dependencies for deployment
├── requirements.txt           # Full dependencies for local development
└── worker_handler.py          # Worker Lambda: SQS consumer to process the jobs
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file (or export in your deployment) with the following credentials and settings:

```bash
# OpenAI
OPENAI_API_KEY=<your-openai-api-key>

# LLM configuration
LLM_MODEL=gpt-4.1-mini
LLM_TEMPERATURE=0

# Caching (whether to cache LLM responses locally)
LLM_USE_CACHE=false

# Embeddings configuration
LLM_EMBEDDING_MODEL=text-embedding-3-small
```

## Build for Deployment

Use the build script to create a self-contained `dist/` folder ready for packaging and deployment.

1. Ensure your `.env` is present and set `LLM_USE_CACHE=false` for deployment (to avoid local cache writes).

2. Run the build script:

```bash
chmod +x ./build.sh
./build.sh
```

3. The script will create `./dist` containing:

- `quick_handler.py`
- `worker_handler.py`
- `requirements.txt` (copied from `requirements-compact.txt`)
- `models/`, `nodes/`, `utils/` (necessary files only)
- `__init__.py` files in Python package directories (`models/`, `nodes/`, `utils/`)

4. Deployment notes for your infra repo:

- Set your Lambda source path to the `dist` folder of this repo, e.g., `JD_RESUME_MATCHER_LAMBDA_SOURCE_PATH=../../course-jd-resume-matcher-starter/dist/`
- Use handler for the web / quick Lambda (triggered by API Gateway), e.g., `JD_RESUME_MATCHER_QUICK_HANDLER=quick_handler.handler`
- Use handler for the worker Lambda (consuming SQS events), e.g., `JD_RESUME_MATCHER_WORKER_HANDLER=worker_handler.handler`
- (Optional) Set your Lambda layer source path to the `dist` folder of this repo, e.g., `JD_RESUME_MATCHER_LAMBDA_LAYER_SOURCE_PATH=../../course-jd-resume-matcher-starter/dist/`
