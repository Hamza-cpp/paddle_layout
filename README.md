# Document Layout API

A Flask-based REST API for document layout analysis using PaddlePaddle's PP-DocLayout model.

## Overview

This API provides an endpoint for analyzing document layouts from uploaded images using PP-DocLayout-L, a model that can detect and classify document components such as text blocks, tables, figures, and more.

## API Endpoints

### Health Check

```
GET /health
```

Returns the health status of the API and whether GPU is enabled.

**Example Response:**

```json
{
  "status": "healthy",
  "gpu_enabled": false
}
```

### Document Layout Analysis

```
POST /predict
```

Analyzes the layout of a document from an uploaded file.

**Request:**

- Content-Type: multipart/form-data
- Body:
  - `file`: The document file to analyze

**Example Response:**

```json
{
  "message": "Prediction successful",
  "inference_time": 1.2345,
  "device": "CPU",
  "results": [
    {
      "page_index": 0,
      "data": [
        {
          "category": "text",
          "bbox": [100, 100, 500, 200],
          "score": 0.98
        },
        {
          "category": "table",
          "bbox": [100, 300, 500, 600],
          "score": 0.95
        }
      ]
    }
  ]
}
```

## Usage

1. Start the server:

   ```bash
   python app.py
   ```

2. The API will be available at <http://localhost:5000>

3. To analyze a document, send a POST request to the /predict endpoint with a file upload:

   ```bash
   curl -X POST -F "file=@example.pdf" http://localhost:5000/predict
   ```
