# PixelProof

PixelProof is a full-stack deepfake video detection project built around a custom computer vision pipeline. The backend accepts uploaded `.mp4` videos, stores metadata in PostgreSQL, queues inference jobs with Celery and Redis, and uses a ResNet18 + LSTM model to classify videos as `real` or `fake`.

## Project Status

- Backend API is implemented with FastAPI
- Background inference is handled with Celery
- Video metadata is stored in PostgreSQL via SQLAlchemy
- Authentication uses JWT bearer tokens
- The deepfake classifier has been trained and evaluated
- The frontend exists as a Next.js scaffold and is not yet fully integrated with the backend

## Model Summary

The current classifier processes 20 face-cropped frames per video sample:

1. A face detector extracts the largest face from selected frames
2. Each frame is passed through a ResNet18 backbone
3. Frame features are fed into an LSTM head
4. The model predicts whether the video is `real` or `fake`

### Current evaluation snapshot

Held-out test results from the current trained checkpoint:

- Accuracy: `0.9632`
- Recall: `0.9675`
- Precision: `0.9613`
- F1 score: `0.9644`
- Loss: `0.1111`

Confusion matrix counts:

- Real predicted as Real: `139`
- Real predicted as Fake: `6`
- Fake predicted as Real: `5`
- Fake predicted as Fake: `149`

## Repository Structure

```text
PixelProof/
├── backend/
│   ├── app/
│   │   ├── api/                  # FastAPI route handlers
│   │   ├── auth/                 # JWT auth helpers
│   │   ├── database/             # SQLAlchemy models, schemas, repositories
│   │   ├── scripts/              # training, testing, dataset, model code
│   │   ├── celery_app.py         # Celery app definition
│   │   ├── celery_tasks.py       # background inference task
│   │   ├── inference_runtime.py  # detector + model loading for inference
│   │   └── main.py               # FastAPI app entrypoint
│   └── requirements.txt
├── frontend/
│   ├── app/                      # Next.js app router
│   ├── public/
│   └── package.json
└── requirements.txt
```

## Tech Stack

### Backend

- FastAPI
- SQLAlchemy
- PostgreSQL
- Celery
- Redis
- PyJWT
- MediaPipe
- OpenCV
- PyTorch

### Frontend

- Next.js
- React
- TypeScript

## Requirements

Before running the project locally, make sure you have:

- Python 3.12+
- Node.js 20+
- PostgreSQL
- Redis


## Known Limitations

- The frontend is still mostly scaffolded and not yet connected to the API
- The backend currently uses simple local file storage for uploaded videos

