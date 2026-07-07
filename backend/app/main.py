from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scripts.model import DeepfakeClassifier
import torch
from database.database import get_db

load_dotenv()

app = FastAPI()

origins = [
  "https://localhost:3000"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

#load in mediapipe detector
base_options = python.BaseOptions(model_asset_path='app/scripts/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

#instantiate custom deepfake classifier model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepfakeClassifier(
  weights=None,
  input_size=512,
  hidden_size=256,
  device=device
)

#make sure to provide path for the best state dict containing the weights for the model for actual predicitng
state_dict = torch.load("scripts/training_outputs/best_state_dict.pt", map_location=device)
model.load_state_dict(state_dict=state_dict)
model.to(device=device)
model.eval()

@app.get("/")
def root():
  return {"status : online"}
