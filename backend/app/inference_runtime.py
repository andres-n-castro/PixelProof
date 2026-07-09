from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scripts.model import DeepfakeClassifier
import torch

#load in mediapipe detector
base_options = python.BaseOptions(model_asset_path='scripts/blaze_face_short_range.tflite')
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