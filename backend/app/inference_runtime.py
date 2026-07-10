from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scripts.model import DeepfakeClassifier
import torch
import os

_detector = None
_detector_pid = None

def get_detector():
  global _detector
  global _detector_pid

  current_pid = os.getpid()
  if _detector is None or _detector_pid != current_pid:
    base_options = python.BaseOptions(model_asset_path='scripts/blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    _detector = vision.FaceDetector.create_from_options(options)
    _detector_pid = current_pid

  return _detector

_model = None
_model_pid = None
_device = None

def get_model():
  global _model
  global _model_pid
  global _device

  current_pid = os.getpid()
  if _model is None or _model_pid != current_pid:
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = DeepfakeClassifier(
      weights=None,
      input_size=512,
      hidden_size=256,
      device=_device
    )
    state_dict = torch.load("scripts/training_outputs/best_state_dict.pt", map_location=_device)
    _model.load_state_dict(state_dict=state_dict)
    _model.to(device=_device)
    _model.eval()
    _model_pid = current_pid

  return _model, _device