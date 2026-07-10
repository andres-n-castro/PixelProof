from celery_app import celery_app
from dotenv import load_dotenv
from torch import Tensor
from inference_runtime import get_model, get_detector
from database.repositories.video_repository import update_video, get_video
from database.schemas import VideoUpdate
import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
from typing import Any
from database.database import SessionLocal
import uuid

load_dotenv()

#completes preprocessing as well as running the model
@celery_app.task
def run_model(video_id: uuid.UUID) -> str | None:
  db = SessionLocal()
  try:
    print(f"[run_model] starting task for video_id={video_id}")
    db_obj = get_video(db=db, video_id=video_id)
    print(f"[run_model] loaded db row with path={db_obj.video_path}")

    preprocessed_input = process_single_video(db_obj.video_path)
    print(f"[run_model] process_single_video returned {'tensor' if preprocessed_input is not None else 'None'}")

    if preprocessed_input is None:
      video_update = VideoUpdate(
        status="failed"
      )
      _ = update_video(db=db, video=video_update, video_id=video_id)

      raise Exception("processed input returned none")

    model, device = get_model()
    preprocessed_input = preprocessed_input.unsqueeze(0).to(device)

    logit = model(preprocessed_input)
    prediction = torch.argmax(logit, dim=1).item()

    video_update = VideoUpdate(
        status="done",
        prediction="real" if prediction == 0 else "fake"
      )
    

    updated_video_row = update_video(db=db, video=video_update, video_id=video_id)

    return f"successfully processed video!, status: {updated_video_row.status}"
  finally:
    db.close()

def process_single_video(video_path: str) -> Tensor | None:
  print(f"[process_single_video] starting for path={video_path}")

  frame_tensor_list = []
  loaded_video = cv.VideoCapture(video_path)
  detector = get_detector()
  contains_face = False
  valid_face_count = 0
  n_frames = 20

  if not loaded_video.isOpened():
    print(f"Error: could not open {video_path}")
    loaded_video.release()
    return None
  
  video_frame_count = int(loaded_video.get(cv.CAP_PROP_FRAME_COUNT))
  print(f"[process_single_video] video_frame_count={video_frame_count}")

  if video_frame_count < n_frames:
    print(f"[process_single_video] video shorter than required n_frames={n_frames}")
    loaded_video.release()
    return None

  frame_idxs = set(np.linspace(0, video_frame_count-1, n_frames, dtype=int))
  print(f"[process_single_video] selected {len(frame_idxs)} frame indices")

  for idx in range(video_frame_count):
    ret, img = loaded_video.read()

    if not ret:
      print(f"[process_single_video] stopped reading at frame idx={idx}")
      break
    
    if idx not in frame_idxs:
      continue

    print(f"[process_single_video] processing selected frame idx={idx}")
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #use mediapipe to detect faces
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    print(f"[process_single_video] running detector on frame idx={idx}")
    result = detector.detect(mp_image)
    print(f"[process_single_video] detector found {len(result.detections)} faces on frame idx={idx}")

    #checks to see if current vido contains a frame that has a face
    if not contains_face and len(result.detections) > 0:
      contains_face = True
    
    if len(result.detections) > 0:
      valid_face_count += 1
      largest_face_area = 0
      largest_face_idx = 0
      for i, detection in enumerate(result.detections):
        current_face_area = detection.bounding_box.width * detection.bounding_box.height
        if current_face_area > largest_face_area:
          largest_face_area = current_face_area
          largest_face_idx = i
      
      largest_face = result.detections[largest_face_idx]
      largest_face_bb = largest_face.bounding_box


      new_face_bb = prepare_face_crop(largest_face_bb, margin_hyperparam=0.20, frame_shape=rgb_img.shape)
      print(f"[process_single_video] prepared crop box={new_face_bb}")

      #redo the extraction
      #obtain corner coordinates for the face using new B.B.
      orig_x = new_face_bb[0]
      orig_y = new_face_bb[1]

      second_x = orig_x +new_face_bb[2] 
      second_y = orig_y + new_face_bb[3]
      print(f"[process_single_video] crop corners=({orig_x}, {orig_y}) -> ({second_x}, {second_y})")

      #extracted face frame
      face_frame = rgb_img[orig_y:second_y, orig_x:second_x]
      print(f"[process_single_video] face_frame shape={face_frame.shape}")

      if face_frame.size == 0:
          continue

      #resizes using inter cubic interpolation for upsampling (making the face frame bigger) since its much smoother than linear interpolation
      resized_face_frame = cv.resize(face_frame, (224,224), interpolation=cv.INTER_CUBIC)
      print("[process_single_video] resized face frame")

      #convert face frame into tensor
      face_tensor = torch.from_numpy(resized_face_frame).permute(2, 0, 1).float() / 255.0

      #append tensor to list
      frame_tensor_list.append(face_tensor)
      print(f"[process_single_video] appended tensor count={len(frame_tensor_list)}")

    #case where there are no faces in frame now check if there is a valid previous face to use, if not then continue
    else:
      continue

  #unload the current video after preprocessing on it is done
  loaded_video.release()
  print(f"[process_single_video] collected {len(frame_tensor_list)} frame tensors")

  if len(frame_tensor_list) == 0:
    print("[process_single_video] no valid frame tensors collected")
    return None

  if contains_face and len(frame_tensor_list) > 0:
    #pads the tensor list with the last tensor obtained in the case that there were < n_frame tensors retieved from the video
    #ensures all frame tensors contain the same shape before stacking
    while len(frame_tensor_list) < n_frames:
      frame_tensor_list.append(frame_tensor_list[-1])
      
    #convert list of tensors into stack of tensors

  preprocessed_video = torch.stack(frame_tensor_list).contiguous()
  print(f"[process_single_video] returning tensor with shape={tuple(preprocessed_video.shape)}")
  return preprocessed_video

def prepare_face_crop(orig_bb: Any, margin_hyperparam: float, frame_shape: tuple) -> list[int]:
  expanded_face_bb = margin_expansion(orig_bb, margin_hyperparam, frame_shape)
  
  #resize face_bb into a square
  if expanded_face_bb[2] < expanded_face_bb[3]:
    center_x = (expanded_face_bb[2] / 2) + expanded_face_bb[0]
    new_width = expanded_face_bb[3]

    final_origin_x = max(0, int(center_x - new_width / 2))
    final_width = min(new_width, frame_shape[1] - final_origin_x)

    return [final_origin_x, expanded_face_bb[1], final_width, expanded_face_bb[3]]
  
  elif expanded_face_bb[2] > expanded_face_bb[3]:
    center_y = (expanded_face_bb[3] / 2) + expanded_face_bb[1]
    new_height = expanded_face_bb[2]

    final_origin_y = max(0, int(center_y - new_height / 2))
    final_height = min(new_height, frame_shape[0] - final_origin_y)

    return [expanded_face_bb[0], final_origin_y, expanded_face_bb[2], final_height]
  
  else: return expanded_face_bb

def margin_expansion(orig_bb: Any, margin_hyperparam: float, frame_shape: tuple) -> list[int]:
  center_x = (orig_bb.width / 2) + orig_bb.origin_x
  center_y = (orig_bb.height / 2) + orig_bb.origin_y

  new_width = orig_bb.width * (1 + margin_hyperparam)
  new_height = orig_bb.height * (1 + margin_hyperparam)

  new_origin_x = int(center_x - (new_width / 2))
  new_origin_y = int(center_y - (new_height / 2))

  final_origin_x = max(0, new_origin_x)
  final_origin_y = max(0, new_origin_y)
  final_width = min(new_width, frame_shape[1] - final_origin_x)
  final_height = min(new_height, frame_shape[0] - final_origin_y)

  return [final_origin_x, final_origin_y, int(final_width), int(final_height)]