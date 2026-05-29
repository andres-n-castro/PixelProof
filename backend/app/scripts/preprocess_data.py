import torch
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse
import os
from typing import Any
from multiprocessing import Pool

detector = None
n_frames = None
label = None
samples_dest_dir = None
data_srce_dir = None
margin_hyperparam = None
std_img_size = None

#create a initialization function for a worker to call when its instantiated
def init_worker(worker_n_frames: int, worker_label: int, worker_samples_dest_dir: str, worker_data_srce_dir: str, worker_margin_hyperparam: float, worker_std_img_size: int):
  global detector
  global n_frames
  global label
  global samples_dest_dir
  global data_srce_dir
  global std_img_size
  global margin_hyperparam

  n_frames = worker_n_frames
  label = worker_label
  samples_dest_dir = worker_samples_dest_dir
  data_srce_dir = worker_data_srce_dir
  std_img_size = (worker_std_img_size, worker_std_img_size)
  margin_hyperparam = worker_margin_hyperparam

  #load in mediapipe detector
  base_options = python.BaseOptions(model_asset_path='detector.tflite')
  options = vision.FaceDetectorOptions(base_options=base_options)
  detector = vision.FaceDetector.create_from_options(options)

  print("Mediapipe Face Detector Initialized!")

def process_videos(data_srce_dir: str, n_frames: int, label: int, samples_dest_dir: str, margin_hyperparam: float, std_img_size: int) -> None:

  video_file_names = [file for file in os.listdir(data_srce_dir) if file.endswith(".mp4")]
  
  with Pool(processes=5, initializer=init_worker, initargs=(n_frames, label, samples_dest_dir, data_srce_dir, margin_hyperparam, std_img_size) ) as p:
    p.map(process_single_video, video_file_names)
  
  return None

def process_single_video(video_name: str) -> None:
  global detector

  frame_tensor_list = []
  video_path = os.path.join(data_srce_dir, video_name)
  loaded_video = cv.VideoCapture(video_path)
  contains_face = False
  valid_face_count = 0

  if not loaded_video.isOpened():
    print(f"Error: could not open {video_name}")
    loaded_video.release()
    return None
  
  video_frame_count = int(loaded_video.get(cv.CAP_PROP_FRAME_COUNT))

  if video_frame_count < n_frames:
    loaded_video.release()
    return None

  frame_idxs = set(np.linspace(0, video_frame_count-1, n_frames, dtype=int))

  for idx in range(video_frame_count):
    ret, img = loaded_video.read()

    if not ret:
      break
    
    if idx not in frame_idxs:
      continue

    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #use mediapipe to detect faces
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    result = detector.detect(mp_image)

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


      new_face_bb = prepare_face_crop(largest_face_bb, margin_hyperparam=margin_hyperparam, frame_shape=rgb_img.shape)

      #redo the extraction
      #obtain corner coordinates for the face using new B.B.
      orig_x = new_face_bb[0]
      orig_y = new_face_bb[1]

      second_x = orig_x +new_face_bb[2] 
      second_y = orig_y + new_face_bb[3]

      #extracted face frame
      face_frame = rgb_img[orig_y:second_y, orig_x:second_x]

      if face_frame.size == 0:
          continue

      #resizes using inter cubic interpolation for upsampling (making the face frame bigger) since its much smoother than linear interpolation
      resized_face_frame = cv.resize(face_frame, std_img_size, interpolation=cv.INTER_CUBIC)

      #convert face frame into tensor
      face_tensor = torch.from_numpy(resized_face_frame).permute(2, 0, 1).float() / 255.0

      #append tensor to list
      frame_tensor_list.append(face_tensor)

    #case where there are no faces in frame now check if there is a valid previous face to use, if not then continue
    else:
      continue

  #unload the current video after preprocessing on it is done
  loaded_video.release()

  if contains_face and len(frame_tensor_list) > 0:
    #pads the tensor list with the last tensor obtained in the case that there were < n_frame tensors retieved from the video
    #ensures all frame tensors contain the same shape before stacking
    while len(frame_tensor_list) < n_frames:
      frame_tensor_list.append(frame_tensor_list[-1])
      
    #convert list of tensors into stack of tensors
    sample = {"frames": torch.stack(frame_tensor_list).contiguous(),
               "valid_frame_count": valid_face_count,
               "label": label,
               "video_name": video_name
              }
    
    video_name_tokens = video_name.split(".")
    torch.save(sample, os.path.join(samples_dest_dir, f"processed_{video_name_tokens[0]}.pt"))

    print(f"successfully processed video: {video_name}")
  else:
    print(f"unsuccessfully processed video: {video_name}")

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
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("num_frames", metavar="number_frames_per_video", type=int, help="int that decides the number of frames to extract from a video")
  parser.add_argument("std_img_size", metavar="standardized_image_size", type=int, help="is the standard image size for the frame tensors")
  parser.add_argument("real_data_srce_dir", metavar="real_data_source_diretory", type=str, help="the source directory for the original dataset")
  parser.add_argument("fake_data_srce_dir", metavar="fake_data_source_diretory", type=str, help="the source directory for the deepfake dataset")
  parser.add_argument("real_samples_dest_dir", metavar="real_samples_destination_directory", type=str, help="the destination directory for the preprocessed original samples")
  parser.add_argument("fake_samples_dest_dir", metavar="fake_samples_destination_directory", type=str, help="the destination directory for the preprocessed fake samples")
  parser.add_argument("margin_hyperparam", metavar="margin_hyperparameter", type=float, help="is the tuned hyperparameter for margin expansion on the retrieved largest face bounding box")
  


  args = parser.parse_args()
  num_frames = args.num_frames
  std_img_size = args.std_img_size
  real_data_srce_dir = args.real_data_srce_dir
  real_samples_dest_dir = args.real_samples_dest_dir
  fake_data_srce_dir = args.fake_data_srce_dir
  fake_samples_dest_dir = args.fake_samples_dest_dir
  margin_hyperparam = args.margin_hyperparam


  try:
    #for real dataset
    process_videos(n_frames=num_frames, std_img_size=std_img_size, samples_dest_dir=real_samples_dest_dir, data_srce_dir=real_data_srce_dir, margin_hyperparam=margin_hyperparam, label=0)
    print("finished processing original video files\n")
  except Exception as e:
    print(f"error: {e}")
    exit()
  
  try:
    #for deepfake dataset
    process_videos(n_frames=num_frames, std_img_size=std_img_size, samples_dest_dir=fake_samples_dest_dir, data_srce_dir=fake_data_srce_dir, margin_hyperparam=margin_hyperparam, label=1)
    print("finished processing deepfake video files\n")
  except Exception as e:
    print(f"error: {e}")
    exit()
  

