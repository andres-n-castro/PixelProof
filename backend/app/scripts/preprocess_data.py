import torch
import pickle
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse
import os
from typing import any

#TODO
def prepare_face_crop(face_bb: any, margin_hyperparam) -> list[int]:
  return 

def process_videos(n_frames: int, std_img_size: tuple[int,int], samples_dest_dir: str, data_srce_dir: str, label: int, margin_hyperparam: int) -> None:
  sample_count = 0
  base_options = python.BaseOptions(model_asset_path='detector.tflite')
  options = vision.FaceDetectorOptions(base_options=base_options)
  detector = vision.FaceDetector.create_from_options(options)
  video_count = 0
  video_file_names = os.listdir(data_srce_dir)

  for video in video_file_names:
    video_count += 1

    video_path = os.path.join(data_srce_dir, video)
    sample_tuple = process_single_video(video_name=video, video_path=video_path, n_frames=n_frames, detector=detector, margin_hyperparam=margin_hyperparam, std_img_size=std_img_size)


    if not sample_tuple:
      if video_count % 80 == 0:
        print(f"processed video: {video_count} but sample had no faces so it will not be used!")
      continue

    #create sample tuple
    sample_count += 1
    sample = {"frame_tensor": sample_tuple[0],
              "valid_frame_count": sample_tuple[1],
              "video_name": video,
              "label": label
             }

    #serialize sample dictionary using pickle
    #add serialized sample to samples_dest_dir folder
    with open(os.path.join(samples_dest_dir, f"sample_{sample_count}")) as file:
      pickle.dump(sample, protocol=pickle.HIGHEST_PROTOCOL, file=file)
    
    if video_count % 80 == 0:
        print(f"processed video: {video_count} and successfully uploaded sample!")
  
  return None


def process_single_video(video_name: str, video_path: str, n_frames: int, detector: any, margin_hyperparam: int, std_img_size: tuple[int, int]) -> tuple[torch.Tensor, int] | None:
  frame_tensor_list = []
  loaded_video = cv.VideoCapture(video_path)
  contains_face = False
  valid_face_count = 0

  if not loaded_video.isOpened():
    print(f"Error: could not open {video_name}")
    return
  
  video_frame_count = int(loaded_video.get(cv.CAP_PROP_FRAME_COUNT))
  frame_idx = video_frame_count//n_frames

  for idx in range(1, video_frame_count+1):
    ret, img = loaded_video.read()
    rbg_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #checks if frame if a valid nth frame to use
    if idx % frame_idx != 0:
      continue

    #use mediapipe to detect faces
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rbg_img)
    result = detector.detect(mp_image)

    #checks to see if current vido contains a frame that has a face
    if not contains_face and len(result.detections) > 0:
      contains_face = True
    
    if len(result.detections) > 0:
      valid_face_count += 1
      largest_face_area = 0
      largest_face_idx = -1
      for idx, detection in enumerate(result.detections):
        current_face_area = (detection.bounding_box.width - detection.bounding_box.origin_x) * (detection.bounding_box.height - detection.bounding_box.origin_y)
        if current_face_area > largest_face_area:
          largest_face_area = current_face_area
          largest_face_idx = idx
      
      largest_face = result.detections[largest_face_idx]
      largest_face_bb = largest_face.bounding_box


      new_face_bb = prepare_face_crop(largest_face_bb, margin_hyperparam=margin_hyperparam)

      #extract face from frame using new bounding box
      orig_x = new_face_bb[0]
      width = new_face_bb[1]
      orig_y = new_face_bb[2]
      height = new_face_bb[3]
      face_frame = img[orig_y:height+1, orig_x:width+1]

      #resizes using inter cubic interpolation for upsampling (making the face frame bigger) since its much smoother than linear interpolation
      resized_face_frame = cv.resize(face_frame, std_img_size, interpolation=cv.INTER_CUBIC)

      #convert face frame into tensor
      face_tensor = torch.from_numpy(resized_face_frame)

      #append tensor to list
      frame_tensor_list.append(face_tensor)

    #case where there are no faces in frame now check if there is a valid previous face to use, if not then continue
    else:
      continue
  
    #pads the tensor list with the last tensor obtained in the case that there were < n_frame tensors retieved from the video
    #ensures all frame tensors contain the same shape before stacking
    while len(frame_tensor_list) < n_frames:
      frame_tensor_list.append(frame_tensor_list[-1])
      
    #convert list of tensors into stack of tensors
    return tuple(torch.stack(frame_tensor_list), valid_face_count)

if __name__ == "__main__":
  parser = argparse.ArgumentParser
  parser.add_argument("num_frames", metavar="number_frames_per_video", type=int, help="int that decides the number of frames to extract from a video")
  parser.add_argument("std_img_size", metavar="standardized_image_size", type=int, help="is the standard image size for the frame tensors")
  parser.add_argument("real_data_srce_dir", metavar="real_data_source_diretory", type=str, help="the source directory for the original dataset")
  parser.add_argument("fake_data_srce_dir", metavar="fake_data_source_diretory", type=str, help="the source directory for the deepfake dataset")
  parser.add_argument("real_samples_dest_dir", metavar="real_samples_destination_directory", type=str, help="the destination directory for the preprocessed original samples")
  parser.add_argument("fake_samples_dest_dir", metavar="fake_samples_destination_directory", type=str, help="the destination directory for the preprocessed fake samples")
  parser.add_argument("margin_hyperparam", metavar="margin_hyperparameter", type=int, help="is the tuned hyperparameter for margin expansion on the retrieved largest face bounding box")
  


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
  
    #for deepfake dataset
    process_videos(n_frames=num_frames, std_img_size=std_img_size, samples_dest_dir=fake_samples_dest_dir, data_srce_dir=fake_data_srce_dir, margin_hyperparam=margin_hyperparam)
    print("finished processing deepfake video files\n")

  except Exception as e:
    print(f"error: {e}")
    exit
  

