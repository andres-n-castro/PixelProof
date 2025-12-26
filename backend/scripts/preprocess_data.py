
import os
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
import argparse
from google.cloud import storage
from zipfile import ZipFile
import pandas as pd
from google.cloud.storage import Bucket

BaseOptions = mp.tasks.BaseOptions
FaceDetector = vision.FaceDetector
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode
  
def process_single_video(full_video_path : str, label : str):

  try:
    parsed_video = full_video_path.split('/')
    video_id = parsed_video[-1]
    video_path_valid = os.path.exists(full_video_path)
    print(f"{full_video_path} is {video_path_valid}")

    current_frame = 0 #frame count for later use when labeling frames
    cv_video = cv.VideoCapture(full_video_path) #uses opencv to create a VideoCapture object from the video and start applying methods to the video

    if cv_video.isOpened():
      frame_step = int(cv_video.get(cv.CAP_PROP_FRAME_COUNT) // 20) #step value in order to obtain every nth frame in a video (20 frames)


      #if conditional in case video is less than 20 frames long, we wont use that video as a sample
      if frame_step == 0: 
        print("less than 20 frames in video file, returning and moving to next video file")
        return []

      frames = []
    
    else:
      print("videocapture failed to create object")
      return []
  
  except Exception as e:
    print(f"error: {e}")
    return []

  #mediapipe implementation (current supported API --> Face Detector)
  try:
    with FaceDetector.create_from_options(options) as detector:
      for current_frame in range(20):
        frame_idx = current_frame * frame_step
        cv_video.set(cv.CAP_PROP_POS_FRAMES, frame_idx)

        success, frame = cv_video.read()

        if not success:
          print("failed to retrieve frame, returning and moving to next video file!")
          return []

        #create a copy of the frame to pass to media pipe so you dont have to convert back to
        #RGB when you load back into cv later
        frame_copy = frame.copy()
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame_copy, cv.COLOR_BGR2RGB)))

        if not result.detections:
          continue
        
        if len(result.detections) > 1:
          largest_face_idx, largest_face_area = 0, 0
          for idx, detection in enumerate(result.detections):
            cur_face_area = detection.bounding_box.width * detection.bounding_box.height
            if largest_face_area < cur_face_area:
              largest_face_idx, largest_face_area = idx, cur_face_area
            else: continue
          
          chosen_face_frame = result.detections[largest_face_idx]
          frame_h, frame_w, _ = frame.shape

          #checks to see if the bouding box origin coords are negative i.e outside of the frame --> returns 0 if it is
          crop_origin_x = int(max(0, chosen_face_frame.bounding_box.origin_x))
          crop_origin_y = int(max(0, chosen_face_frame.bounding_box.origin_y))

          #checks to see if width/height of face is greater than width/height of frame
          min_width = min(frame_w, crop_origin_x + int(chosen_face_frame.bounding_box.width))
          min_height = min(frame_h, crop_origin_y + int(chosen_face_frame.bounding_box.height))

          if (min_width - crop_origin_x > 0) and (min_height - crop_origin_y > 0):
            cropped_frame = frame[crop_origin_y : min_height, crop_origin_x : min_width]
            resized_cropped_frame = cv.resize(cropped_frame, (256, 256))
          else:
            print("could not crop frame in order to obtain a face!")
            return []

        else:
          for detection in result.detections:
            frame_h, frame_w, _ = frame.shape

            #checks to see if the bouding box origin coords are negative i.e outside of the frame --> returns 0 if it is
            crop_origin_x = int(max(0, detection.bounding_box.origin_x))
            crop_origin_y = int(max(0, detection.bounding_box.origin_y))

            #checks to see if width/height of face is supercedes the width/height of frame
            min_width = min(frame_w, crop_origin_x + int(detection.bounding_box.width))
            min_height = min(frame_h, crop_origin_y + int(detection.bounding_box.height))

            if (min_width - crop_origin_x > 0)  and (min_height - crop_origin_y > 0):
              cropped_frame = frame[crop_origin_y : min_height, crop_origin_x : min_width]
              resized_cropped_frame = cv.resize(cropped_frame, (256, 256))
            else:
              print("could not crop frame in order to obtain a face!")
              return []
          
        if label == 'REAL':
          filename = f"{video_id}_frame_{current_frame}_dataset_real.png"

          success, frame_buffer = cv.imencode(".png", resized_cropped_frame)
          if success:
            frames.append((frame_buffer.tobytes(), filename))
          else:
            print("writing frame to memory was unsuccessfull!")
            return []

        elif label == 'FAKE':
          filename = f"{video_id}_frame_{current_frame}_dataset_fake.png"
          
          success, frame_buffer = cv.imencode(".png", resized_cropped_frame)
          if success:
            frames.append((frame_buffer.tobytes(), filename))
          else:
            print("writing frame to memory was unsuccessfull!")
            return []
        else:
          continue
  
    cv_video.release()
    return frames

  except Exception as e:
    print(f"error: {e}")
    return []

def obtain_face_frames(input_dir : str, master_df : pd.DataFrame, bucket : Bucket):
  try:
    print("creating real_frames.zip and fake_frames.zip and opening them for writing...")
    with ZipFile("real_frames.zip", 'w') as real_zip, ZipFile("fake_frames.zip", 'w') as fake_zip:
      print("iterating through master.csv...\n")
      for row in master_df[['File Path', 'Label']].itertuples(name=None):
        if row[2] == "REAL":
          print("found an original video file!")
          full_video_path = os.path.join(input_dir, row[1])
          print("processing original video file..\n")
          frames = process_single_video(full_video_path, row[2])

          for frame_info in frames:
            real_zip.writestr(frame_info[1], frame_info[0])

        elif row[2] == "FAKE":
          print("found an deepfake video file!")
          full_video_path = os.path.join(input_dir, row[1])
          print("processing deepfake video file..\n")
          frames = process_single_video(full_video_path, row[2])

          for frame_info in frames:
            fake_zip.writestr(frame_info[1], frame_info[0])
        
        else:
          print("label not REAL or FAKE, continuing to next row/sample in csv")
          continue

    print("uploading real_frames.zip...")
    blob_name = "processed_real/real_frames.zip"
    blob = bucket.blob(blob_name=blob_name)
    blob.upload_from_filename("real_frames.zip")
    print("upload successfull!\n")

    print("uploading fake_frames.zip...")
    blob_name = "processed_fake/fake_frames.zip"
    blob = bucket.blob(blob_name=blob_name)
    blob.upload_from_filename("fake_frames.zip")
    print("upload successfull!\n")


  except Exception as e:
    print(f"error: {e}")
    return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="finds the required dataset")
  parser.add_argument("kaggle_input_dir", metavar="input dir", type=str, help="arg for input dir")
  parser.add_argument("bucket_name", metavar="gcs bucket", type=str, help="arg for gcs bucket name")
  parser.add_argument("project_id", metavar="proj_id", type=str, help="the project id in order to access the project and then the bucket within")
  parser.add_argument("master_path", metavar="master_csv_path", type=str)
  parser.add_argument("blaze_model", metavar="face_model", type=str, help="name of the face detection model")

  args = parser.parse_args()
  bucket_name = args.bucket_name
  input_dir = args.kaggle_input_dir
  proj_id = args.project_id
  master_csv_path = args.master_path
  face_detect_model = args.blaze_model
  kaggle_work_dir = '../../../'

  try:
    storage_client = storage.Client(proj_id)
    bucket = storage_client.bucket(bucket_name=bucket_name)

    print("retrieving face detect model..")
    blob = bucket.blob(blob_name=os.path.join("FaceDetector_Model", face_detect_model))
    blob.download_to_filename(os.path.join(kaggle_work_dir, face_detect_model))
    print("face detect model successfully retrieved!")
    
    options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(kaggle_work_dir, face_detect_model)),
    running_mode=VisionRunningMode.IMAGE)

    print("downloading csv.zip file from bucket...")
    blob = bucket.blob(blob_name=master_csv_path)
    blob.download_to_filename(os.path.join(kaggle_work_dir, 'csv.zip'))
    print("download successfull!")

    with ZipFile(os.path.join(kaggle_work_dir, 'csv.zip'), 'r') as csv_zip:
      with csv_zip.open('master.csv') as master_csv:
        master_df = pd.read_csv(master_csv)

    print(master_df.columns)
    obtain_face_frames(input_dir, master_df, bucket)
  
  except Exception as e:
    print(f"error: {e}")
