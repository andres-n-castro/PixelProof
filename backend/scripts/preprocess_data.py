
import os
import cv2 as cv
from mtcnn import MTCNN
import argparse
from tqdm import tqdm
from google.cloud import storage
from zipfile import ZipFile
import pandas as pd
mtcnn = MTCNN(device="CPU:0")


#full_video_path, label, destination(bucket)
def process_single_frame(full_video_path : str, label : str, bucket : storage.Client.bucket):

  '''
  video_path = os.path.join(dataset, video) #path for sample video
  current_frame = 0 #frame count for later use when labeling frames
  cv_video = cv.VideoCapture(video_path) #uses opencv to create a VideoCapture object from the video and start applying methods to the video
  frame_step = int(cv_video.get(cv.CAP_PROP_FRAME_COUNT) / 20) #step value in order to obtain every nth frame in a video (20 frames)
  os.makedirs(os.path.join(dest_folder, "Processed Real"), exist_ok=True)
  os.makedirs(os.path.join(dest_folder, "Processed Fake"), exist_ok=True)
  '''
  parsed_video = full_video_path.split('/')
  video_id = parsed_video[-1]

  current_frame = 0 #frame count for later use when labeling frames
  cv_video = cv.VideoCapture(full_video_path) #uses opencv to create a VideoCapture object from the video and start applying methods to the video
  frame_step = int(cv_video.get(cv.CAP_PROP_FRAME_COUNT) / 20) #step value in order to obtain every nth frame in a video (20 frames)

  #SAVE FROM HERE

  #if conditional in case video is less than 20 frames long, we wont use that video as a sample
  if frame_step == 0: 
    return

  #loops through all 20 frames in a video since for each video we are extracting 20 frames
  for current_frame in range(20):
    
    frame_index = current_frame * frame_step # index for the current frame we want 
    cv_video.set(cv.CAP_PROP_POS_FRAMES, frame_index) #sets the VideoCapture var cv_video to capture video[frame_index]
    success, frame = cv_video.read() #processes the frame at frame_index and retrieve frame information as a well as return a value if it was successfull or not

    if success == False:
      return

    #changes the channels from BGR to RGB so that mtcnn can detect faces (mtcnn requires RBG and opencv process in BGR)
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    result = mtcnn.detect_faces(image) #mtcnn retrieves any faces within the frame information

    #if a face was retrieved then use boundbox coordinates to obtain face from original frame information
    #and then resize the face frame into the original 256 x 256 that the frame was in. Each frame goes into
    #their respective folder whether the frame belongs to a video that is fake(use deepfake) or real. The sample
    #is stored in its folder with 
    if result:
      box = result[0]['box']
      x, y, w, h = box[0], box[1], box[2], box[3]
      x, y, = max(0, x), max(0, y) #ensures the coordinates cannot be negative in the case that the bounding box captures a face oustide the bounds of the orig frame size

      face_frame = frame[y : y + h, x : x + w] # obtains the face within the frame using the bounding box coords

      if face_frame.size != 0:
    
        resized_face_frame = cv.resize(face_frame, (256, 256)) #resizes face frame

        if label == 'REAL':
          filename = f"{video_id}_frame_{current_frame}_dataset_real.png"

          #must fix since cannot store directly to bucket   
          store_path = os.path.join(dest_folder, "Processed Real",filename)
          cv.imwrite(store_path, resized_face_frame)
        
        else:
          filename = f"{video_id}_frame_{current_frame}_dataset_fake.png"

          #must fix since cannot store directly to bucket          store_path = os.path.join(dest_folder, "Processed Fake",filename)
          cv.imwrite(store_path, resized_face_frame)


def obtain_face_frames(input_dir : str, master_df : pd.DataFrame, bucket : storage.Client.bucket):
  try:
    for row in master_df[['File Path, Label']].itertuples(name=None):
      full_video_path = os.path.join(input_dir, row[1])
      process_single_frame(full_video_path, row[2], bucket)

  except Exception as e:
    print(f"error: {e}")
    exit()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="finds the required dataset")
  parser.add_argument("kaggle_input_dir", metavar="input dir", type=str, help="arg for input dir")
  parser.add_argument("bucket_name", metavar="gcs bucket", type=str, help="arg for gcs bucket name")
  parser.add_argument("project_id", metavar="proj_id", type=str, help="the project id in order to access the project and then the bucket within")
  parser.add_argument("master_path", metavar="master_csv_path", type=str)
  args = parser.parse_args()
  path = args.dataset_path
  bucket_name = args.bucket_name
  input_dir = args.kaggle_input_dir
  proj_id = args.project_id
  master_csv_path = args.master_path
  kaggle_work_dir = 'kaggle/working'

  try:
    storage_client = storage.Client(proj_id)
    bucket = storage_client.bucket(bucket_name=bucket_name)

    blob = bucket.blob(blob_name=master_csv_path)
    blob.download_to_filename(os.path.join(kaggle_work_dir, 'master.csv'))

    with ZipFile(os.path.join(kaggle_work_dir, 'master.csv'), 'r') as csv_zip:
      with csv_zip.open('master.csv') as master_csv:
        master_df = pd.read_csv(master_csv)

    obtain_face_frames(input_dir, bucket, master_df)
  
  except Exception as e:
    print(f"error: {e}")
    exit()

