
import os
import cv2 as cv
from mtcnn import MTCNN
import argparse
from tqdm import tqdm
mtcnn = MTCNN(device="CPU:0")



def process_single_frame(video : str, dataset : str, dataset_type : int, dest_folder : str):

  video_path = os.path.join(dataset, video) #path for sample video
  current_frame = 0 #frame count for later use when labeling frames
  cv_video = cv.VideoCapture(video_path) #uses opencv to create a VideoCapture object from the video and start applying methods to the video
  frame_step = int(cv_video.get(cv.CAP_PROP_FRAME_COUNT) / 20) #step value in order to obtain every nth frame in a video (20 frames)
  os.makedirs(os.path.join(dest_folder, "Processed Real"), exist_ok=True)
  os.makedirs(os.path.join(dest_folder, "Processed Fake"), exist_ok=True)

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

        if dataset_type == 0:
          filename = f"{video}_frame_{current_frame}_dataset_real.png"
          store_path = os.path.join(dest_folder, "Processed Real",filename)
          cv.imwrite(store_path, resized_face_frame)
        
        else:
          filename = f"{video}_frame_{current_frame}_dataset_fake.png"
          store_path = os.path.join(dest_folder, "Processed Fake",filename)
          cv.imwrite(store_path, resized_face_frame)


def obtain_face_frames(dataset_path : str, dest_folder_path : str):
  try:
    datasets = []
    datasets.append(dataset_path + '/real')
    datasets.append(dataset_path + '/fake')

    for dataset in datasets:
      videos = os.listdir(dataset)

      try:
        for video in tqdm(videos):
          
          if dataset == datasets[0]:
            process_single_frame(video, dataset, 0, dest_folder_path)
          else:
            process_single_frame(video, dataset, 1, dest_folder_path)

      except Exception as e:
        print(e)
        exit()

  except Exception as e:
    print(e)
    exit()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="finds the required dataset")
  parser.add_argument("dataset_path", metavar="dataset_path", type=str, help="finds the correct path to the required dataset")
  parser.add_argument("destination_path", metavar="google drive folder", type=str, help="this is the full path for the google drive folder")
  args = parser.parse_args()
  path = args.dataset_path
  dest_folder_path = args.destination_path
  obtain_face_frames(path, dest_folder_path)

