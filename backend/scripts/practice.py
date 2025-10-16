import os
import cv2 as cv
from mtcnn import MTCNN
import argparse
PROCESSED_REAL = '../data/processed_data/real/'
PROCESSED_FAKE = '../data/processed_data/fake/'


def preprocess_data(dataset_path : str):
  #! 2 paths for the given dataset path, ane for real folder and one for fake folder

  try:
    mtcnn = MTCNN(device="CPU:0")
    paths = []
    paths.append(dataset_path + "real")
    paths.append(dataset_path + "fake")

    #!for loop for each path
      #!for loop through each video
        #! for loop through n frames per video using opencv
          #! for each frame, process it by using mtcnn for face scanning
          #!then use coords from mtcnn result to find face in frame and then resize face to a 256 by 256
          #!store frame in a new folder using opencv

    for path in paths:
      videos = os.listdir(path)
      for video in videos:
        frame_counter = 0
        video_path  = os.path.join(path, video)
        cv_video = cv.VideoCapture(video_path)
        frame_step = int(cv_video.get(cv.CAP_PROP_FRAME_COUNT) / 20)

        if frame_step == 0:
          continue
      
        for i in range(20):

          frame_step = frame_step * frame_counter
          cv_video.set(cv.CAP_PROP_POS_FRAMES, frame_step)

          success, frame = cv_video.read()

          if success:
            #!change frame from BRG to RGB
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = mtcnn.detect_faces(image)

            box = result[0]['box']
            x = box[0]
            w = box[1]
            y = box[2]
            h = box[3]

            new_frame = frame[y : y + h, x : x + w]
            resized_frame = cv.resize(new_frame, (256, 256))

            if path == paths[0]:
              filename = f"{video}_frame_{frame_counter}_real.png"
              store_path = os.join.path(PROCESSED_REAL, filename)
              cv.imwrite(store_path, resized_frame)
            else:
              filename = f"{video}_frame_{frame_counter}_fake.png"
              store_path = os.path.join(PROCESSED_FAKE, filename)
              cv.imwrite(store_path, resized_frame)
  except Exception as e:
    print(e)
    exit()

        
        



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", metavar="dataset", type=str, help="finds the path of the dataset")
  args = parser.parse_args()
  path = args.dataset
  preprocess_data(path)
  