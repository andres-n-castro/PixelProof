import torch
import numpy as np
import os
import cv2 as cv
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
import argparse
PROCESSED_REAL = '../data/processed_data/real'
PROCESSED_FAKE = '../data/processed_data/fake'

def obtain_face_frames(dataset_path : str):
  try:
    mtcnn = MTCNN(device="CPU:0")
    datasets = []
    datasets.append(dataset_path + 'real')
    datasets.append(dataset_path + 'fake')

    for dataset in datasets:
      videos = os.listdir(dataset)
      for video in videos:
        current_frame = 0
        video_path = os.path.join(dataset, video)
        cv_video = cv.VideoCapture(video_path)
        frame_step = int(cv_video.get(cv.CAP_PROP_FRAME_COUNT) / 20)

        if frame_step == 0:
          continue

        for current_frame in range(20):
          
          frame_index = current_frame * frame_step
          cv_video.set(cv.CAP_PROP_POS_FRAMES, frame_index)
          success, frame = cv_video.read()

          if success == False:
            break

          image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
          result = mtcnn.detect_faces(image)

          if result:
            box = result[0]['box']
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            face_frame = frame[y : y + h, x : x + w]
            
            resized_face_frame = cv.resize(face_frame, (256, 256))

            if dataset == datasets[0]:
              filename = f"{video}_frame_{current_frame}_dataset_real.png"
              store_path = os.path.join(PROCESSED_REAL, filename)
              cv.imwrite(store_path, resized_face_frame)
            
            else:
              filename = f"{video}_frame_{current_frame}_dataset_fake.png"
              store_path = os.path.join(PROCESSED_FAKE, filename)
              cv.imwrite(store_path, resized_face_frame)
          else: continue

  except Exception as e:
    print(f"{dataset_path} does not exist")
    exit()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="finds the required dataset")
  parser.add_argument("dataset_path", metavar="dataset_path", type=str, help="finds the correct path to the required dataset")
  args = parser.parse_args()
  path = args.dataset_path
  obtain_face_frames(path)

