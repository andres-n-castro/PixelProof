
import os
import cv2 as cv
from mtcnn import MTCNN
import argparse
from tqdm import tqdm
PROCESSED_REAL = '../data/processed_data/real/'
PROCESSED_FAKE = '../data/processed_data/fake/'
mtcnn = MTCNN(device="CPU:0")

def process_single_frame(video : str, dataset : str, dataset_type : int):
  video_path = os.path.join(dataset, video)
  current_frame = 0
  cv_video = cv.VideoCapture(video_path)
  frame_step = int(cv_video.get(cv.CAP_PROP_FRAME_COUNT) / 20)

  if frame_step == 0:
    return

  for current_frame in range(20):
    
    frame_index = current_frame * frame_step
    cv_video.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cv_video.read()

    if success == False:
      return

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

      if dataset_type == 0:
        filename = f"{video}_frame_{current_frame}_dataset_real.png"
        store_path = os.path.join(PROCESSED_REAL, filename)
        cv.imwrite(store_path, resized_face_frame)
      
      else:
        filename = f"{video}_frame_{current_frame}_dataset_fake.png"
        store_path = os.path.join(PROCESSED_FAKE, filename)
        cv.imwrite(store_path, resized_face_frame)


def obtain_face_frames(dataset_path : str):
  try:
    datasets = []
    datasets.append(dataset_path + 'real')
    datasets.append(dataset_path + 'fake')

    for dataset in datasets:
      videos = os.listdir(dataset)

      try:
        for video in tqdm(videos):
          
          if dataset == datasets[0]:
            process_single_frame(video, dataset, 0)
          else:
            process_single_frame(video, dataset, 1)

      except Exception as e:
        print(e)
        exit()

  except Exception as e:
    print(e)
    exit()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="finds the required dataset")
  parser.add_argument("dataset_path", metavar="dataset_path", type=str, help="finds the correct path to the required dataset")
  args = parser.parse_args()
  path = args.dataset_path
  obtain_face_frames(path)

