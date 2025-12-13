import torch
import PIL
import os
import cv2 as cv
import random as rand
from torch.utils.data import Dataset, DataLoader

REAL_FOLDER = "Processed Real"
FAKE_FOLDER = "Processed Fake"


def videos_dataset_creation(root_dir, labels : list[tuple[str, int], tuple[str, int]]):
  videos = []
  for label in labels:
    temp_videos = set()
    frames = os.listdir(os.path.join(root_dir, label[0]))
    for frame in frames:
      frame_name_items = frame.split("_frame_")
      video_id = frame_name_items[0]
      if video_id not in temp_videos:
        temp_videos.add(video_id)
        videos.append((video_id, label[1]))

  rand.shuffle(videos)

  return videos

#custom dataset
class DeepFakeDataset(Dataset):
  def __init__(self, sequence_length, root_dir, dataset,transforms=None):
    self.root_dir = root_dir
    self.sequence_length = sequence_length
    self.transforms = transforms
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    try:
      video_id, label  = self.dataset[idx]
      sample = []

      for i in range(self.sequence_length):
        if label == 0:
          frame_name = f"{video_id}_frame_{i}_dataset_real.png"
          frame_path = os.path.join(self.root_dir, REAL_FOLDER,frame_name)
        else:
          frame_name = f"{video_id}_frame_{i}_dataset_fake.png"
          frame_path = os.path.join(self.root_dir, FAKE_FOLDER,frame_name)
        
        
        loaded_frame = cv.imread(frame_path)
        loaded_frame = cv.cvtColor(loaded_frame, cv.COLOR_BGR2RGB)
        loaded_frame = PIL.Image.fromarray(loaded_frame)

        if self.transforms:
          loaded_frame = self.transforms(loaded_frame)

        sample.append(loaded_frame)

      return torch.stack(sample), label
      
    except Exception as e:
      print(f"error loading sample {idx} : {e}")
      return self.__getitem__((idx+1) % len(self))
