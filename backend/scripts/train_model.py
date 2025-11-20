#imports
import torch
import random as rand
import os
import argparse
import torch.nn as nn
import cv2 as cv
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import PIL

ROOT_DIR = "../data"
REAL_FOLDER = "real"
FAKE_FOLDER = "fake"


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



#LSTM class head implementation



#device instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#compose for data preprocessing
compose = transforms.Compose([
  transforms.Resize(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels = [(REAL_FOLDER, 0), (FAKE_FOLDER, 1)]

#create list of tuples of videos
videos_dataset = videos_dataset_creation(ROOT_DIR, labels)

#creating and splitting datasets
train_ratio, val_ratio = int(len(videos_dataset) * 0.70), int(len(videos_dataset) * 0.15)
train_split, val_split, test_split = videos_dataset[:train_ratio], videos_dataset[train_ratio:train_ratio+val_ratio], videos_dataset[train_ratio+val_ratio:]

train_dataset = DeepFakeDataset(20, ROOT_DIR, train_split, transforms=compose)
val_dataset = DeepFakeDataset(20, ROOT_DIR, val_split, transforms=compose)
test_dataset = DeepFakeDataset(20, ROOT_DIR, test_split, transforms=compose)


#Dataloaders for all dataset splits
train_loader = DataLoader(
  train_dataset,
  batch_size=32,
  shuffle=True,
  num_workers=2,
  pin_memory=True,
  drop_last=True
)

val_loader = DataLoader(
  val_dataset,
  batch_size=32,
  shuffle=False,
  num_workers=2,
  pin_memory=True
)

test_loader = DataLoader(
  test_dataset,
  batch_size=32,
  shuffle=False,
  num_workers=2,
  pin_memory=True
)

#load pre-trained resnet model





#configure resnet model to have LSTM as head




#train resnet-lstm model for head weights first




#fine-tune training for entire resnet-lstm model




#save state dict from final resnet-lstm model
