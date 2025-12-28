import torch
import PIL
import os
import cv2 as cv
import random as rand
from torch.utils.data import Dataset, DataLoader
from pandas import DataFrame
import zipfile as zp
from zipfile import ZipFile

REAL_FOLDER = "Processed Real"
FAKE_FOLDER = "Processed Fake"

#custom dataset
class DeepFakeDataset(Dataset):
  def __init__(self, sequence_length : int, master_df : DataFrame, real_zip : ZipFile, fake_zip : ZipFile, transforms=None):
    self.sequence_length = sequence_length
    self.transforms = transforms
    self.master_df = master_df
    self.real_zip = real_zip
    self.fake_zip = fake_zip

  #returns # of rows in the master dataframe
  def __len__(self):
    return len(self.master_df)

  #idx is a row in the dataframe
  def __getitem__(self, idx):
    try:
      video_id, label  = self.master_df.iloc[idx, :]
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
