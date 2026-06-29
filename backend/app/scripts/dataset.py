import torch
from torch.utils.data import Dataset
import os


class DeepfakeDataset(Dataset):
  def __init__(self, parent_dir: str, real_folder: str, deepfake_folder: str):
    #load in datasets
    self.dataset = self.create_combined_dataset(parent_dir=parent_dir, real_folder=real_folder, deepfake_folder=deepfake_folder)


  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    sample_path = self.dataset[index]
    sample = torch.load(sample_path)
    return (sample["frames"], sample["label"])

  def create_combined_dataset(self, parent_dir: str, real_folder: str, deepfake_folder: str):
    real_list = sorted(
      os.path.join(parent_dir, real_folder, x)
      for x in os.listdir(os.path.join(parent_dir, real_folder))
      if x.endswith(".pt")
    )
    fake_list = sorted(
      os.path.join(parent_dir, deepfake_folder, x)
      for x in os.listdir(os.path.join(parent_dir, deepfake_folder))
      if x.endswith(".pt")
    )
    combined_dataset = real_list + fake_list
    return combined_dataset