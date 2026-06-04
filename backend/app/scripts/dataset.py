import torch
from torch.utils.data import Dataset
import random as rand
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
    real_list = [os.path.join(parent_dir, real_folder,x) for x in os.listdir(os.path.join(parent_dir, real_folder)) if x.endswith(".pt")]
    fake_list = [os.path.join(parent_dir, deepfake_folder,x) for x in os.listdir(os.path.join(parent_dir, deepfake_folder)) if x.endswith(".pt")]
    combined_dataset = real_list + fake_list
    rand.shuffle(combined_dataset)
    return combined_dataset


#testing code to see if dataset class correctly creates dataset between the strings of the real and fake folders
dataset = DeepfakeDataset(parent_dir="preprocessed_data/", real_folder="preprocessed_real", deepfake_folder="preprocessed_fake")

print(len(dataset))
print(dataset[0][0].shape)
print(dataset[0][1])


for i in range(10):
    _, label = dataset[i]
    print(label)