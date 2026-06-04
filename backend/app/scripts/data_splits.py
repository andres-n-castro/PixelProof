from torch.utils.data import DataLoader, random_split
from types import Any
from dataset import DeepfakeDataset

def create_dataloader_splits(dataset: DeepfakeDataset, batch_size: int, train_ratio: float, val_ratio: float, num_workers: int) -> None | tuple[DataLoader, DataLoader, DataLoader]:
  
  #check if ratios are valid
  if sum((train_ratio, val_ratio)) >= 1:
    raise ValueError("invalid ratio splits!")
  
  train_size, val_size, test_size = int(len(dataset) * train_ratio), int(len(dataset) * val_ratio), int(len(dataset) - (train_size+val_size))

  train_split, val_split, test_split = random_split(dataset=dataset, lengths=(train_size, val_size, test_size))
  train_loader = DataLoader(dataset=train_split, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  val_loader = DataLoader(dataset=val_split, batch_size=batch_size, num_workers=num_workers, shuffle=False)
  test_loader = DataLoader(dataset=test_split, batch_size=batch_size, num_workers=num_workers, shuffle=False)

  return train_loader, val_loader, test_loader


  

