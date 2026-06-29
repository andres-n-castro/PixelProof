import torch
from torch.utils.data import DataLoader, Subset, random_split
from dataset import DeepfakeDataset

def create_dataloader_splits(
  dataset: DeepfakeDataset,
  batch_size: int,
  num_workers: int,
  train_ratio: float | None = None,
  val_ratio: float | None = None,
  seed: int | None = None,
  split_indices_path: str | None = None,
  load_saved_indices: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
  
  if load_saved_indices:
    if split_indices_path is None:
      raise ValueError("split_indices_path is required when load_saved_indices=True")
    split_indices = torch.load(split_indices_path, map_location="cpu")
  else:
    if train_ratio is None or val_ratio is None:
      raise ValueError("train_ratio and val_ratio are required when creating new splits")
    if sum((train_ratio, val_ratio)) >= 1:
      raise ValueError("invalid ratio splits!")
  
    train_size, val_size = int(len(dataset) * train_ratio), int(len(dataset) * val_ratio)
    test_size = int(len(dataset) - (train_size + val_size))

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_split, val_split, test_split = random_split(
      dataset=dataset,
      lengths=(train_size, val_size, test_size),
      generator=generator
    )
    split_indices = {
      "train": train_split.indices,
      "val": val_split.indices,
      "test": test_split.indices,
    }

    if split_indices_path is not None:
      torch.save(split_indices, split_indices_path)

  train_subset = Subset(dataset, split_indices["train"])
  val_subset = Subset(dataset, split_indices["val"])
  test_subset = Subset(dataset, split_indices["test"])

  train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
  test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

  return train_loader, val_loader, test_loader


  

