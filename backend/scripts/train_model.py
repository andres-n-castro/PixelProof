#imports
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from dataset import videos_dataset_creation, DeepFakeDataset, DataLoader
from model import DeepfakeDetector
FILE_P1 = "save_model_p1.pth"
FILE_P2 = "save_model_p2.pth"
FILE_P3 = "save_model_p3.pth"

def train_one_epoch(model, loader, optimizer, criterion):
  model.train()
  total_loss, num_batches = 0, 0
  for batches, labels in loader:
    batches, labels = batches.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(batches)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    num_batches += 1
  return total_loss / num_batches

@torch.no_grad()
def evaluate_model(model, loader, criterion):
  model.eval()

  num_batches, total_loss = 0, 0
  for batches, labels in loader:
    batches, labels = batches.to(device), labels.to(device)
    outputs = model(batches)
    loss = criterion(outputs, labels)
    total_loss += loss.item()
    num_batches += 1
  return total_loss / num_batches

def training(model, train_loader, val_loader, num_epochs_lstm, num_epochs_resnet):
  optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
  criterion = nn.CrossEntropyLoss()
  model = model.to(device)
  best_val_loss = float('inf')

  #Phase 1 --> Train LSTM head params only
  for epoch in range(num_epochs_lstm):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

    val_loss = evaluate_model(model, val_loader,criterion)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), FILE_P1)

    print(f"train loss: {train_loss:.2f} | val loss: {val_loss:.2f}")

  
  #Phase 2 --> Train LSTM head and ResNet Body Block 1 params
  best_val_loss = float('inf')
  model.unfreeze_resnet_layers(num_blocks=1)
  optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

  for epoch in range(num_epochs_resnet):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

    val_loss = evaluate_model(model, val_loader,criterion)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), FILE_P2)

    print(f"train loss: {train_loss:.2f} | val loss: {val_loss:.2f}")


  #Phase 3 --> Train LSTM head and ResNet Body Block 1 and 2 params
  best_val_loss = float('inf')
  model.unfreeze_resnet_layers(num_blocks=2)
  optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

  for epoch in range(num_epochs_resnet):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

    val_loss = evaluate_model(model, val_loader,criterion)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), FILE_P3)

    print(f"train loss: {train_loss:.2f} | val loss: {val_loss:.2f}")

  return

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="trains model and saves state_dict")
  parser.add_argument("root_data_dir", metavar="root_dir_data", type=str, help="this is the root dir for the 2 folders for our reala and fake parsed images")
  parser.add_argument("--lr", metavar="learning rate", type=float, default=0.001, help="provides the leraning rate for pahse 1 training")
  parser.add_argument("--epochs_lstm", metavar="num epochs lstm", type=int,default=10,  help="provides num of epochs for phase 1 training")
  parser.add_argument("--epochs_resnet", metavar="num epochs resnet", type=int, default=15, help="provides num of epochs for phase 2 and 3 training")
  parser.add_argument("--batch_size", metavar="batch_size", type=int, default=32, help="provdies batch size for each DataLoader")
  args = parser.parse_args()

  #device instantiation
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #compose for data preprocessing
  compose = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  labels = [('real', 0), ('fake', 1)]

  print(f"Loading data from: {args.root_data_dir}")

  #create list of tuples of videos
  videos_dataset = videos_dataset_creation(args.root_data_dir, labels)

  #creating and splitting datasets
  train_ratio, val_ratio = int(len(videos_dataset) * 0.70), int(len(videos_dataset) * 0.15)
  train_split, val_split, test_split = videos_dataset[:train_ratio], videos_dataset[train_ratio:train_ratio+val_ratio], videos_dataset[train_ratio+val_ratio:]

  train_dataset = DeepFakeDataset(20, args.root_data_dir, train_split, transforms=compose)
  val_dataset = DeepFakeDataset(20, args.root_data_dir, val_split, transforms=compose)
  test_dataset = DeepFakeDataset(20, args.root_data_dir, test_split, transforms=compose)


  #Dataloaders for all dataset splits
  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
  )

  test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
  )

  #training phase
  model = DeepfakeDetector(
    models.ResNet34_Weights.DEFAULT,
    resnet_prog=True,
    input_size=512,
    hidden_size=512,
    num_layers=2,
    num_classes=2)

  training(model, train_loader, val_loader, num_epochs_lstm=args.epochs_lstm, num_epochs_resnet=args.epochs_resnet)
