from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from data_splits import create_dataloader_splits
from model import DeepfakeDetector
from torchvision import models
import argparse
import torch.optim as optim
import torch.nn as nn
import torch
import copy

#initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training(model: DeepfakeDetector, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, optimizer: optim.Adam, criterion: nn.CrossEntropyLoss, num_layers_unfreeze:int, finetune_lr: float) -> tuple[dict, float]:
  best_state_dict = {k : v.cpu().clone() for k,v in model.state_dict().items()}
  best_val_loss = float("inf")
  train_loss = 0

  #phase 1: Train lstm head only
  best_state_dict, best_val_loss, train_loss = train_phase(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, optimizer=optimizer, criterion=criterion)
  print(f"Phase 1 Training Final Loss: {train_loss}")
  
  #phase 2: Finetune the resnet body with the lstm head
  model.load_state_dict(best_state_dict)

  #first unfreeze specfied model layers
  model.unfreeze_resnet_layers(num_layers_unfreeze)
  new_optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=finetune_lr)

  best_state_dict, best_val_loss, train_loss = train_phase(model, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, optimizer=new_optim, criterion=criterion)
  print(f"Phase 2 Training Loss: {train_loss}")

  return best_state_dict, best_val_loss

def train_phase(model: DeepfakeDetector, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, optimizer: optim.Adam, criterion: nn.CrossEntropyLoss) -> tuple[dict, float, float]:
  best_val_loss = float("inf")
  best_state_dict = {k : v.cpu().clone() for k,v in model.state_dict().items()}
  avg_train_loss = 0

  for epoch in range(num_epochs):
    model.train()
    avg_train_loss = train_one_epoch(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion)
    print(f"Train Loss for Epoch {epoch}: {avg_train_loss}\n")

    if epoch % 1 == 0 or epoch == num_epochs -1:
      model.eval()
      curr_val_loss = evaluate(model=model, val_loader=val_loader, criterion=criterion)

      if curr_val_loss < best_val_loss:
        best_val_loss = curr_val_loss
        best_state_dict = {k : v.cpu().clone() for k,v in model.state_dict().items()}

  return best_state_dict, best_val_loss, avg_train_loss

def train_one_epoch(model: DeepfakeDetector, train_loader: DataLoader, optimizer: optim.Adam, criterion: nn.CrossEntropyLoss) -> float:

  tot_train_loss = 0
  
  #iterate through the dataset in batches using dataloader
  for batch, labels in train_loader:
    #empty gradient accumulation so gradients per batch dont stack
    optimizer.zero_grad()

    #send batch and labels to gpu
    batch = batch.to(device=device)
    labels = labels.to(device=device)

    #forward prop
    predictions = model(batch)

    #find loss
    l = criterion(predictions, labels)

    #acculumulate batch loss
    tot_train_loss += l.item()

    #obtain loss gradients using backward prop
    l.backward()
    
    #update weights
    optimizer.step()

  return tot_train_loss / len(train_loader)
 
@torch.no_grad()
def evaluate(model: DeepfakeDetector, val_loader: DataLoader, criterion: nn.CrossEntropyLoss) -> float:

  tot_loss = 0

  for batch, labels in val_loader:

    batch = batch.to(device=device)
    labels = labels.to(device=device)

    #forward pass
    predictions = model(batch)
    
    #calculate loss
    l = criterion(predictions, labels)

    tot_loss += l.item()
  
  return tot_loss / len(val_loader)
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--parent_samples_dir", type=str, metavar="root_samples_dir", help="root folder containing the real and fake samples folders")
  parser.add_argument("--num_epochs", type=int, metavar="num_epochs", help="value that represents the number of epochs the model will iterate through for training")
  parser.add_argument("--batch_size", type=int, metavar="batch_size", help="value representing how large a data batch should be for a dataloader")
  parser.add_argument("--learn_rate", type=float, metavar="learning_rate", help="the length of the step that gradient descent will go down towards for convergence")
  parser.add_argument("--finetune_learn_rate", type=float, metavar="finetuning_learning_rate", help="the length factor of the step that gradient descent will go down towards for convergence for finetuning")
  parser.add_argument("--analytics_output_folder_path", type=str, metavar="analytics_output_path", help="the path of the folder that the analytics file will be place in")
  parser.add_argument("--best_state_dict_filename", type=str, metavar="", help="the name of the file that the best state dict for training will be placed in")
  parser.add_argument("--num_unfreeze_layers", type=int, metavar="num_unfreeze_layers", help="value that represents the code block that will unfeeze a certain number of layers from the resnet body")
  parser.add_argument("--train_ratio", type=float, metavar="training_data_split_ratio", help="value that represents the ratio of the dataset that will be used as training data")
  parser.add_argument("--val_ratio", type=float, metavar="val_data_split_ratio", help="value that represents the ratio of the dataset that will be used as validation data")
  parser.add_argument("--num_workers", type=int, metavar="dataloader_num_workers", help="value represents the number of workers to pass through for dataloaders")

  args = parser.parse_args()

  #initialize the dataloaders
  dataset = DeepfakeDataset(parent_dir=args.parent_samples_dir, real_folder="preprocessed_real", deepfake_folder="preprocessed_fake")

  train_dataloader, val_dataloader, _ = create_dataloader_splits(
    dataset=dataset,
    batch_size=args.batch_size,
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    num_workers=args.num_workers,
    )

  #initialize the model
  #will need to later adjust argument values (input, hidden, and num_layers)
  model = DeepfakeDetector(
    resnet_weights=models.ResNet50_Weights.DEFAULT, 
    resnet_prog=True, 
    input_size=224, 
    hidden_size=224,
    num_layers=2,
    num_classes=2
    )
  
  #send model to gpu
  model.to(device=device)

  #initialize optimizer
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learn_rate)

  #initialize loss function
  criterion = nn.CrossEntropyLoss()

  #call training function
  best_state_dict, best_val_loss = training(
    model=model, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader, 
    num_epochs=args.num_epochs, 
    optimizer=optimizer, 
    criterion=criterion, 
    num_layers_unfreeze=args.num_unfreeze_layers,
    finetune_lr=args.finetune_learn_rate
    )
  
  print(f"Best weights: {best_state_dict} | Best Val Loss: {best_val_loss}")
  torch.save(best_state_dict, args.best_state_dict_filename)





