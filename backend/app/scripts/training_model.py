from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from data_splits import create_dataloader_splits
from model import DeepfakeClassifier
import os
from torchvision import models
import argparse
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

#initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training(model: DeepfakeClassifier, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, optimizer: optim.Adam, criterion: nn.CrossEntropyLoss, num_layers_unfreeze:int, finetune_lr: float):
  best_state_dict = {k : v.cpu().clone() for k,v in model.state_dict().items()}
  best_val_loss = float("inf")
  train_loss = 0
  history = {
  "train_loss": [],
  "val_loss": [],
  "val_accuracy": [],
  "val_recall": [],
  "val_precision": [],
  "val_f1": [],
  }

  #phase 1: Train lstm head only
  best_state_dict, best_history_idx, best_val_loss, train_loss = train_phase(model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, optimizer=optimizer, criterion=criterion, history=history)
  print(f"Phase 1 Training Final Loss: {train_loss}")
  
  #phase 2: Finetune the resnet body with the lstm head
  model.load_state_dict(best_state_dict)

  #first unfreeze specfied model layers
  model.unfreeze_resnet_layers(num_layers_unfreeze)
  new_optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=finetune_lr)

  best_state_dict, best_history_idx, best_val_loss, train_loss = train_phase(model, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, optimizer=new_optim, criterion=criterion, history=history)
  print(f"Phase 2 Training Loss: {train_loss}")

  return best_state_dict, best_val_loss, best_history_idx, history 

def train_phase(model: DeepfakeClassifier, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, optimizer: optim.Adam, criterion: nn.CrossEntropyLoss, history: dict) -> tuple[dict, int, float, float]:
  best_val_loss = float("inf")
  best_history_idx = -1
  best_state_dict = {k : v.cpu().clone() for k,v in model.state_dict().items()}
  train_loss = 0

  for epoch in range(num_epochs):
    model.train()
    train_loss = train_one_epoch(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion)

    
    model.eval()
    val_accuracy, val_recall, val_precision, val_f1, curr_val_loss = evaluate(model=model, val_loader=val_loader, criterion=criterion)

    #append performance metrics to history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(curr_val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_recall"].append(val_recall)
    history["val_precision"].append(val_precision)
    history["val_f1"].append(val_f1)

    print(f"Train Loss for Epoch {epoch}: {train_loss}\n")
    print(f"Val Loss for Epoch {epoch}: {curr_val_loss}\n")

    #condition block that obtains the history idx with the best val loss
    if curr_val_loss < best_val_loss:
      best_val_loss = curr_val_loss
      best_state_dict = {k : v.cpu().clone() for k,v in model.state_dict().items()}
      best_history_idx = len(history["val_loss"]) - 1

  return best_state_dict, best_history_idx, best_val_loss, train_loss

def train_one_epoch(model: DeepfakeClassifier, train_loader: DataLoader, optimizer: optim.Adam, criterion: nn.CrossEntropyLoss) -> float:

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
def evaluate(model: DeepfakeClassifier, val_loader: DataLoader, criterion: nn.CrossEntropyLoss) -> tuple[float, float, float, float, float]:

  all_preds = []
  all_labels = []
  tot_loss = 0

  for batch, labels in val_loader:

    batch = batch.to(device=device)
    labels = labels.to(device=device)

    #forward pass
    predictions = model(batch)
    pred_classes = torch.argmax(predictions, dim=1)

    all_preds.extend(pred_classes.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())
    
    #calculate loss
    l = criterion(predictions, labels)

    tot_loss += l.item()

  val_accuracy = accuracy_score(y_true=all_labels, y_pred=all_preds)
  val_recall = recall_score(y_true=all_labels, y_pred=all_preds)
  val_precision = precision_score(y_true=all_labels, y_pred=all_preds)
  val_f1 =  f1_score(y_true=all_labels, y_pred=all_preds)
  val_loss = tot_loss / len(val_loader)

  
  return (val_accuracy, val_recall, val_precision, val_f1, val_loss)
    
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
  parser.add_argument("--split_seed", type=int, metavar="split_seed", default=42, help="the random seed used to generate deterministic train, val, and test splits")
  parser.add_argument("--split_indices_filename", type=str, metavar="split_indices_filename", default="split_indices.pt", help="the file name used to save the generated train, val, and test indices")

  args = parser.parse_args()

  #initialize the dataloaders
  dataset = DeepfakeDataset(parent_dir=args.parent_samples_dir, real_folder="preprocessed_real", deepfake_folder="preprocessed_fake")
  split_indices_path = os.path.join(args.analytics_output_folder_path, args.split_indices_filename)

  train_dataloader, val_dataloader, _ = create_dataloader_splits(
    dataset=dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    seed=args.split_seed,
    split_indices_path=split_indices_path,
    )

  #initialize the model
  #will need to later adjust argument values (input, hidden, and num_layers)
  model = DeepfakeClassifier(
    weights=models.ResNet18_Weights.DEFAULT, 
    input_size=512, 
    hidden_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu"
    )
  
  #send model to gpu
  model.to(device=device)

  #initialize optimizer
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learn_rate)

  #initialize loss function
  criterion = nn.CrossEntropyLoss()

  #call training function
  best_state_dict, best_val_loss, best_history_idx, history = training(
    model=model, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader, 
    num_epochs=args.num_epochs, 
    optimizer=optimizer, 
    criterion=criterion, 
    num_layers_unfreeze=args.num_unfreeze_layers,
    finetune_lr=args.finetune_learn_rate
    )
  
  best_history_metrics = {
    "best_train_loss": history["train_loss"][best_history_idx],
    "best_val_loss": history["val_loss"][best_history_idx],
    "best_val_accuracy": history["val_accuracy"][best_history_idx],
    "best_val_recall": history["val_recall"][best_history_idx],
    "best_val_precision": history["val_precision"][best_history_idx],
    "best_val_f1_score": history["val_f1"][best_history_idx],
  }
  
  print(f"Best weights: {best_state_dict}")

  print(f"Best History Metrics: {best_history_metrics}")

  torch.save(best_state_dict, args.best_state_dict_filename)
  torch.save(history, os.path.join(args.analytics_output_folder_path, "history.pt"))





