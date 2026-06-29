from data_splits import create_dataloader_splits
from model import DeepfakeClassifier
from dataset import DeepfakeDataset
import os
import torch
import argparse
from training_model import evaluate
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--best_state_dict_path", type=str, metavar="best_state_dict_path", help="the file path where the best_state_dict.pt is located")
  parser.add_argument("--split_indices_path", type=str, metavar="split_indices_path", help="the file path where the saved train, val, and test split indices are located")
  parser.add_argument("--parent_samples_dir", type=str, metavar="parent_samples_dir", help="the parent directory containing the data samples necessary for model training/testing")
  parser.add_argument("--output_path", type=str, metavar="performance_metrics_path", help="the file path where the testing performance metrics dict will be saved to ")
  parser.add_argument("--batch_size", type=int, metavar="batch_size", help="the size of the batch for the creatoing of the dataloaders")
  parser.add_argument("--num_workers", type=int, metavar="num_workers", help="the number of wrokers for multiprocessing in dataloaders")

  args = parser.parse_args()

  device = "cuda" if torch.cuda.is_available() else "cpu"

  dataset = DeepfakeDataset(parent_dir=args.parent_samples_dir, real_folder="preprocessed_real", deepfake_folder="preprocessed_fake")

  _, _, test_loader = create_dataloader_splits(
    dataset=dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    split_indices_path=args.split_indices_path,
    load_saved_indices=True
    )
  
  criterion = nn.CrossEntropyLoss()

  model = DeepfakeClassifier(
    weights=None,
    input_size=512,
    hidden_size=256,
    device=device
  )

  state_dict = torch.load(args.best_state_dict_path, map_location=device)
  model.load_state_dict(state_dict=state_dict)
  model.to(device=device)
  model.eval()

  #uses evaluate function from training_model file
  #test_accuracy, test_recall, test_precision, test_f1, test_loss = evaluate(model=model, val_loader=test_loader, criterion=nn.CrossEntropyLoss())

  all_labels = []
  all_preds = []
  tot_loss = 0

  with torch.no_grad():
    for batch, labels in test_loader:
      batch = batch.to(device)
      labels = labels.to(device)

      logits = model(batch)
      loss = criterion(logits, labels)
      tot_loss += loss.item()

      preds = torch.argmax(logits, dim=1)

      all_labels.extend(labels.cpu().numpy())
      all_preds.extend(preds.cpu().numpy())
    
  test_accuracy = accuracy_score(y_true=all_labels, y_pred=all_preds)
  test_recall = recall_score(y_true=all_labels, y_pred=all_preds)
  test_precision = precision_score(y_true=all_labels, y_pred=all_preds)
  test_f1 = f1_score(y_true=all_labels, y_pred=all_preds)
  test_loss = tot_loss / len(test_loader)

  cm = confusion_matrix(y_true=all_labels, y_pred=all_preds)

  fig, ax = plt.subplots(figsize=(6, 6))
  cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
  cm_display.plot(cmap="Blues", ax=ax, colorbar=False)
  ax.set_title("Confusion Matrix")

  fig.savefig(os.path.join(args.output_path, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
  plt.show()
  plt.close(fig)


  test_output = {
    "accuracy": test_accuracy,
    "recall": test_recall,
    "precision": test_precision,
    "f1_score": test_f1,
    "loss" : test_loss
  }

  torch.save(test_output, os.path.join(args.output_path, "testing_performance_metrics.pt"))

  print(f"test accuracy: {test_accuracy}\n")
  print(f"test recall: {test_recall}\n")
  print(f"test precision: {test_precision}\n")
  print(f"test f1_score: {test_f1}\n")
  print(f"test loss: {test_loss}\n")



