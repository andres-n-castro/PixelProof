import torch.nn as nn
from torchvision.models.resnet import resnet18
import torchvision.models as models


class DeepfakeClassifier(nn.Module):
  def __init__(self, weights: models.ResNet18_Weights, input_size: int, hidden_size: int, device: str):
    super().__init__()
    self.resnet_model = resnet18(weights=weights, progress=True)
    self.resnet_body = nn.Sequential(*list(self.resnet_model.children())[:-1]) #obtains resnet body WITHOUT head
    for param in self.resnet_body.parameters():
      param.requires_grad = False

    self.lstm_head = LSTMHead(hidden_size=hidden_size, input_size=input_size, device=device)
  
  #will probably need my own version since this calls the forward prop through not Resnet body + Lstm layer
  def forward(self, x):

    B, F, C, H, W = x.size()
    x = x.view(B*F, C, H, W)

    features = self.resnet_body(x)

    features = features.view(B, F, -1)

    out = self.lstm_head(features)
    
    return out

  def unfreeze_resnet_layers(self, num_blocks=None):
    if num_blocks == 1:
      for param in self.resnet_body[-2].parameters():
        param.requires_grad = True

    elif num_blocks == 2:
      for param in self.resnet_body[-2].parameters():
        param.requires_grad = True
      for param in self.resnet_body[-3].parameters():
        param.requires_grad = True 
    else:
      return None



class LSTMHead(nn.Module):
  def __init__(self, hidden_size: int, input_size: int, device: str):
    super().__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.num_layers = 2
    self.dropout = 0.5 if self.num_layers > 1 else 0
    self.bias = True
    self.lstm = nn.LSTM(
      hidden_size=self.hidden_size,
      input_size=self.input_size,
      dropout=self.dropout,
      bias=self.bias,
      num_layers=self.num_layers,
      batch_first=True,
      device=device
    )

    self.fc = nn.Linear(hidden_size, 2)

  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out[:, -1, :])

    return out