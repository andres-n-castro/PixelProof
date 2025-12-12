import torch
import torch.nn as nn
import torchvision.models as models

#LSTM class head implementation
class LSTMHead(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.num_classes = num_classes
    self.lstm = nn.LSTM(
      self.input_size, 
      self.hidden_size, 
      self.num_layers, 
      batch_first=True,
      bidirectional=True,
      dropout=0.5 if self.num_layers > 1 else 0
    )

    #!why did I multiply hidden size by 2????
    self.out_layer = nn.Linear(self.hidden_size*2, self.num_classes)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    out,_ = self.lstm(x)
    
    #!how does this mean pooling work?
    mean_pooling = torch.mean(out, dim=1)
    out = self.dropout(mean_pooling)
    out = self.out_layer(out)
    return out

class DeepfakeDetector(nn.Module):
  def __init__(self, resnet_weights, resnet_prog, input_size, hidden_size, num_layers, num_classes):
    super().__init__()

    self.resnet_model = models.resnet34(weights=resnet_weights, progress=resnet_prog)
    for param in self.resnet_model.parameters():
      param.requires_grad = False

    self.resnet_body = nn.Sequential(*list(self.resnet_model.children())[:-1])

    self.lstm_head = LSTMHead(input_size, hidden_size, num_layers, num_classes)

  def forward(self, x):
    batch_size, seq_len, c, w, h= x.size()

    #fold input tnesor
    x = x.view(batch_size*seq_len, c, w, h)

    features = self.resnet_body(x)

    #unfold output tensor
    features = features.view(batch_size, seq_len, -1)

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
