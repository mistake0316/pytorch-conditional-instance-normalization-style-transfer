from torchvision import models
import torch

class VGG(torch.nn.Module):
  def __init__(
    self,
    requires_grad=False,
    model_name="vgg16",
  ):
    super(VGG, self).__init__()
    assert "vgg" in model_name
    
    vgg_features = getattr(models, model_name)(pretrained=True).features
    
    features = self.features = torch.nn.ModuleDict()
    
    block=1
    idx=1
    for layer in vgg_features:
      if isinstance(layer, torch.nn.Conv2d):
        features[f"conv{block}_{idx}"] = layer
      elif isinstance(layer, torch.nn.ReLU):
        features[f"relu{block}_{idx}"] = torch.nn.ReLU(inplace=False)
        idx += 1
      elif isinstance(layer, torch.nn.MaxPool2d):
        features[f"pool{block}"] = layer
        idx = 1
        block += 1
      else:
        raise ValueError(f"{type(layer)} is not allow now.")
    
    if not requires_grad:
      for param in self.parameters():
        param.requires_grad=False
  
  def forward(self, input, outkeys=None):
    outdict = {}
    x = input
    for name, layer in self.features.items():
      x = layer(x)
      if outkeys and name in outkeys:
        outdict[name] = x
    if not outkeys:
      return x
    else:
      return outdict