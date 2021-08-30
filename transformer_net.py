import torch
from functools import partial

class ConditionInstanceNorm2d(torch.nn.Module):
  def __init__(self, num_features, n_style):
    super(ConditionInstanceNorm2d, self).__init__()
    self.norm = torch.nn.InstanceNorm2d(num_features)
    
    self.gamma = torch.nn.Parameter(
      1+0.02*torch.randn(n_style, num_features, 1, 1)
    )
    self.beta = torch.nn.Parameter(
      torch.zeros(n_style, num_features, 1, 1)
    )
  
  def forward(self, input, style_idx):
    x = input
    x = self.norm(x)
    _get = partial(torch.index_select, dim=0, index=style_idx)
    x = x*_get(self.gamma) + _get(self.beta)
    return x

class TransformerNet(torch.nn.Module):
    def __init__(self, n_style):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.cin1 = ConditionInstanceNorm2d(32, n_style)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.cin2 = ConditionInstanceNorm2d(64, n_style)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.cin3 = ConditionInstanceNorm2d(128, n_style)
        # Residual layers
        self.res1 = ResidualBlock(128, n_style)
        self.res2 = ResidualBlock(128, n_style)
        self.res3 = ResidualBlock(128, n_style)
        self.res4 = ResidualBlock(128, n_style)
        self.res5 = ResidualBlock(128, n_style)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.cin4 = ConditionInstanceNorm2d(64, n_style)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.cin5 = ConditionInstanceNorm2d(32, n_style)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.cin6 = ConditionInstanceNorm2d(3, n_style)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X, style_idx):
        y = self.relu(self.cin1(self.conv1(X),style_idx))
        y = self.relu(self.cin2(self.conv2(y),style_idx))
        y = self.relu(self.cin3(self.conv3(y),style_idx))
        y = self.res1(y,style_idx)
        y = self.res2(y,style_idx)
        y = self.res3(y,style_idx)
        y = self.res4(y,style_idx)
        y = self.res5(y,style_idx)
        y = self.relu(self.cin4(self.deconv1(y),style_idx))
        y = self.relu(self.cin5(self.deconv2(y),style_idx))
        y = 255*self.sigmoid(self.cin6(self.deconv3(y),style_idx))
#         y = self.deconv3(y)
#         y = self.cin6(self.deconv3(y),style_idx)
#         y = 255*self.sigmoid(self.deconv3(y))
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, noise_std=0.01):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.noise_std = noise_std

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if (self.training and self.noise_std):
          noise_val = torch.normal(0, torch.ones_like(out)*self.noise_std)
          out = out + noise_val
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, n_style):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.cin1 = ConditionInstanceNorm2d(channels, n_style)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.cin2 = ConditionInstanceNorm2d(channels, n_style)
        self.relu = torch.nn.ReLU()

    def forward(self, x, style_idx):
        residual = x
        out = self.relu(self.cin1(self.conv1(x), style_idx))
        out = self.cin2(self.conv2(out), style_idx)
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out