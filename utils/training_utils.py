import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize


def patience_calculator(epoch, t_0, t_m, max_patience=50):
    """ Calculate the patience for the scheduler. """
    if epoch <= t_0:
        return t_0

    p = [t_0 * t_m ** i for i in range(100) if t_0 * t_m ** i <= epoch][-1]
    if p > max_patience:
        return max_patience

    return p



class TiledMSE(nn.Module):
    """
    Calculates the MSE at full image level and at the pixel level and weights the two.
    result = (sum_mse * (1 - bias)) + (mse * bias)
    """
    def __init__(self, bias=0.2):
        super(TiledMSE, self).__init__()
        self.bias = bias

    def forward(self, y_pred, y_true):
        y_pred_sum = torch.sum(y_pred, dim=(1, 2, 3)) / (y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3])
        y_true_sum = torch.sum(y_true, dim=(1, 2, 3)) / (y_true.shape[1] * y_true.shape[2] * y_true.shape[3])

        sum_mse = ((y_pred_sum - y_true_sum) ** 2).mean()
        mse = ((y_pred - y_true) ** 2).mean()

        weighted = (sum_mse * (1 - self.bias)) + (mse * self.bias)
        
        return weighted 
    
           
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 

        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)

        return self.gamma * (x * Nx) + self.beta + x
