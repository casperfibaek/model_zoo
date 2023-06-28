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

class LowerMSE(nn.Module):
    """
    Calculates MSE at a lower resolution.
    If input is 64x64, then the MSE is calculated at 32x32 using a 3x3 average pooling with same padding.
    """
    def __init__(self):
        super(LowerMSE, self).__init__()
        self.pad = nn.ReplicationPad2d(1)
        self.pool = nn.AvgPool2d(3, stride=2)

    def forward(self, y_pred, y_true):
        y_pred_padded = self.pad(y_pred)
        y_true_padded = self.pad(y_true)

        y_pred_low = self.pool(y_pred_padded)
        y_true_low = self.pool(y_true_padded)

        mse = ((y_pred_low - y_true_low) ** 2).mean()

        return mse


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