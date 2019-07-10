import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb



## code for moving average of parameters student-teacher model
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

## Entropy function for unlabelled data



## Sigmoid ramp-up function over the course of training
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

## sampling, extract only the values specified in the mask and the remaining values are returned as zeros
def subsample(x, mask):
    x = torch.index_select(x, 0, mask.cuda())
    return x

