import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import imgaug.augmenters as iaa


def update_ema_variables(model, ema_model, alpha, global_step):
    # code for moving average of parameters student-teacher model
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def sigmoid_rampup(current, rampup_length):
    # Sigmoid ramp-up function over the course of training
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def subsample(x, mask):
    # sampling, extract only the values specified in the mask and the remaining values are returned as zeros
    x = torch.index_select(x, 0, mask.cuda())
    return x


def augmenter(x):
    # Basic transformations, can change accordingly
    # Among the given augmentations it always applies any 2 randomly chosen augmentations.
    seq = iaa.Sequential([iaa.SomeOf((0, 2),
                                     [
                                         iaa.Crop(px=(0, 16)),
                                         iaa.Fliplr(0.5),
                                         iaa.GaussianBlur(sigma=(0, 1.0)),
                                         iaa.Affine(translate_px=(-15, 15)),
                                         iaa.Affine(rotate=(-15, 15)),
                                         iaa.Dropout(p=(0, 0.2))
                                     ], random_order=True)
                          ])
    return seq.augment_images(x)


class Entropy(nn.Module):
    # Entropy loss, input the predictions.
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
