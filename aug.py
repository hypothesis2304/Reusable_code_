import numpy as np
import torch
from torch.distributions import uniform

def cutmix(s_x, s_y, t_x, t_y, alpha):
    new_x = s_x.clone()
    W, H = new_x.size(2), new_x.size(3)
    lmbd = np.random.beta(alpha, alpha)
    rX = np.random.uniform(0, W)
    rY = np.random.uniform(0, Y)
    rW = W * np.sqrt(1 - lmbd)
    rH = H * np.sqrt(1 - lmbd)
    x1 = int(np.round(max(rX - rW/2, 0)))
    x2 = int(np.round(min(rX + rW/2, W)))
    y1 = int(np.round(max(rY - rH/2, 0)))
    y2 = int(np.round(min(rY - rH/2, H)))
    new_x[:,:,x1:x2, y1:y2] = t_x[:,:,x1:x2, y1:y2]
    new_y = (lmbd * s_y) + ((1 - lmbd) * t_y)
    return new_x, new_y

def mixup(s_x, s_y, t_x, t_y, alpha):
    mu = np.random.beta(alpha, alpha)
    lmbd = torch.max(mu, 1 - mu)
    x_mix = (s_x * lmbd) + (t_x * (1 - lmbd))
    y_mix = (s_y * lmbd) + (t_y * (1 - lmbd))
    return x_mix, y_mix

