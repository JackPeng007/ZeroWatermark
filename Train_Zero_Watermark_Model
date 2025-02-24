import einops
import cv2
import torch
import numpy as np  # NumPy https://numpy.org/
import matplotlib.pyplot as plt  # Matplotlib https://matplotlib.org/
from scipy.optimize import curve_fit  # https://www.scipy.org/
import torch.nn as nn
from sklearn.metrics import mean_squared_error,mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms as transforms

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from PIL import Image  # 用于读取图像
import cv2  # 用于读取图像

import matplotlib.pyplot as plt
import skimage
import cv2

from Model.VQVAE import VQVAE
from Model.NUP import *


dirpath = "log"
filename = "test"
host_img = 'test.png'

host_img = np.expand_dims(cv2.imread(host_img, cv2.IMREAD_GRAYSCALE), axis = -1)

img_Patch = einops.rearrange(host_img, '(h ph) (w pw) c -> (h w) ph pw c', ph=int(256 / 8), pw=int(256 / 8))

img_Patch_Squeeze = np.squeeze(img_Patch, axis=-1)
img_Patch = np.expand_dims(img_Patch_Squeeze, axis=1)


Normalized_img_Patch = img_Patch / 255.0
process_img_Patch = torch.from_numpy(Normalized_img_Patch).float()


train_loader = torch.utils.data.DataLoader(process_img_Patch, batch_size=batch_size)

model = VQVAE(1, embedding_dim, num_embeddings)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

for epoch in range(epochs):
    print("Start training epoch {}".format(epoch, ))
    for i, images in enumerate(train_loader):
        # images = images - 0.5  # normalize to [-0.5, 0.5]
        images = images.cuda()
        loss = model(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("\t [{}/{}]: loss {}".format(i, len(train_loader), loss.item()))


param_save_path = dirpath + "/" + filename + ".pth"
torch.save(model.state_dict(), param_save_path)
