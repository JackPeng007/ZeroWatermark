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
from PIL import Image
from utils import *


host_img = 'test.png'
dirpath = "log"
filename = ""
Params_Position = ".pth"
save_Name = ""

import numpy as np
import matplotlib.pyplot as plt



model = VQVAE(1, embedding_dim, num_embeddings)

model.load_state_dict(torch.load(Params_Position))
model = model.cuda()

model.eval()


host_img = np.expand_dims(cv2.imread(host_img, cv2.IMREAD_GRAYSCALE), axis = -1)
water_mark_img = create_watermark_matrix()

img_Patch = einops.rearrange(host_img, '(h ph) (w pw) c -> (h w) ph pw c', ph=int(256 / 8), pw=int(256 / 8))

img_Patch_Squeeze = np.squeeze(img_Patch, axis=-1)
img_Patch = np.expand_dims(img_Patch_Squeeze, axis=1)

Normalized_img_Patch = img_Patch / 255.0
process_img_Patch = torch.from_numpy(Normalized_img_Patch).float()


with torch.no_grad():
    e_img, recon_images_img = model(process_img_Patch.cuda().float())

Ready_to_e_img = e_img.cpu().numpy()

All = np.zeros((32, 32, 1))
length = Ready_to_e_img.shape[0]

for each_e_img in Ready_to_e_img:
    img_features = einops.rearrange(np.expand_dims(each_e_img, axis=-1),
                                    '(b1 b2) h w c -> (b1 h) (b2 w) c ', b1=4, b2=4)
    All += img_features


average_features = All / length

seed_value = np.mean(water_mark_img.flatten())

chaotic_array = generate_chaotic_array(seed_value)
Coor_Matrix = Coord_Data(average_features.flatten(), chaotic_array)

Zero_Watermark = extract_Image_WaterMark(Coor_Matrix, water_mark_img)

from PIL import Image

image = Image.fromarray((Zero_Watermark * 255).astype(np.uint8))

image.save(save_Name)







