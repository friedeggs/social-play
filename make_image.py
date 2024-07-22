import numpy as np
import cv2
import torch
from torchvision.utils import make_grid

# n_img = 10
# nrow = 2
n_img = 3
nrow = 1
ims = []
# for i in range(1, n_img+1):
for i in [2,5,8]:
	im = cv2.imread(f'gym_multigrid-{i}.png')
	print(im.shape)
	im = im[60:425,145:510]
	print(im.shape)
	im = torch.tensor(im).permute(2,0,1)#.unsqueeze(0)
	ims.append(im)

img = make_grid(ims, nrow=nrow, padding=5, pad_value=255)
img = img.permute(1,2,0).detach().numpy()
print(img.shape)
cv2.imwrite(f'gym_multigrid-{n_img}-{nrow}.png', img)