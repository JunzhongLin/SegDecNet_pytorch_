import glob
from PIL import Image
import numpy as np
import cv2
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard.writer import SummaryWriter


#
# x = torch.randn(2, 2, 3, 3)
# y = torch.empty(2, 3, 3, dtype=torch.long).random_(2)
#
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
#
# loss = CrossEntropyLoss(reduction='none')
#
# res = loss(x, y)

from dataset import KolektorDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import glob
from models import SegmentNet
from torch.nn import BCEWithLogitsLoss

dataSetRoot = './Data'
img_height, img_width = (704, 256)

imgFiles = sorted(glob.glob(r'./Data/Test/*.jpg'))
maskFiles = sorted(glob.glob(r'./Data/Test/*.bmp'))


transforms_ = transforms.Compose([
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transforms_mask = transforms.Compose([
    transforms.Resize((img_height//8, img_width//8)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


dl_r = DataLoader(KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask,
                                subFold="Train_NG",),
                batch_size=2,
                shuffle=True,
                num_workers=0,
                )

ds = KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask,
                     subFold="Train_NG",)

dl = DataLoader(KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask,
                                subFold="Train_NG",),
                batch_size=2,
                shuffle=False,
                num_workers=0,
                drop_last=True
                )


true_mask = dl.dataset.__getitem__(1, aug=False)['mask']
# plt.imshow(image.permute(1,2 ,0))
#
# m



# img_list = sorted(glob.glob(r'./Data/Train_OK/*.jpg'))
mask_list = sorted(glob.glob(r'./Data/Train_NG/*.bmp'))
# img_file = img_list[0]
# img = Image.open(img_file)
# img1 = img.convert('RGB')
mask = cv2.imread(mask_list[0], cv2.IMREAD_GRAYSCALE)
mask_i = Image.fromarray(mask)
tran_mask = transforms_mask(mask_i)

mask_temp = np.random.randint(0,2, (5,5))*255
mask_temp_i = Image.fromarray(mask_temp.astype(np.uint8))
tran_mask_temp = transforms_mask(mask_temp_i)






