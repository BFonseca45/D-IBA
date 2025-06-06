import random

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from DIBA import *
from models import *

random.seed(10)
torch.manual_seed(10)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

core = YOLO_DIBA(conf_t=0)
# core.eval()    
core = core.to(device)

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])
    
set = torchvision.datasets.CocoDetection(root='C:/Users/bfons/OneDrive/Datasets/Coco2017/val2017/',
                                          annFile= 'C:/Users/bfons/OneDrive/Datasets/Coco2017/annotations/instances_val2017.json',
                                          transform=transform)

sub_set = Subset(set, range(0,1000))
set_loader = DataLoader(sub_set, batch_size=1, shuffle=False)

(count, mean, M) = (0,0,0)

for img in tqdm(set_loader):
    img = img[0]
    img = img.to(device)

    _ = core(img)
    act = core.get_activation()

    act = act.detach().cpu().numpy()
    act = act.flatten()

    for value in act:
        count, mean, M = welford_estimator(value, mean=mean, count=count, M=M)

std = np.sqrt(M/count)
print("mean: ", mean, "+/-", std)