import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from utils.metrics import *
from utils.utils import *

class BottleneckLayer(nn.Module):
    def __init__(self, mu_r, std_r, shape=(1,128,40,40), activation='relu'):
        super(BottleneckLayer, self).__init__()
        self.shape = shape

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'silu':
            self.activation = torch.nn.SiLU()
        elif activation == None:
            self.activation = None
        else:
            raise AssertionError("Provide a valid activation function, options: ['relu', 'silu', 'None']")

        self.alp = torch.empty(shape, dtype=torch.float32)
        nn.init.constant_(self.alp, 5)
        self.alp = nn.Parameter(self.alp)

        self.gauss = torch.empty(shape, dtype=torch.float32)
        nn.init.normal_(self.gauss, mean=mu_r, std=std_r)
        self.gauss = self.gauss.to(device)

    def forward(self,x):
        self.lamb = torch.sigmoid(self.alp)
        self.lamb = self.lamb.to(device)

        x = torch.mul(x, self.lamb) + torch.mul(1 - self.lamb, self.gauss)
        if self.activation:
            x = self.activation(x)
        return x

    def reset_alpha(self):
        self.alp = nn.Parameter(torch.empty(self.shape, dtype=torch.float32))
        nn.init.constant_(self.alp, 5)

    def get_lambda(self):
        return self.lamb

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, regression, cross, information, gamma, sigma, beta):
        return gamma*regression + sigma*cross + beta*information

def train_bottleneck(model, metric_model, img, gamma, sigma, beta, scale_shape, epochs=5, run_metrics=False):
    P = []
    G = []
    MoRF = []
    LeRF = []
    Times = []
    count = 0

    score_threshold = 0.2

    loss_inf = CustomLoss()
    loss_smooth = nn.SmoothL1Loss()
    loss_ce = nn.CrossEntropyLoss()

    # Detector output
    if metric_model.name == 'YOLO':
        metric_model.conf_t = score_threshold
    pred_c = metric_model(img)
    scr_c = pred_c[0]['scores']
    boxes_c = pred_c[0]['boxes']
    labels_c = pred_c[0]['labels']

    boxes_c = boxes_c[scr_c > score_threshold]
    labels_c = labels_c[scr_c > score_threshold]

    img_ex = img.expand((epochs,3,scale_shape,scale_shape))
    gauss = torch.empty((epochs,3,scale_shape,scale_shape))
    nn.init.normal_(gauss, mean=0.0, std=0.01)
    gauss = gauss.to(device)

    img_noise = img_ex + gauss 

    for j in range(len(labels_c)):
        truth_box = boxes_c[j].to(device)
        truth_box.retain_graph = True

        target_class_index = labels_c[j]
        truth_class = nn.functional.one_hot(target_class_index-1, 90).view((1,-1)).double().to(device) # YOLO was trained with only 80 classes
        # truth_class = nn.functional.one_hot(target_class_index, 80).view((1,-1)).double().to(device)

        # performing the bottleneck, this part here might not be needed for yolo 
        for p in model.parameters():
            p.requires_grad = False
        opt_params = list(model.parameters())
        opt_params[0].requires_grad = True

        optm = torch.optim.Adam(opt_params, lr = 0.5)
        optm.zero_grad()

        tic = time.time()

        for i in range(epochs):
            min_loss = np.inf
            optm.zero_grad()

            # getting the most closest prediction to the original one
            aux = img_noise[i,...]
            aux = aux[None,...]
            pred = model(aux)

            boxes = pred[0]['boxes']
            logits = pred[0]['logits']
            labels = pred[0]['labels']
            labels = labels.detach().cpu().numpy()

            ind = 0
            for box in boxes:
                smooth = loss_smooth(truth_box, box)
                if min_loss > smooth:
                    min_loss = smooth
                    ind_log = ind
                ind = ind+1

            logits_right = logits[ind_log,:]
            logits_right = logits_right[None,:]

            # Optmizing the bottleneck
            cross_entr = loss_ce(logits_right, truth_class)
            information = torch.mean(model.information_loss())

            # print(min_loss*gamma)
            # print(cross_entr*sigma)
            # print(information*beta)
            # print()

            loss = loss_inf(min_loss, cross_entr, information, gamma, sigma, beta)
            loss.backward(retain_graph=True)

            optm.step()

        heatmap = saliency_map(model.information_loss(), shape=(scale_shape,scale_shape))        
        heatmap = heatmap.clone().detach().cpu().numpy()
        heatmap = np.squeeze(heatmap)

        toc = time.time()

        if run_metrics:        
            metric = EvalMetric(model=metric_model, img=img, target_index=target_class_index, truth_box = truth_box)
            # metric = EvalMetric(img=img, target_index=target_class_index, truth_box = truth_box)
            _, _ = metric.get_tiles(saliency=heatmap, STEP=8)

            _, g = metric.gauss_deg(saliency=heatmap, scale_shape=scale_shape)
            pointing, _ = metric.pointing_game(box_coord=truth_box, saliency=heatmap)

            _, area_MoRF = metric.compute_score(metric='MoRF')
            _, area_LeRF = metric.compute_score(metric='LeRF')

            MoRF.append(area_MoRF)
            LeRF.append(area_LeRF)
            P.append(pointing)
            G.append(g)
            Times.append(toc-tic)

            model.reset_model()

        else:

            imgp = img*255
            imgp = torch.squeeze(imgp)

            truth_box = rescale_boxes(bbox=truth_box, im_h=224, im_w=224, new_shape=scale_shape)
            truth_box = truth_box[None,...]
            result = draw_bounding_boxes(imgp.to(torch.uint8), boxes=truth_box, colors='yellow', width=3)
            show(result)

            plt.imshow(heatmap, cmap='turbo', alpha=0.5)
            plt.show()

            count += 1
            if count > 5:
                break

            model.reset_model()

    if run_metrics:
        return MoRF, LeRF, P, G, Times
    else:
        return None