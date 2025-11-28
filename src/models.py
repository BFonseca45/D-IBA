from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                          retinanet_resnet50_fpn)
from ultralytics import YOLO

from src.DIBA import BottleneckLayer
from utils.metrics import *
from utils.utils import *


class Retina_DIBA(nn.Module):
    def __init__(self):
        super(Retina_DIBA, self).__init__()
        self.name = 'RetinaNET'

        retina = retinanet_resnet50_fpn(weights='DEFAULT')
        retina = retina.to(device)

        # disecting the model
        self.trans = retina.transform
        self.resnet = retina.backbone.body
        self.fpn = retina.backbone.fpn
        self.anchor = retina.anchor_generator
        self.head = retina.head

        # defining the bottleneck parameters
        self.shape = (1,2048,25,25)

        # mean and standard deviation for the target activation map
        self.std_r = 0.1849502604522136
        self.mu_r = 0.04322044551631088

        self.alp = torch.empty(self.shape, dtype=torch.float32)
        nn.init.constant_(self.alp, 5)
        self.alp = nn.Parameter(self.alp)

        self.gauss = torch.empty(self.shape, dtype=torch.float32)
        nn.init.normal_(self.gauss, mean=self.mu_r, std=self.std_r)
        self.gauss = self.gauss.to(device)

    def forward(self, img):

        val  = img.shape[-2:]
        original_image_sizes: List[Tuple[int, int]] = []
        original_image_sizes.append((val[0], val[1]))

        images = self.trans(img)
        images = images[0]

        self.act = self.resnet(images.tensors)

        # passing the output of the backbone through the bottleneck
        self.lamb = torch.sigmoid(self.alp)
        self.lamb = self.lamb.to(device)

        self.act['2'] = torch.mul(self.act['2'], self.lamb) + torch.mul(1 - self.lamb, self.gauss)
        self.act['2'] = torch.relu(self.act['2'])

        y = self.fpn(self.act)

        # passing through the next layers of the network and getting logits
        features = list(y.values())
        anchors = self.anchor(images,features)
        head_outputs = self.head(features)

        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs["cls_logits"].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        split_head_outputs: Dict[str, List[torch.Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        detections = postprocess_detections_retina(split_head_outputs, split_anchors, images.image_sizes)
        # detections = self.trans.postprocess(detections, images.image_sizes, original_image_sizes)

        return detections

    def reset_model(self):
        # defining the bottleneck parameters
        self.alp = nn.Parameter(torch.empty(self.shape, dtype=torch.float32))
        nn.init.constant_(self.alp, 5)

    def information_loss(self):
        self.kl = kl_div(r=self.act['2'], lambda_=self.lamb,
                          mean_r=self.mu_r, std_r=self.std_r)
        return self.kl

    def get_activation(self):
        return self.act['2']

class Faster_DIBA(nn.Module):
    def __init__(self):
        super(Faster_DIBA, self).__init__()
        self.name = 'Faster_RCNN'

        # get Faster RCNN model
        faster = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        faster = faster.to(device)

        # defining the bottleneck parameters
        self.shape = (1,2048,25,25)

        # mean and standard deviation for the target activation map
        self.std_r = 0.3402918930020553
        self.mu_r = 0.10955733663576685

        self.alp = torch.empty(self.shape, dtype=torch.float32)
        nn.init.constant_(self.alp, 5)
        self.alp = nn.Parameter(self.alp)

        self.gauss = torch.empty(self.shape, dtype=torch.float32)
        nn.init.normal_(self.gauss, mean=self.mu_r, std=self.std_r)
        self.gauss = self.gauss.to(device)

        # disecting the model
        self.transform = faster.transform
        self.resnet = faster.backbone.body
        self.fpn = faster.backbone.fpn
        self.rpn = faster.rpn
        self.roi = faster.roi_heads

        # used to get the logits
        self.box_roi_pool = faster.roi_heads.box_roi_pool
        self.box_head = faster.roi_heads.box_head
        self.box_predictor = faster.roi_heads.box_predictor

    def forward(self, x):
        x = self.transform(x)
        x = x[0]

        self.act = self.resnet(x.tensors)

        # passing the output of the backbone through the bottleneck
        self.lamb = torch.sigmoid(self.alp)
        self.lamb = self.lamb.to(device)

        self.act['3'] = torch.mul(self.act['3'], self.lamb) + torch.mul(1 - self.lamb, self.gauss)
        self.act['3'] = torch.relu(self.act['3'])

        self.y = self.fpn(self.act)
        self.z = self.rpn(x,self.y)
        out = self.roi(self.y,self.z[0],(x.image_sizes))

        pooled = self.box_roi_pool(self.y, self.z[0], x.image_sizes)
        headed = self.box_head(pooled)
        self.pred = self.box_predictor(headed)

        # getting the logits
        self.boxes, self.scores, self.labels, self.logits = postprocess_detections_faster(self.pred[0], self.pred[1],
                                                                                   self.z[0], x.image_sizes)

        out[0][0]['logits'] = self.logits[0]

        return out[0]

    def reset_model(self):
        self.alp = nn.Parameter(torch.empty(self.shape, dtype=torch.float32))
        nn.init.constant_(self.alp, 5)

    def information_loss(self):
        self.kl = kl_div(r=self.act['3'], lambda_=self.lamb,
                          mean_r=self.mu_r, std_r=self.std_r)
        return self.kl

    def get_logits(self):
        return self.logits

    def get_activation(self):
        return self.act['3']


class YOLO_DIBA(nn.Module):
    def __init__(self, mu_r = 0.0163, std_r = 0.3840, conf_t=0.25):
        super(YOLO_DIBA, self).__init__()
        self.name = 'YOLO'
        self.mu_r = mu_r
        self.std_r = std_r
        self.conf_t = conf_t        
        self.block_id = 8 # layer to place the bottleneck
        self.backbone = YOLO('yolo11n.pt')
        self.act = None

        target_block = self.backbone.model.model[self.block_id]

        bn_layer = BottleneckLayer(shape=(1,256,20,20), mu_r=self.mu_r, std_r=self.std_r, activation=None)
        # bn_layer = BottleneckLayer(shape=(1,32,160,160), mu_r=self.mu_r, std_r=self.std_r, activation=None)
        bn_layer.requires_grad_ = True

        block_bn = torch.nn.Sequential(bn_layer, target_block)
        block_bn.f = -1
        block_bn.i = self.block_id

        self.backbone.model.model[self.block_id] = block_bn
        self.backbone.model.model[self.block_id-1].register_forward_hook(self.hook)

        # self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((640,640)),
        #                     torchvision.transforms.ToTensor()])
        self.transform = torchvision.transforms.Resize((640,640))
                            

    def forward(self, x):
        self.x = x
        x_tr = self.transform(x)
        y = self.backbone.model(x_tr) 
        y_pred = postprocess_detections_yolo(y[0], conf_t = self.conf_t)

        return y_pred
    
    def hook(self, module, inputs, act):
        self.act = act  

    def get_activation(self):
        return self.act

    def information_loss(self):
        lamb = self.backbone.model.model[self.block_id][0].get_lambda()
        act = self.get_activation()

        self.kl = kl_div(r=act, lambda_= lamb,
                          mean_r=self.mu_r, std_r=self.std_r)
        return self.kl
    
    def reset_model(self):
        self.backbone.model.model[self.block_id][0].reset_alpha()