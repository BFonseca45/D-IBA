from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
from torchvision import transforms
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import box_convert
from torchvision.ops import boxes as box_ops
from ultralytics.utils.ops import non_max_suppression

bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
box_coder = det_utils.BoxCoder(bbox_reg_weights)

def postprocess_detections_retina(head_outputs, anchors, image_shapes):
    # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
    class_logits = head_outputs["cls_logits"]
    box_regression = head_outputs["bbox_regression"]
    score_thresh=0.05
    topk_candidates=1000
    detections_per_img=300
    nms_thresh=0.5
    box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    num_images = len(image_shapes)

    detections: List[Dict[str, Tensor]] = []

    for index in range(num_images):
        box_regression_per_image = [br[index] for br in box_regression]
        logits_per_image = [cl[index] for cl in class_logits]
        anchors_per_image, image_shape = anchors[index], image_shapes[index]

        image_boxes = []
        image_scores = []
        image_labels = []
        image_logits = []

        for box_regression_per_level, logits_per_level, anchors_per_level in zip(
            box_regression_per_image, logits_per_image, anchors_per_image
        ):
            num_classes = logits_per_level.shape[-1]

            # remove low scoring boxes
            scores_per_level = torch.sigmoid(logits_per_level).flatten()
            keep_idxs = scores_per_level > score_thresh
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # keep only topk scoring predictions
            num_topk = det_utils._topk_min(topk_idxs, topk_candidates, 0)
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            boxes_per_level = box_coder.decode_single(
                box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
            )
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)
            image_logits.append(logits_per_level[anchor_idxs, 1:])

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_logits = torch.cat(image_logits, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, nms_thresh)
        keep = keep[: detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
                "logits": image_logits[keep],
            }
        )

    return detections

def postprocess_detections_faster(
    class_logits,
    box_regression,
    proposals,
    image_shapes,
    score_thresh = 0.05,
    nms_thresh = 0.5,
    detections_per_img = 100,
):
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = box_coder.decode(box_regression, proposals)

    pred_scores = Func.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_logits = []

    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        logits = scores.clone()

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > score_thresh)[0]
        idx,idy = torch.where(logits > score_thresh)
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        logits = logits[idx,:]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        logits = logits[keep,:]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        logits = logits[keep,:]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_logits.append(logits)

    return all_boxes, all_scores, all_labels, all_logits

def postprocess_detections_yolo(enc_preds, conf_t = 0.25):
    detections = []

    result, idx = non_max_suppression(enc_preds, return_idxs=True, conf_thres = conf_t)
    logits = enc_preds[0,4:,:]

    if not(result[0].shape[0] == 0):
        bboxes = result[0][:,:4]/2.857 # convert to the 224x224 shape
        scores = result[0][:,4]
        clas = result[0][:,-1].to(torch.int64)
        logits = logits[:, idx[0]]

        detections.append({
            "boxes": bboxes,
            "scores": scores,
            "labels": clas,
            "logits": logits.T
        })

    else:
        bboxes = enc_preds[0,:4,:]/2.857
        scores, clas = torch.max(logits, dim=0)
        clas = clas.to(torch.int64)

        detections.append({
            "boxes": bboxes.T,
            "scores": scores,
            "labels": clas,
            "logits": logits.T
        })

    return detections


def iou(box1, box2):
    box1 = box1.detach().cpu().numpy()
    box2 = box2.detach().cpu().numpy()
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])

    return intersection / (area1 + area2 - intersection)

def iou_torch(box1, box2):
    tl = torch.vstack([box1[:2], box2[:2]]).max(axis=0)[0]
    br = torch.vstack([box1[2:], box2[2:]]).min(axis=0)[0]
    intersection = torch.prod(br - tl) * torch.all(tl < br).float()
    area1 = torch.prod(box1[2:] - box1[:2])
    area2 = torch.prod(box2[2:] - box2[:2])

    return intersection / (area1 + area2 - intersection)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def kl_div(r, lambda_, mean_r, std_r):
    '''Computes the KL Divergence between the noise (Q(Z)) and
    the noised activations P(Z|R)).
    '''
    # Function provided by the authors of the paper: https://arxiv.org/abs/2001.00396
    # through https://github.com/BioroboticsLab/IBA

    # Normalizing [1]
    r_norm = (r - mean_r) / std_r

    # Computing mean and var Z'|R' [2,3]
    var_z = (1 - lambda_) ** 2
    mu_z = r_norm * lambda_

    log_var_z = torch.log(var_z)

    # For computing the KL-divergence:
    # See eq. 7: https://arxiv.org/pdf/1606.05908.pdf
    capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
    return capacity

def saliency_map(kl, shape=(224,224)):
    '''
    Generates the saliency map based on Dkl[P(Z|R)||Q(Z)]
    '''
    # scale = shape // kl.shape[1]
    map_kl = torch.sum(kl,axis=1, keepdim=True)
    map_kl = map_kl/kl.shape[1]

    # Converting to bits and upsampling
    map_kl = map_kl/float(np.log(2))
    map_kl = nn.Upsample(size=shape, mode='bilinear')(map_kl)
    # map_kl = nn.Upsample(size=shape, mode='bicubic')(map_kl)
    return map_kl


def welford_estimator(new_value, mean, count, M):
    '''
    Estimates the mean and std using the Welford online algorithm,
    link: ``https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance``
    '''
    # Function provided by the authors of the paper: https://arxiv.org/abs/2001.00396
    # through https://github.com/BioroboticsLab/IBA

    count += 1
    delta = new_value - mean
    mean += delta/count
    delta2 = new_value - mean
    M += delta*delta2

    return (count, mean, M)

def rescale_boxes(bbox, im_h, im_w, new_shape = 800):
    # bbox = box_convert(bbox, in_fmt="xywh", out_fmt="xyxy")
    factor_x = im_w/new_shape
    factor_y = im_h/new_shape

    bbox[0] = bbox[0]/factor_x
    bbox[2] = bbox[2]/factor_x

    bbox[1] = bbox[1]/factor_y
    bbox[3] = bbox[3]/factor_y

    return bbox