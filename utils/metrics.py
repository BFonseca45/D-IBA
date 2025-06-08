import numpy as np
import torch
import cv2
from tqdm import tqdm

from utils.utils import *

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
np.random.seed(10)

class EvalMetric():
    def __init__(self, model, img, target_index, truth_box) -> None:
        if model.name == 'YOLO':
            model.conf_t = 0     # return even the least confident predictions

        self.model = model
        self.truth_box = truth_box
        self.img = img
        self.target_index = target_index

    def get_tiles(self, saliency, STEP=8):
        self.STEP = STEP
        self.importance = []
        self.coords = []

        for i in range(0, len(saliency[0,:]), self.STEP):
            for n in range(0, len(saliency[:,0]), self.STEP):
                aux = np.sum(saliency[i:i+self.STEP, n:n+self.STEP])
                self.importance.append(aux)
                self.coords.append([i, n])

        return self.coords, self.importance

    def sort_coordenates(self, reverse = True):
        importance_sorted = np.unique(self.importance)
        importance_sorted = sorted(importance_sorted, reverse=reverse) # Change this line to remove the least important pixels first

        coords_sorted = []
        for t in range(len(importance_sorted)):
            ind = np.where(self.importance == importance_sorted[t])
            for p in range(len(ind[0])):
                coords_sorted.append(self.coords[ind[0][p]])

        return coords_sorted

    def compute_sim(self, image):
        score_threshold = 0.1
        with torch.no_grad():
          out = self.model(image)[0]        

        box = out['boxes']
        logits = out['logits']
        labels = out['labels']

        scores = torch.nn.functional.softmax(logits, dim=1)
        scores = scores[:, self.target_index-1]
        
        similarity = torch.tensor(0)

        if any(labels == self.target_index):            
            box = box[labels == self.target_index]      
            scores = scores[labels == self.target_index]      

            for t in range(len(box)):
                loc = iou(self.truth_box, box[t])
                if similarity < loc*scores[t]:
                    similarity = loc*scores[t]                

        elif any(scores > score_threshold):
            box = box[scores > score_threshold]            
            scores = scores[labels > score_threshold]

            for t in range(len(box)):
                loc = iou(self.truth_box, box[t])
                if similarity < loc*scores[t]:
                    similarity = loc*scores[t]         

        return similarity

    def first_score(self):
        score = []
        score.append(self.compute_sim(image = self.img).detach().cpu().numpy())

        return score

    def compute_score(self, metric):
        if metric == 'MoRF':
            mask = torch.ones_like(self.img, dtype=torch.float32)
            ocl = 0
            coords_sorted = self.sort_coordenates(reverse=True)
            score = self.first_score()

        elif metric == 'LeRF':
            mask = torch.ones_like(self.img, dtype=torch.float32)
            ocl = 0
            coords_sorted = self.sort_coordenates(reverse=False)
            score = self.first_score()

        for i in range(len(coords_sorted) - 1):
            idx = coords_sorted[i][0]
            idy = coords_sorted[i][1]
            mask[:,:,idx:idx+self.STEP, idy:idy+self.STEP] = ocl

            # masking the input image
            img_in = self.img * mask
            # img_in = torch.Tensor(img_in)
            img_in = img_in.to(device)

            similarity = self.compute_sim(image=img_in)
            score.append(similarity.detach().cpu().numpy())

        score = np.array(score, dtype=np.float32)
        score = (score - min(score))/(max(score) - min(score))

        x = np.linspace(start=0, stop=1, num=len(score))
        area = np.trapz(score,x)

        return score, area

    def gauss_deg(self, saliency, scale_shape):
        imgg = self.img.detach().cpu().numpy()
        # saliency = saliency.detach().cpu().numpy()

        similarity0 = self.compute_sim(image=self.img)

        noise = np.random.randn(1,3,scale_shape,scale_shape)*np.std(imgg) + np.mean(imgg)

        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        saliency_norm = np.stack((saliency_norm, saliency_norm, saliency_norm))
        saliency_norm = saliency_norm[None,...]

        img_in = saliency_norm*imgg + (1 - saliency_norm)*noise
        img_in = torch.Tensor(img_in)
        img_in = img_in.to(device)

        similarity = self.compute_sim(image=img_in)
        ratio = (similarity + 1e-6)/(similarity0 + 4e-6)
        ratio = torch.clip(ratio, 0, 1)

        return img_in, ratio.detach().cpu().numpy()

    def pointing_game(self, box_coord, saliency):        
        saliency_new = np.squeeze(saliency)

        ind = np.unravel_index(np.argmax(saliency_new), saliency_new.shape)
        if (ind[1] > box_coord[0] and ind[1] < box_coord[2]) and (ind[0] > box_coord[1] and ind[0] < box_coord[3]):
            return 1, ind
        else:
            return 0, ind