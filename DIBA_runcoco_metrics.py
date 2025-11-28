import random

import torch
import torch.nn as nn
import torchvision
import pickle 
import hydra

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.models import *
from src.DIBA import *

from utils.metrics import *
from utils.utils import *


@hydra.main(version_base=None, config_path='.', config_name='config.yaml')
def run_main(cfg:DictConfig) -> None:     
    random.seed(10)
    torch.manual_seed(10)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)
    print(device)
    scale_shape = 224
    
    k = 3*scale_shape*scale_shape
    # passing the config arguments to variables
    beta = cfg.param.beta/k
    gamma = cfg.param.gamma/k
    sigma = cfg.param.sigma/k

    dataset_name = cfg.db.dataset
    architecture = cfg.db.model

    run_metrics = cfg.run_metrics

    print('Running metrics for the ' + architecture + ' architecture on the ' + dataset_name + ' dataset with beta equals to ' + str(beta))

    if dataset_name == 'Coco':
        dset = torchvision.datasets.CocoDetection(root='C:/Users/bfons/OneDrive/Documents/Datasets/Coco2017/val2017/',
                                                annFile= 'C:/Users/bfons/OneDrive/Documents/Datasets/Coco2017/annotations/instances_val2017.json',
                                                transform=transforms.ToTensor())
    elif dataset_name == 'VOC':    
        dset = torchvision.datasets.VOCDetection(root='./VOC', image_set='val', download=True, transform=transforms.ToTensor())
        
    # defining dataloader
    sub_set = Subset(dset, range(0,1500))
    set_loader = DataLoader(sub_set, batch_size=1, shuffle=False, num_workers=0)

    # initialize faster model 
    if architecture == 'Faster':
        core = Faster_DIBA()
        core.eval()    
        core = core.to(device)

        # initialize metric model
        metric_model = Faster_DIBA()
        metric_model.eval()
        metric_model = metric_model.to(device)

    elif architecture == 'Retina':
        core = Retina_DIBA()
        core.eval()    
        core = core.to(device)

        # initialize metric model
        metric_model = Retina_DIBA()
        metric_model.eval()
        metric_model = metric_model.to(device)

    elif architecture == 'YOLO':
        # initialize faster model 
        core = YOLO_DIBA(conf_t = 0)
        core = core.to(device)

        # initialize faster model 
        metric_model = YOLO_DIBA()
        metric_model = metric_model.to(device)

    # computing the loss and the gradients
    loss_inf = CustomLoss()
    loss_smooth = nn.SmoothL1Loss()
    loss_ce = nn.CrossEntropyLoss()

    scale_shape = 224
    k = 3*scale_shape*scale_shape

    P = []
    G = []
    Morf = []
    Lerf = []
    Times = []

    score_threshold = 0.2
    epochs = 20
    run_metrics = True

    i = 0 
    for img in tqdm(set_loader):
        i = i+1
        img = img[0]
        img = transforms.Resize((scale_shape,scale_shape))(img)
        img = img.to(device)

        if run_metrics:
            m, l, p, g, t = train_bottleneck(model = core, metric_model = metric_model, img = img, gamma = gamma, sigma = sigma, beta = beta, scale_shape=scale_shape, epochs=epochs, run_metrics=run_metrics)
            Morf.extend(m)
            Lerf.extend(l)
            P.extend(p)
            Times.extend(t)
            G.extend(g)
        else:
            train_bottleneck(model = core, metric_model = metric_model, img = img, gamma = gamma, sigma = sigma, beta = beta, scale_shape=scale_shape, epochs=epochs, run_metrics=run_metrics)

        if not i % 20:
            metric_dict = {"Morf": Morf, "Lerf": Lerf, "Pointing": P, "Gaussian": G, "Time": Times}
            out_dir = './out/metrics' + dataset_name + '/'
            savefile = out_dir + architecture +  '_beta_' + str(int(beta*k)) + "_gamma_" + str(int(gamma*k)) + "k.pkl"
            open_file = open(savefile, 'wb')
            pickle.dump(metric_dict, open_file)
            open_file.close()

if __name__ == "__main__":
    run_main()