import torch
from torch.utils.data import dataloader
from torchvision import models, datasets, transforms


import numpy as np
import os
import time


path ='../../data/classification'
path_folder = ['train','val']

transformers = {'train': transforms.Compose([transforms.RandomResizedCrop(224),\
    transforms.RandomHorizontalFlip(),#transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),\
    'val':transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),\
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }

image_dataset = {x: datasets.ImageFolder(os.path.join(path,x), transformers[x] )\
    for x in path_folder
    }

image_dataloaders = {x: dataloader.DataLoader(image_dataset[x], batch_size=64,num_workers=os.cpu_count(), shuffle=True )
    for x in path_folder
    }

dataset_size = {x: len(image_dataset[x]) for x in path_folder}

class_names = image_dataset['train'].classes

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

