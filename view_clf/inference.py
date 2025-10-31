import argparse
import torch
from .DataLoader import OSADataset, transforms
from .utils.data_model import Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

import torch, torch.nn as nn, torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
from .utils import utils
from tqdm.auto import tqdm
import pickle

# python inference.py

class Test:
    def __init__(self, test_loader, model, device, model_path):
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.model_path = model_path

    def test(self, ):
        best_path = os.path.join(self.model_path) #enter your model path

        weights = self.model.state_dict()

        load_weights = list(torch.load(best_path, map_location=torch.device('cpu'))['model_state'].items())
        i=0
        for k, _ in weights.items():
            weights[k] = load_weights[i][1]
            i += 1

        self.model.load_state_dict(weights)
        self.model.eval().to(self.device)
        softmax = nn.Softmax()
        
        y_pred = []
        y_score = []
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                if len(data.shape) != 4:
                    y_pred += list([9])
                    y_score += list([1.0])
                    continue
                data = data.to(self.device)
                output = self.model(data)
                preds = torch.argmax(output, dim=1).cpu().detach().numpy().astype(int)
                y_pred += list(preds)
                # add a softmax layer to get probabilities
                scores = softmax(output)
                score = scores.squeeze()[0].cpu().detach().numpy()
                y_score.append(score)
                      
            return y_pred, y_score


def view_clf_inference(inference_df, model_path='./view_clf/best.pth.tar', model_name='Resnet18', device='cuda:3'):
    batch_size = 1
    num_workers = 1
    
    datagen = OSADataset(df=inference_df, transform=transforms, mode="test")
    dataloaders = torch.utils.data.DataLoader(dataset=datagen, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    tester = Test(
        test_loader=dataloaders,
        model=Model(model_name),
        device=device,
        model_path=model_path # enter your model path
        )
    y_pred, y_score = tester.test()
    #print(outputs)
    
    inference_df['view_pred'] = y_pred  # labels: AP = 0, lateral = 1
    #inference_df['view_pred_score'] = y_score
    
    return inference_df
    
