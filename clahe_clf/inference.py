import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import transforms
import torchvision
import torch
from collections import defaultdict
import os
from skimage import exposure
from skimage import io

from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import pydicom 

DEVICE = "cuda:1"


class CXR_Dataset(torch.utils.data.Dataset):
    """
        Class for loading the images and their corresponding labels.
        Parameters:
        df (pandas.DataFrame): DataFrame with image information.
        transform (callable): Data augmentation method.
    """
    def __init__(self, dataframe, transforms=None, mode=None):
        super().__init__()
        self.df = dataframe
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df["path"].iloc[idx]
        try:
            if img_path.endswith(".dcm"):
                x = pydicom.dcmread(img_path)
                x = x.pixel_array
            else:
                x = io.imread(img_path)
            x = exposure.equalize_hist(x) # Histogram equalization
            x = (((x - x.min()) / x.max() - x.min())*255).astype(np.uint8)
            x = np.stack((x, )*3)
            x = np.transpose(x, (1, 2, 0))
            
            x = self.transforms(x)
            
            return x

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.tensor([999]) 


def inference(model, dataloader, DEV):
    # run inference
    preds = []
    model.eval()
    with torch.no_grad():
        for image in tqdm(dataloader, desc="Validation"):
            if len(image.shape) != 4:
                preds.append(9)
                continue
            image = image.to(DEV)
            y_preds = model(image)
            y_preds = torch.argmax(y_preds, dim=1)
            preds.append(y_preds.cpu().item())
            
    return preds


def load_model(DEV, model_path='best_model_config.pth.tar'):
    # load the resnet50 model and its weights
    model = torchvision.models.resnet50()
    model.fc = nn.Sequential(
            nn.Linear(in_features=(model.fc.in_features),out_features=512),
            nn.Dropout(0.3),
            nn.Linear(in_features=512,out_features=256),
            nn.Dropout(0.3),
            nn.Linear(in_features=256,out_features=128),
            nn.Dropout(0.3),
            nn.Linear(in_features=128,out_features=3),
        )
    
    model = model.to(DEV)
    model_config = torch.load(model_path)
    model.load_state_dict(model_config['model_weights'])
    
    return model


def CLAHE_clf_inference(inference_df, device='cuda:0', model_path='./clahe_clf/best_model_config.pth.tar', batch_size=1, workers=4):
    # make dataloader transforms
    valid_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    # make dataset
    dataset = CXR_Dataset(inference_df, valid_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    # load the model
    model = load_model(device, model_path=model_path)
    # run inference
    outputs = inference(model, dataloader, device)
    # output dataframe
    inference_df['CLAHE_pred'] = outputs
    
    return inference_df
    