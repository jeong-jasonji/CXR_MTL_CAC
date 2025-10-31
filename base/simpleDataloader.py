import os
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

from skimage.exposure import match_histograms

def simpleDataLoader(opt, dataframe_path, transforms=None, shuffle=True):
    """
    A simple dataloader class
        - opt is an argparser
    """
    dataset = simpleDataset(opt=opt, dataframe_path=dataframe_path, transform=transforms)
    if opt.weighted_sampling and shuffle == True:
        w_sampler = simpleWeightedSampler(dataset)
        shuffle = False
    else:
        w_sampler = None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.workers, sampler=w_sampler)

    return dataloader

def simpleWeightedSampler(dataset):
    """
    Makes a weighted sampler based on the input dataset.
    Computes class weights for each class based on its sample distributions.
    Requires the class labels to be numerical and integers.
    """
    # load the dataframe for the dataset and get the weights
    dataset_df = dataset.label
    cls_weights = {}
    for i in dataset_df.label.unique():
        cls_weights[i] = 1.0 / dataset_df.label[dataset_df.label == i].count()

    # make a tensor of the weights for each sample
    sample_weights = []
    for img_cls in dataset_df.label.to_list():
        sample_weights.append(cls_weights[img_cls])
    sample_weights = torch.tensor(sample_weights)
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

class simpleDataset(Dataset):
    """
    A simple custom dataset class
    """
    def __init__(self, opt, dataframe_path, transform=None):
        """
        Args:
            dataframe_path (string): Path to the dataframe with image names, labels, and other information.
                - Can be a pickle (.pkl) or CSV (.csv)
            transform (callable, optional): Optional transform(s) to be applied on a sample.
                - Transforms shoud have the traditional transforms and any on-the-fly processing
        """
        # load the csv
        self.label = pd.read_pickle(dataframe_path) if '.pkl' in dataframe_path else pd.read_csv(dataframe_path)
        # check if there's manual checking
        if opt.manual_check:
            self.label = self.label[self.label['manual_check']]
        if opt.df_labels is not None:
            self.label.rename(columns=opt.df_labels, inplace=True)
        if opt.df_filter is not None:
            self.label = self.label[self.label[opt.df_filter[0]] == opt.df_filter[1]]
        if opt.cls_select is not None:  # if selecting specific classes
            cls_ids = [int(i) for i in opt.cls_select.split(',')]
            self.label = self.label[self.label['img_label'].isin(cls_ids)].reset_index(drop=True)
            for i in range(len(cls_ids)):
                self.label['img_label'] = self.label['img_label'].apply(lambda x: i if x == cls_ids[i] else x)
        
        # reset index
        self.label = self.label.reset_index(drop=True)
        self.transform = transform
        self.opt = opt
        self.to_tensor = transforms.ToTensor()
        # check if lut is in columns and apply if necessary
        self.apply_lut = True if 'lut' in self.label.columns else False
        # apply histogram normalization
        if opt.hist_norm is not None and opt.manual_check:
            self.opt.norm_hist = np.array(pickle.load(open(opt.hist_norm, 'rb'))).reshape(1, 65414) if 'lateral' in opt.hist_norm else np.array(pickle.load(open(opt.hist_norm, 'rb'))).reshape(1, 65410)
            self.opt.bone_hist = np.array(pickle.load(open(opt.hist_norm.replace('_hist_', '_bonesupp_'), 'rb'))).reshape(1, 65413)
        elif opt.hist_norm is not None and not opt.manual_check:
            try:
                self.opt.norm_hist = np.array(pickle.load(open(opt.hist_norm, 'rb'))).reshape(1, 65414) if 'lateral' in opt.hist_norm else np.array(pickle.load(open(opt.hist_norm, 'rb'))).reshape(1, 65409)
                self.opt.bone_hist = np.array(pickle.load(open(opt.hist_norm.replace('_hist_', '_bonesupp_'), 'rb'))).reshape(1, 65407)
            except:
                self.opt.norm_hist = np.array(pickle.load(open(opt.hist_norm, 'rb'))).reshape(1, 65414) if 'lateral' in opt.hist_norm else np.array(pickle.load(open(opt.hist_norm, 'rb'))).reshape(1, 65410)
                self.opt.bone_hist = np.array(pickle.load(open(opt.hist_norm.replace('_hist_', '_bonesupp_'), 'rb'))).reshape(1, 65413)
        else: 
            self.opt.norm_hist = opt.hist_norm
            
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        img_path = os.path.join(self.label['cropped_path'].iloc[idx])
        #invert = True if self.apply_lut and self.label['lut'].iloc[idx] else False
        invert = True if self.label['path'].iloc[idx] == 'MONOCHROME1' else False
        
        if self.opt.segmentation == 'channel':
            # adding segmentation and bone suppression in the channels 
            orig_img = np.invert(np.array(Image.open(img_path).convert(mode='L').resize((256,256)))) if invert else np.array(Image.open(img_path).convert(mode='L').resize((256,256)))
            if self.opt.hist_norm is not None:
                orig_img = match_histograms(orig_img, self.opt.norm_hist)
            # segment the heart
            seg_img = np.array(Image.open(img_path.replace('.png', '_crop.png')).convert(mode='L').resize((256,256)))
            seg_img = seg_img / seg_img.max()
            seg_img[seg_img < 0.5] = 0
            seg_img[seg_img >= 0.5] = 1
            seg_img = np.abs(seg_img - 1)  # invert image
            seg_img = (orig_img * seg_img).astype('uint8')
            # get the bone suppressed image
            bonesup_img = np.invert(np.array(Image.open(img_path.replace('.png', '_supp.png').replace('processed_pci', 'regular_bone_supp')).convert(mode='L').resize((256,256)))) if invert else np.array(Image.open(img_path.replace('.png', '_supp.png').replace('processed_pci', 'regular_bone_supp')).convert(mode='L').resize((256,256)))
            if self.opt.hist_norm is not None:
                bonesup_img = match_histograms(bonesup_img, self.opt.bone_hist)
            # stack them in order
            array_a_3c = np.stack((orig_img, seg_img, bonesup_img), axis=-1)
            image = Image.fromarray(array_a_3c.astype('uint8'))
        elif self.opt.segmentation == 'full':
            # segmenting only the lungs
            image_array = np.invert(np.array(Image.open(img_path).convert(mode='L'))) if invert else np.array(Image.open(img_path).convert(mode='L'))
            if self.opt.hist_norm is not None:
                image_array = match_histograms(image_array, self.opt.hist_norm)
            cropped_img = np.array(Image.open(img_path.replace('.png', '_crop.png')).convert(mode='L'))
            # convert to 0/1
            cropped_img = cropped_img / cropped_img.max()
            cropped_img[cropped_img < 0.5] = 0
            cropped_img[cropped_img >= 0.5] = 1
            segmented_img = image_array * cropped_img
            image = Image.fromarray(segmented_img)
        else:
            # temp for view classification
            img_array = np.array(Image.open(img_path))
            img_array = img_array / img_array.max()
            img_array = (img_array * 255).astype('uint8')
            if invert:
                img_array = np.invert(img_array)
            image = Image.fromarray(img_array)
            #image = Image.fromarray(np.invert(np.array(Image.open(img_path)))) if invert else Image.open(img_path)
        # save the original image for extract if needed
        original_img = orig_img if self.opt.segmentation == 'channel' else self.to_tensor(image)
        if self.opt.parallel_model:
            img_label_1 = self.label['img_label_1'].iloc[idx]
            img_label_2 = self.label['img_label_2'].iloc[idx]
        else:
            img_label = self.label['label'].iloc[idx]
        img_id = self.label['path'].iloc[idx].split('/')[-1]
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.opt.data_vis and self.opt.parallel_model:
            sample = (original_img, image, img_label_1, img_label_2, img_id)
        elif self.opt.data_vis:
            sample = (original_img, image, img_label, img_id)
        elif self.opt.joint_fusion:
            img_meta = torch.from_numpy(np.array(self.label[list(self.label.columns[-self.opt.n_EHR:])].iloc[idx])).type(torch.float32) 
            sample = (image, img_label, img_id, img_meta)
        elif self.opt.parallel_model:
            sample = (image, img_label_1, img_label_2, img_id, 0)
        else:
            sample = (image, img_label, img_id, 0)

        return sample