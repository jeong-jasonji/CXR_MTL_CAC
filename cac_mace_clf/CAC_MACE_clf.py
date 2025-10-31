# !/usr/bin/env python
# coding: utf-8
import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from skimage.exposure import match_histograms

from options.baseOptions import BaseOptions
from base import simpleModels, simpleDataloader, simpleClassification, simpleTransforms
from tqdm import tqdm


class makeRGB(object):
    """
    make a grayscale image RGB
    """

    def __init__(self):
        self.initalized = True

    def __call__(self, img):
        return img.convert('RGB')

def min_max_standard(img):
    # min-max stretch the image
    img += img.min()
    img = img / img.max()
    img = (img * 255).astype('uint8')
    
    return img

def make_image_input(img_path, lut, norm_hist, bone_hist, hist_norm=True):
    # read image channels and make image
    orig_img = np.invert(np.array(Image.open(img_path).convert(mode='L').resize((256,256)))) if not lut else np.array(Image.open(img_path).convert(mode='L').resize((256,256)))
    if hist_norm:
        orig_img = match_histograms(orig_img, norm_hist)
    #orig_img = min_max_standard(orig_img)
    # get the heart segmentation channel
    seg_img = np.array(Image.open(img_path.replace('.png', '_crop.png')).convert(mode='L').resize((256,256)))
    seg_img = seg_img / seg_img.max()
    seg_img[seg_img < 0.5] = 0
    seg_img[seg_img >= 0.5] = 1
    seg_img = np.abs(seg_img - 1)  # invert image
    seg_img = (orig_img * seg_img).astype('uint8')
    # get bone suppression channel
    bonesup_img = np.invert(np.array(Image.open(img_path.replace('.png', '_supp.png')).convert(mode='L').resize((256,256)))) if not lut else np.array(Image.open(img_path.replace('.png', '_supp.png')).convert(mode='L').resize((256,256)))
    if hist_norm:
        bonesup_img = match_histograms(bonesup_img, bone_hist)
    #bonesup_img = min_max_standard(bonesup_img)
    # stack them in order
    array_a_3c = np.stack((orig_img, seg_img, bonesup_img), axis=-1)
    image = Image.fromarray(array_a_3c.astype('uint8'))
    
    return image

def eval_MACE(args):
    # set random seed - for reproducibility
    random_seed = 1991
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    # initialization
    parallel_model = False
    torch.set_num_threads(1)
    device = args.device
    task1_cls = 2
    
    # load normalized histograms
    norm_hist = np.array(pickle.load(open('./cac_mace_clf/normalized_hist_array_manual.pkl', 'rb'))).reshape(1, 65410)
    bone_hist = np.array(pickle.load(open('./cac_mace_clf/normalized_bonesupp_array_manual.pkl', 'rb'))).reshape(1, 65413)
    
    # initialize model
    model = torch.load(args.trained_model, map_location=torch.device(args.device))
    model.eval()
    
    # make preprocessing and transforms
    transforms_eval = transforms.Compose([
        simpleTransforms.convertLUT(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    opt = BaseOptions(json_filepath='train.json', args=args, is_train=False).parse()
    eval_loader = simpleDataloader.simpleDataLoader(opt, args.df_path, transforms=transforms_eval, shuffle=False)
    print('evaluation size: {}'.format(len(eval_loader)))
    
    # run evaluation
    out_dict = simpleClassification.model_predict(opt, model, eval_loader, args.cac)
    out_df = pd.DataFrame(out_dict)
    
    # load the eval dataframe to merge with the output
    eval_df = pd.read_csv(args.df_path)
    merged_df = pd.concat([eval_df, out_df], axis=1)
    
    """
    softmax = torch.nn.Softmax(dim=1)
    # get items to save in output
    # set up output dictionary
    out_dict = {'dcm': [], 't1_pred': [], 't1_score_0': [], 't1_score_1': []}
    
    for j in tqdm(range(len(eval_list)), leave=False):
        lut = False
        # get original image channel
        img_path = eval_list[j]
        inputs = make_image_input(img_path, lut, norm_hist, bone_hist)
        inputs = transforms_eval(inputs)
        # evaluate image
        outputs1, outputs2 = model(inputs.unsqueeze(dim=0))
        score1 = softmax(outputs1)
        _, preds1 = torch.max(score1, 1)
        #out_dict['t1_true'].extend(labels1.data.tolist())  # don't have true mace labels yet
        out_dict['t1_pred'].append(preds1.item())
        for i in range(task1_cls):
            out_dict['t1_score_{}'.format(i)].append(score1[:, i].item())
        # save outputs
        out_dict['dcm'].append(img_path)

    mace_out = pd.DataFrame(out_dict)
    """
    
    return merged_df
