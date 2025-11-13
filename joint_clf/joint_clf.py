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

def eval_MACE_joint(eval_list, test_csv=None):
    # set random seed - for reproducibility
    random_seed = 1991
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    # initialization
    parallel_model = True
    torch.set_num_threads(1)
    device = 'cpu'
    task1_cls = 2
    
    # load normalized histograms
    norm_hist = np.array(pickle.load(open('./cac_mace_clf/normalized_hist_array_manual.pkl', 'rb'))).reshape(1, 65410)
    bone_hist = np.array(pickle.load(open('./cac_mace_clf/normalized_bonesupp_array_manual.pkl', 'rb'))).reshape(1, 65413)
    
    # initialize model
    model = torch.load('./joint_clf/joint_clf.pth', map_location=torch.device('cpu'))
    model.eval()
    
    # load the ehr csv
    ehr_csv = pd.read_csv('./joint_clf/mapped_ehr.csv')
    
    # make preprocessing and transforms
    transforms_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    softmax = torch.nn.Softmax(dim=1)
    # get items to save in output
    # set up output dictionary
    out_dict = {'dcm': [], 't1_pred': [], 't1_score_0': [], 't1_score_1': []}
    
    for j in tqdm(range(len(eval_list)), leave=False):
        lut = False
        # get original image channel
        img_path = eval_list[j]
        # make image input
        img_inputs = make_image_input(img_path, lut, norm_hist, bone_hist)
        img_inputs = transforms_eval(img_inputs)
        # make ehr input
        ehr_inputs = ehr_csv[ehr_csv['img_path'] == img_path]
        img_meta = torch.from_numpy(np.array(ehr_inputs[list(ehr_inputs.columns[-32:])])).type(torch.float32) 
        # evaluate image
        outputs = model(img_inputs.unsqueeze(dim=0), img_meta)
        score = softmax(outputs)
        _, preds = torch.max(score, 1)
        #out_dict['t1_true'].extend(labels1.data.tolist())  # don't have true mace labels yet
        out_dict['t1_pred'].append(preds.item())
        for i in range(task1_cls):
            out_dict['t1_score_{}'.format(i)].append(score[:, i].item())
        # save outputs
        out_dict['dcm'].append(img_path)

    mace_out = pd.DataFrame(out_dict)
    
    return mace_out
