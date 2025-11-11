# !/usr/bin/env python
# coding: utf-8

import json
import torch
import pickle
import random
import pandas as pd
from torchvision import transforms
from options.baseOptions import BaseOptions
from base import simpleModels, simpleDataloader, parallelClassification, simpleTransforms

# python parallelTrain.py

# load and modify the test options
base_json = json.load(open('parallel_options.json', "r"))
json.dump(base_json, open('test.json', 'w'))

opt = BaseOptions(json_filepath='test.json').parse()

# set random seed - for reproducibility
random_seed = 1991
random.seed(random_seed)
torch.manual_seed(random_seed)
# initialization

torch.set_num_threads(1)

# specifically for heart calcification:
opt.df_labels = {'crop_file': 'img_path', 'adverse_event': 'img_label_1', 'Clinical_bins': 'img_label_2', 'AccessionNumber': 'img_id'}
opt.df_filter = ('view', 0) # only using AP views (0) or Lateral (1)

# initialize model
model_ft, opt.params_to_update, opt.input_size, opt.is_inception = simpleModels.initialize_model(opt)
print(model_ft)

# make preprocessing and transforms
transforms_train = transforms.Compose([
    # add transforms from: simpleTransforms
    simpleTransforms.makeRGB(),
    #simpleTransforms.convertLUT(),  # convert images to LUT
    #transforms.Pad(50),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((opt.input_size, opt.input_size)),
    #transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
transforms_val = transforms.Compose([
    simpleTransforms.makeRGB(),
    #simpleTransforms.convertLUT(),
    transforms.Resize((opt.input_size, opt.input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# make dataloaders
train_loader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_train, transforms_train)
print('training size: {}'.format(len(train_loader)), file=opt.log)
val_loader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_val, transforms_val, shuffle=False)
print('validation size: {}'.format(len(val_loader)), file=opt.log)
dataloaders = {'train': train_loader, 'val': val_loader}

# make data parallel if multi-gpu
if len(opt.gpu_ids) > 1:
    model_ft = torch.nn.DataParallel(model_ft, device_ids=opt.gpu_ids)

# load the model, optimizer, and loss functions
model_ft, optimizer_ft, criterion1, criterion2 = simpleModels.load_model(opt, model_ft)

# train and evaluate
model_ft, histories, best_states = parallelClassification.train_epochs(opt, model_ft, dataloaders, criterion1, criterion2, optimizer_ft)

# save the final versions
torch.save(model_ft, opt.save_dir + '{}.pth'.format(opt.model_name))
pickle.dump(histories, open(opt.save_dir + "{}_history.pkl".format(opt.model_name), "wb"))

# option to add the train and val predictions outputs as csvs
if opt.output_csv:
    # default transform
    transforms_eval = transforms.Compose([
        simpleTransforms.makeRGB(),
        transforms.Resize((opt.input_size, opt.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # load the model, optimizer, and loss functions
    model_ft, optimizer_ft, criterion1, criterion2 = simpleModels.load_model(opt, model_ft)

    for test_mode in ['train', 'val']:
        dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_val if test_mode == 'val' else opt.dataframe_train, transforms_eval, shuffle=False)
        pred_dict = parallelClassification.model_predict(opt, model_ft, dataloader)
        mode_df = pd.DataFrame(pred_dict)
        mode_df.to_csv(opt.save_dir + '{}_{}.csv'.format(opt.model_name, test_mode))
    for test_mode in ['test', 'ext']:
        if test_mode == 'test' and opt.dataframe_test is not None:
            dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_test, transforms_eval, shuffle=False)
        elif test_mode == 'ext' and opt.dataframe_ext is not None:
            dataloader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_ext, transforms_eval, shuffle=False)
        else:
            print('{} does not exist'.format('test dataframe' if test_mode == 'test' else 'external dataframe'))
            break
        pred_dict = parallelClassification.model_predict(opt, model_ft, dataloader)
        mode_df = pd.DataFrame(pred_dict)
        mode_df.to_csv(opt.save_dir + '{}_{}.csv'.format(opt.model_name, test_mode))