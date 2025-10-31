import os
import glob
import torch
import argparse
import pandas as pd
from config import parse_args

from mace_clf.CAC_MACE_clf import *
from joint_clf.joint_clf import *
from view_clf.inference import *
from clahe_clf.inference import *
from lung_crop.preprocess_dcms import *
from bone_supp.bone_suppression import *
# training libraries
from options.baseOptions import BaseOptions
from base import simpleModels, simpleDataloader, simpleClassification, simpleTransforms

def preprocess(args):
    print('processing images') if args.verbose == 1 else None
    # make directory for saving processed images
    os.makedirs(args.processed_dir, exist_ok=True)
    
    # make dataframe for view classification
    # must have columns 'path' (for dicom paths) and 'label' (for the MACE label)
    eval_df = pd.read_csv(args.df_path)
    
    # run CXR view classification - 0=AP, 1=lateral
    print('running CXR view classifier') if args.verbose == 1 else None
    eval_df = view_clf_inference(eval_df, device=args.device)
    
    # run CXR CLAHE classification - 0=CLAHE, 1=inverted, 2=normal
    print('running CLAHE classifier') if args.verbose == 1 else None
    eval_df = CLAHE_clf_inference(eval_df, device=args.device)
    
    # use only AP images
    eval_df = eval_df[eval_df['view_pred'] == 0]
    # use only non-CLAHE and non-inverted images
    eval_df = eval_df[eval_df['CLAHE'] == 2]
    eval_df = eval_df.reset_index(drop=True)
    print('eval_df:', len(eval_df)) if args.verbose == 1 else None
    
    # extract images from dcms and lung crop
    print('running lung cropping') if args.verbose == 1 else None
    eval_df = process_dicom(eval_df, args.processed_dir, device=args.device)
    
    # run bone suppression
    print('running bone suppression')
    eval_df = bone_suppress(eval_df, args.processed_dir, device=args.device)
    
    # save the processed dataframe
    print('finished pre-processing') if args.verbose == 1 else None
    eval_df.to_csv('processed_df.csv')

def train(args):
    # initialize options for training
    print('getting training config') if args.verbose == 1 else None
    opt = BaseOptions(json_filepath='train.json', args=args).parse()
    
    # initialize model
    print('initializing model') if args.verbose == 1 else None
    model_ft, opt.params_to_update, opt.input_size, opt.is_inception = simpleModels.initialize_model(opt)
    
    # make preprocessing and transforms
    transforms_train = transforms.Compose([
        simpleTransforms.convertLUT(),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((opt.input_size, opt.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transforms_val = transforms.Compose([
        simpleTransforms.convertLUT(),
        transforms.Resize((opt.input_size, opt.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # make dataloaders
    print('making dataloaders') if args.verbose == 1 else None
    train_loader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_train, transforms=transforms_train)
    print('training size: {}'.format(len(train_loader)), file=opt.log)
    val_loader = simpleDataloader.simpleDataLoader(opt, opt.dataframe_val, transforms=transforms_val, shuffle=False)
    print('validation size: {}'.format(len(val_loader)), file=opt.log)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # load the model, optimizer, and loss functions
    print('initializing losses and optimizers') if args.verbose == 1 else None
    model_ft, optimizer_ft, criterion = simpleModels.load_model(opt, model_ft)
    
    # train and evaluate
    print('starting training') if args.verbose == 1 else None
    model_ft, histories, best_states = simpleClassification.train_epochs(opt, model_ft, dataloaders, criterion, optimizer_ft)
    
    # save the best model
    print('saving the best model') if args.verbose == 1 else None
    torch.save(model_ft, opt.save_dir + '{}_best.pth'.format(opt.model_name))

def main(args):
    # pre-processing
    if args.process == 1:
        print('preprocessing') if args.verbose == 1 else None
        preprocess(args)
        args.df_path = '/'.join(args.df_path.split('/')[:-1] + ['processed_df.csv'])
    
    # training
    print('running training') if args.verbose == 1 else None
    train(args)
    
if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    main(args)