import os
import glob
import argparse
import pandas as pd
from config import parse_args

from joint_clf.joint_clf import *
from view_clf.inference import *
from clahe_clf.inference import *
from lung_crop.preprocess_dcms import *
from bone_supp.bone_suppression import *

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
    eval_df = eval_df[eval_df['CLAHE_pred'] == 2]
    eval_df = eval_df.reset_index(drop=True)
    print('eval_df:', len(eval_df)) if args.verbose == 1 else None
    
    # extract images from dcms and lung crop
    print('running lung cropping') if args.verbose == 1 else None
    eval_df = process_dicom(eval_df, args.processed_dir, device=args.device)
    
    # run bone suppression
    print('running bone suppression')
    eval_df = bone_suppress(eval_df, device=args.device)
    
    # save the processed dataframe
    print('finished pre-processing') if args.verbose == 1 else None
    eval_df.to_csv('processed_df.csv')
    
if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    print('preprocessing') if args.verbose == 1 else None
    preprocess(args)