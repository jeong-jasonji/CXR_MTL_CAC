import os
import pandas as pd
from config import parse_args

from mace_clf.CAC_MACE_clf import *
from joint_clf.joint_clf import *
from view_clf.inference import *
from clahe_clf.inference import *
from lung_crop.preprocess_dcms import *
from bone_supp.bone_suppression import *


def preprocess(args):
    print('processing images') if args.verbose == 1 else None
    # make directory for saving processed images
    os.makedirs(args.processed_dir, exist_ok=True)
    
    # read dataframe for view classification
    eval_df = pd.read_csv(args.df_path)  # must have columns 'path' and optionally 'label'
    
    # run CXR view classification - 0=AP, 1=lateral
    print('running CXR view classifier') if args.verbose == 1 else None
    eval_df = view_clf_inference(eval_df, device=args.device)
    
    # run CXR CLAHE classification - 0=CLAHE, 1=inverted, 2=normal
    print('running CLAHE classifier') if args.verbose == 1 else None
    eval_df = CLAHE_clf_inference(eval_df, device=args.device)
    
    # save eval_df to check manually if desired
    eval_df.to_csv('eval_df.csv') if args.save_df == 1 else None
    
    # use only AP images
    eval_df = eval_df[eval_df['view_pred'] == 0]
    # use only non-CLAHE and non-inverted images
    eval_df = eval_df[eval_df['CLAHE'] == 2]
    eval_df = eval_df.reset_index(drop=True)
    print('eval_df:', len(eval_df)) if args.verbose == 1 else None
    
    # extract images from dcms and lung crop
    print('running lung cropping') if args.verbose == 1 else None
    process_dicom(eval_df, args.processed_dir, device=args.device)
    
    # run bone suppression
    print('running bone suppression')
    bone_suppress(crop_img_base=args.processed_dir, device=args.device)
    print('finished pre-processing') if args.verbose == 1 else None

def main(args):
    # pre-processing
    if args.process == 1:
        preprocess(args)
        args.df_path = '/'.join(args.df_path.split('/')[:-1] + ['processed_df.csv'])
    
    # run MACE classification
    print('running MACE classification') if args.verbose == 1 else None
    eval_df = eval_MACE_joint(args) if args.joint == 1 else eval_MACE(args)
    eval_df.to_csv('cac_inference.csv')
    
    
if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    main(args)