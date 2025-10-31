import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    # preprocessing arguments
    parser.add_argument("--processed_dir", default='./processed_fromDICOM/', type=str, help="default directory to save the processed images to or use the processed images from")
    parser.add_argument("--process", type=int, default=0, help="process the images or not (0=False, 1=True)")
    # training arguments
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train the model")
    parser.add_argument("--checkpoints_dir", type=str, default='./trained_model/', help="directory to save the model")
    # inference arguments
    parser.add_argument("--cac", type=int, default=0, help="predict cac or not (0=False, 1=True)")
    parser.add_argument("--joint", type=int, default=0, help="use the joint model or not (0=False, 1=True)")
    # general arguments
    parser.add_argument("--df_path", type=str, default=None, help="root directory of the dicom files to process or dataframe that is preprocessed")
    parser.add_argument("--trained_model", type=str, default='./mace_clf/MACE_clf.pth', help="path to the trained model to use")
    parser.add_argument("--device", type=str, default='cuda:3', help="cuda to use")
    parser.add_argument("--verbose", type=int, default=1, help="print process steps (0=False, 1=True)")
    args = parser.parse_args()
    
    # parse args and return it
    config = parser.parse_args()
    return config