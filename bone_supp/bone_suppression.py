# import libraries
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms as tvtransforms
from . import GusarevModel
from tqdm import tqdm

def bone_suppress(eval_df, device='cuda:0', model_path="./bone_supp/network_intermediate_4.tar"):
    # Paths
    PATH_SAVE_NETWORK_INTERMEDIATE = model_path
    
    # Data
    _batch_size = 1
    crop_image_spatial_size = (256,256)
    interp_mode = TF.InterpolationMode.NEAREST #BILINEAR #BILINEAR#
    
    transforms = tvtransforms.Compose([
                                    tvtransforms.ToTensor(),
                                     tvtransforms.Resize(crop_image_spatial_size, interpolation=interp_mode),
                                     ])
    
    # Network
    input_array_size = (_batch_size, 1, crop_image_spatial_size[0], crop_image_spatial_size[1])
    net = GusarevModel.MultilayerCNN(input_array_size)
    #net = nn.DataParallel(net, list(range(ngpu)))
    if os.path.isfile(PATH_SAVE_NETWORK_INTERMEDIATE):
        print("=> loading checkpoint '{}'".format(PATH_SAVE_NETWORK_INTERMEDIATE))
        checkpoint = torch.load(PATH_SAVE_NETWORK_INTERMEDIATE, map_location='cpu')
        start_epoch = checkpoint['epochs_completed']
        reals_shown_now = checkpoint['reals_shown']
        net.load_state_dict(checkpoint['model_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}, reals shown {})".format(PATH_SAVE_NETWORK_INTERMEDIATE, 
                                                                            start_epoch, reals_shown_now))
    else:
        print("=> NO CHECKPOINT FOUND AT '{}'" .format(PATH_SAVE_NETWORK_INTERMEDIATE))
        raise RuntimeError("No checkpoint found at specified path.")
    
    net = net.to(device)
    # Set to testing mode
    net.eval()
    print("Loaded.")
    
    # bone suppress the images and save
    #images = [i for i in os.listdir(crop_img_base) if '_crop.png' not in i if '_supp.png' not in i and '_view_clf.png' not in i]
    images = eval_df.cropped_path.to_list()
    
    bone_supp_paths = []
    file_log = open('bone_suppression_log.txt', 'w')
    for img in tqdm(images):
        try:
            inputImage = np.array(Image.open(img))
            #print(inputImage.dtype, inputImage.min(), inputImage.max())
            inputImage = (inputImage - inputImage.min() / inputImage.max()) * 255  # original 8bit code
            inputImage = inputImage - inputImage.min() 
            inputImage = (inputImage / inputImage.max()) * 255
            inputImage = inputImage.astype('uint8')
            inputImage = transforms(inputImage)
            inputImage = torch.unsqueeze(inputImage[0], dim=0)
            inputImage = torch.unsqueeze(inputImage, dim=0)
            inputImage = inputImage.to(device)
            out = net(inputImage)
            out = out.detach()
            out = out.cpu()
            suppressed_img = Image.fromarray(np.array(out.squeeze()*255).astype('uint8'), mode='L')
            suppressed_img.save(img.replace('.png', '_supp.png'))
            bone_supp_paths.append(img.replace('.png', '_supp.png'))
        except:
            print('failed: {}'.format(img), file=file_log)
            file_log.flush()
            bone_supp_paths.append(False)
    file_log.close()
    
    # add column to the dataframe and return
    eval_df['supp_path'] = bone_supp_paths
    
    return eval_df

# python bone_suppression.py