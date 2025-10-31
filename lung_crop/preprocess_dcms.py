# code to preprocess all of the heartCalcification data (segmenting the lungs)

import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from skimage.measure import label, regionprops
from tqdm import tqdm


def check_LUT(img):
    img_array = img
    # check different corners
    x_corner = int(img_array.shape[0] * 0.1)
    y_corner = int(img_array.shape[1] * 0.1)
    tl_corner = np.ravel(img_array[:x_corner, :y_corner])  # top left
    tr_corner = np.ravel(img_array[:x_corner, -y_corner:])  # top right
    bl_corner = np.ravel(img_array[-x_corner:, :y_corner])  # bottom left
    br_corner = np.ravel(img_array[-x_corner:, -y_corner:])  # bottom right
    # concat all corners
    corners = np.concatenate([tl_corner, tr_corner, bl_corner, br_corner])
    lut = False if ((251 < corners).sum() < (corners < 5).sum()) else True
    
    #if lut:
    #    print('found LUT, converting image')
    
    return np.invert(img) if lut else img

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

def segment_samples(model, dcm_path, save_dir, X_shape=510, i=0, save=True, aspect_wh=False, verbose=True):
    if verbose:
        print('loading {}'.format(dcm_path))
    original_img = pydicom.dcmread(dcm_path).pixel_array
    # convert to 8 bit for original image save
    original_img = (original_img / original_img.max()) * 255
    original_img = original_img.astype('uint8')
    original_img = np.stack((original_img, original_img, original_img), axis=-1)
    original_img = check_LUT(original_img)
    if verbose:
        print(original_img.shape)
    # load original image
    im_array = []
    # getting the scaling factor for remapping the images
    height, width, channels = original_img.shape
    h_scale = height/X_shape
    w_scale = width/X_shape
    im = cv2.resize(original_img,(X_shape,X_shape))[:,:,0]
    im = (im-127.0)/127.0
    im_array.append(im)
    # convert into format for model
    y_test = np.array(im_array).reshape(len(im_array),X_shape,X_shape,1)
    # get segmentation
    if verbose:
        print('getting segmentations')
    preds = model.predict(y_test,verbose=0)
    if verbose:
        print('preds.shape', preds.shape)
    # get the bounding box
    lbl_0 = label(np.squeeze(preds)) 
    props = regionprops(lbl_0)
    if verbose:
        print('props', props)
    if props == []:
        print('no lung segmentation')
        return
    img_1 = im.copy()
    x_start, y_start, x_end, y_end = [], [], [], []
    for prop in props:
        x_start.append(prop.bbox[1])
        y_start.append(prop.bbox[0])
        x_end.append(prop.bbox[3])
        y_end.append(prop.bbox[2])
        cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
    if verbose:
        print('cropping image')
        print('original_img', original_img.shape)
        print('x_start, y_start, x_end, y_end', x_start, y_start, x_end, y_end)
    if aspect_wh:
        lung_crop = original_img[int(min(y_start)*w_scale):int(max(y_end)*w_scale), int(min(x_start)*h_scale):int(max(x_end)*h_scale)]
    else:
        lung_crop = original_img[int(min(y_start)*h_scale):int(max(y_end)*h_scale), int(min(x_start)*w_scale):int(max(x_end)*w_scale)]
    lung_crop = cv2.resize(lung_crop,(X_shape,X_shape))
    if verbose:
        print('cropping segmentation')
    preds = np.squeeze(preds)*255
    preds_crop = preds[min(y_start):max(y_end), min(x_start):max(x_end)]
    if verbose:
        print('resizing segmentation')
    preds_crop = cv2.resize(preds_crop, (lung_crop.shape[1], lung_crop.shape[0]))
    if verbose:
        print('saving image')
        print(lung_crop.dtype)
    if save:
        cropped_img_path = os.path.join(save_dir, '{}.png'.format(i))
        cv2.imwrite(cropped_img_path, lung_crop)
        inverted_lung_seg_path = os.path.join(save_dir, '{}_crop.png'.format(i))
        cv2.imwrite(inverted_lung_seg_path, preds_crop)
        return cropped_img_path, inverted_lung_seg_path
    else:
        return original_img, y_test, lung_crop, preds_crop, preds, props

# python preprocess_lung_crop_from_DICOM.py
def process_dicom(dcm_df, save_dir, model_dir='./lung_crop/cxr_reg_weights.best.hdf5', device='cuda:3'):
    # load the trained model weights
    model = unet(input_size=(512,512,1))
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
    model.load_weights(model_dir)
    
    # initialize processed paths
    crop_paths = []
    inverted_paths = []
    file_log = open('dcm_lung_crop_log.txt', 'w')
    for i in tqdm(range(len(dcm_df))):
        try:
            cropped_img_path, inverted_lung_seg_path = segment_samples(model, dcm_df.path.iloc[i], save_dir, X_shape=512, i=i, verbose=False)
            crop_paths.append(cropped_img_path)
            inverted_paths.append(inverted_lung_seg_path)
        except:
            print(dcm_df.path.iloc[i], file=file_log)
            file_log.flush()
            crop_paths.append(False)
            inverted_paths.append(False)
        i += 1
    file_log.close()
    
    # add column to the dataframe and return
    dcm_df['cropped_path'] = crop_paths
    dcm_df['heart_path'] = inverted_paths
    
    return dcm_df

# function to doublecheck proper cropping
def check_lung_cropping(dcm_df, i, save_dir=None, model_dir='./lung_crop/cxr_reg_weights.best.hdf5', device='cuda:3'):
    # load the trained model weights
    model = unet(input_size=(512,512,1))
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
    model.load_weights(model_dir)
    
    original_img, y_test, lung_crop, preds_crop = segment_samples(model, dcm_df.path.iloc[i], save_dir, X_shape=512, i=i, save=False, verbose=False)
    
    return original_img, y_test, lung_crop, preds_crop