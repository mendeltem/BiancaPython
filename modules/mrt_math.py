#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:17:27 2022

@author: temuuleu
"""


import os
from os.path import join
from os.path import relpath
from os.path import basename

import matplotlib.pyplot as plt

#from nipype import Node, Workflow
#from nipype.pipeline import engine as pe
#from nipype.interfaces import utility as niu, fsl
#from niworkflows.interfaces.bids import DerivativesDataSink

#from nipype.interfaces.fsl import SliceTimer, MCFLIRT, Smooth, BET,ApplyXFM,ConvertXFM
#from nipype.interfaces.fsl import Reorient2Std
#from nipype.interfaces.ants import N4BiasFieldCorrection
#from modules.bids_library import get_main_paths
#from modules.path_utils import get_all_dirs,  plot_tree, clear, copy_file,copy_dir
#from modules.path_utils import create_result_dir,create_dir

#from modules.path_utils import  get_feature_paths
#from modules.library_bids_proscis import create_proscis_data_dict,create_bids_dir
# from modules.utils import  resize_niftii_image,cosize_images
# from modules.utils import normalize_volume,write_text_on_mri_image
import subprocess
import subprocess
from subprocess import run,call    
import pandas as pd
import numpy as np

#from modules.path_utils import get_all_dirs,  plot_tree, clear, copy_file,copy_dir
#from modules.path_utils import create_result_dir,create_dir
#from modules.path_utils import create_result_dir,create_dir,get_dat_paths
#from modules.bids_library import get_main_paths
from os.path import join, relpath
from os import system
#from modules.bianca_library import  create_master_file_for_bianca, get_sessions_info,create_working_dict,create_data_dictionary_for_manual

import skimage.transform as skTrans
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
from numpy import flip
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import time
import shutil
from nibabel import load,save
from numpy import *
import cv2 
import configparser
#!pip install nibabel

import tensorflow as tf

def dice_coef_tf(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_4d_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels # taking average

def dice_3d_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,index], y_pred[:,:,index])
    return dice/numLabels # taking average


session_dir           = "/home/temuuleu/CSB_NeuroRad/temuuleu/Projekts/BELOVE/Bianca_Project/bianca/data/prepared/belove_2/sub-22/ses-01"
bianca_result         = "seg_prob_bianca.nii.gz"
manual_mask_path      = join(session_dir,"WMHmask.nii.gz")
bianca_result_path    = join(session_dir,bianca_result)

#load manual mask as numpy array
manual_mask_image     = nib.load(manual_mask_path).get_fdata()
#load bianca result as numpy array
bianca_result_image   = nib.load(bianca_result_path).get_fdata()

#thresholding
th                            = 0.95
output_mask                   = np.copy(bianca_result_image)
output_mask[output_mask >=th] = 1
output_mask[output_mask <th]  = 0
segmentation                  = output_mask
y_pred                        = segmentation
y_true                        = manual_mask_image


dice_list = {}
image_count = segmentation.shape[2] -1

for i in range(image_count):

    if i == 0: continue

    #for index in range(i):
    dice_score = dice_coef(y_true[:,:,i], y_pred[:,:,i])
    dice_score  = round(dice_score,4)
    
    y_true_f   = y_true[:,:,i].flatten()
    y_pred_f   = y_pred[:,:,i].flatten()
    
    fp = sum((y_true_f == 1) & (y_pred_f == 0))

    #dice_score = round(dice/i,4) # taking average
    dice_list[i] = round(dice_score,4)
    
    #dice_coef = dice_3d_coef_multilabel(manual_mask_image,segmentation,i+1)
    print(f"slice {i}  dice_score      {dice_score}")
    #print(f"slice {i}  false positives {fp}")
print("")
    
    
dice_list_sorted = {k: v for k, v in sorted(dice_list.items(), key=lambda item: item[1])}
last_key_index = list(dice_list_sorted.keys())[-1]

image_count- last_key_index


print(f"last index {last_key_index}")

for i in range(last_key_index) : 
    plt.imshow(segmentation[:,:,i])
    plt.imshow(mask_image[:,:,i])
    plt.show()




# os.listdir(session_dir)
