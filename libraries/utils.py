#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:42:27 2022

@author: temuuleu
"""


import nibabel as nib
import skimage.transform as skTrans
import numpy as np
import cv2
import os
from os.path import join
from os.path import relpath
from os.path import basename

import matplotlib.pyplot as plt

from nipype import Node, Workflow
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl
#from niworkflows.interfaces.bids import DerivativesDataSink

from nipype.interfaces.fsl import SliceTimer, MCFLIRT, Smooth, BET,ApplyXFM,ConvertXFM
from nipype.interfaces.fsl import Reorient2Std

from nipype.interfaces.ants import N4BiasFieldCorrection
# from bids_library import get_main_paths
from path_utils import get_all_dirs,  plot_tree, clear, copy_file,copy_dir
from path_utils import create_result_dir,create_dir

from path_utils import  get_feature_paths
from library_bids_proscis import create_proscis_data_dict,create_bids_dir


import subprocess
import subprocess
from subprocess import run,call    
import pandas as pd
import numpy as np

from path_utils import get_all_dirs,  plot_tree, clear, copy_file,copy_dir
from path_utils import create_result_dir,create_dir
from path_utils import create_result_dir,create_dir,get_dat_paths

from os.path import join, relpath
from os import system
from bianca_library import  create_data_dictionary_for_manual, get_sessions_info,create_working_dict,create_data_dictionary_for_manual

from bianca_library import  create_master_file_for_bianca


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




def append_to_global_df(data):
    global global_dict
    if len(global_dict) == 0:
        # if global dictionary is empty, initialize it with the data
        global_dict['df'] = data
    else:
        # if global dictionary already has data, append the new data
        global_dict['df'] = pd.concat([global_dict['df'], data], ignore_index=True)

# from modules.utils import normalize_volume,write_text_on_mri_image,dice_score,copy_mask
# from modules.utils import vox2mni_mat, get_mm_per_voxel, get_mm_per_voxel
# from modules.utils import get_volume, get_voxel_res, create_normal_mri_text

def normalize_volume(image):
    """normalize 3D Volume image"""
    
    
    dim     = image.shape
    volume = image.reshape(-1, 1)
    
    rvolume = np.divide(volume - volume.min(axis=0), (volume.max(axis=0) - volume.min(axis=0)))*1
    rvolume = rvolume.reshape(dim)
    
    return rvolume


    

    
def write_text_on_mri_image(img, text="mendula",
                            thickness = 1, 
                            fontScale = 0.5,
                            color = (255, 255, 255),    
                            org = (2, 20)):
    """
    

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    text : TYPE, optional
        DESCRIPTION. The default is "mendula".
    thickness : TYPE, optional
        DESCRIPTION. The default is 1.
    fontScale : TYPE, optional
        DESCRIPTION. The default is 0.5.
    color : TYPE, optional
        DESCRIPTION. The default is (255, 255, 255).
    org : TYPE, optional
        DESCRIPTION. The default is (2, 20).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_image = np.zeros(img.shape)
    
    
    for i in range(new_image.shape[2]):

        image_normal = normalize_volume(img[:,:,i])*255
        image_normal_ascontiguousarray = np.ascontiguousarray(image_normal, dtype=np.uint8)
        

        
        image_with_text =  cv2.putText(image_normal_ascontiguousarray, text, org, font, 
                           fontScale, color, thickness, cv2.LINE_AA, True)
        


        new_image[:,:,i] = image_with_text
        
        
        # plt.imshow(image_normal_ascontiguousarray)
        # plt.show()
        
        
        # plt.imshow(new_image[:,:,i])
        # plt.show()
        
    return normalize_volume(new_image)





def dice_score(segmentation_path,mask_path):
    """

    Parameters
    ----------
    segmentation_path : Path to niftii
        path to segmentation.
    mask_path : Path to niftii
        path to ground truth mask.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    ground_truth   = nib.load(mask_path).get_fdata()
    segmentation   = nib.load(segmentation_path).get_fdata()

    #formula
    # union *2  / sum
    union =  np.sum(segmentation[ground_truth == 1])
    summe  =  (np.sum(segmentation) + np.sum(ground_truth))
    
    dice_score = (union *2.0 )/ summe
    
    print('Dice similarity score is {}'.format(dice_score))

    return dice_score

def copy_mask(source_mask,dest_mask, biascorrected_bet_image_path):
    
    standard_mask_img_d            = nib.load(source_mask)
    standard_mask_img              = standard_mask_img_d.get_fdata()
    mask_shape                     = standard_mask_img.shape
    
    flair_image_img_ref            = nib.load(biascorrected_bet_image_path)
    flair_image_img_ref            = flair_image_img_ref.get_fdata()
    ref_shape                      = flair_image_img_ref.shape
      
    standard_mask_img_resized      = skTrans.resize(standard_mask_img, ref_shape, order=1, preserve_range=True) 
    mask_img_resized = nib.Nifti1Image(standard_mask_img_resized, standard_mask_img_d.affine)
    nib.save(mask_img_resized, dest_mask)
                

    
def vox2mni_mat(flirtmat,im):
    
    mnisform = nib.load(os.getenv('FSLDIR')+"/data/standard/MNI152_T1_1mm.nii.gz").get_sform()
    vox2mm=identity(4)
    pixdims=im.header.get_zooms()
    for n in range(3):
        vox2mm[n,n]=pixdims[n]
    [qform,qcode]=im.get_qform(True)
    [sform,scode]=im.get_sform(True)
    radiological=True
    if qcode>0 and linalg.det(qform)>0: radiological=False
    if scode>0 and linalg.det(sform)>0: radiological=False
    if not radiological:   # convert to nifti voxel coordinate convention
        swapmat=identity(4)
        swapmat[0,0]=-1
        swapmat[0,3]=im.shape[0]-1
        vox2mm=dot(vox2mm,swapmat)
    flirtmat=dot(flirtmat,vox2mm)
    retmat=dot(mnisform,flirtmat)
    return retmat

def get_mm_per_voxel(path):

    img_d                      = nib.load(path)

    x = img_d.header["pixdim"][1]
    y = img_d.header["pixdim"][2]
    z = img_d.header["pixdim"][3]
    
    return (x,y,z)
    
def create_normal_mri_text(path,shape =(300,300,30) , text = ""):
    
    img_d                      = nib.load(path)
    img                        = img_d.get_fdata()   
    img_trans                  = skTrans.resize(normalize_volume(img),
                                                       shape, 
                                                       order=1, preserve_range=True)
    
    img_trans_text             = rotate(write_text_on_mri_image(img_trans,
                                                                text),90)
    
    return img_trans_text


def get_volume(path):
    
    output_all_matter = run(["fslstats",
                    path,
                    "-V"], 
                    capture_output=True)
            
    result_str = str(output_all_matter.stdout)
    all_matter_result = result_str.split(" ")

    return float(all_matter_result[1])


def get_voxel_res(path, dim = 2):

    img_d                                   = nib.load(path)
    img                                     = img_d.get_fdata()   
    
    if dim == 2:
        resolution =  img.shape[0] * img.shape[1] 
    
    if dim == 3:
        resolution =  img.shape[0] * img.shape[1] * img.shape[2]
    
    print(f"resolution {resolution}")

    return resolution
