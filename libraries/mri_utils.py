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
from subprocess import run,call   
import os
import nibabel as nib


# def copy_mask(mask_name,flair_image_path,prep_subject_dir):

    
#     if mask_name.endswith(".nii.gz"):
#         standard_mask_name                    =  "WMHmask.nii.gz" 	#brain extracted
#     elif mask_name.endswith(".nii"):
#         standard_mask_name                    =  "WMHmask.nii" 	#brain extracted
    
    
#     standard_mask_path   =  os.path.join(prep_subject_dir,standard_mask_name)
    
#     os.system(f"cp {mask_path} {standard_mask_path} ")
    
#     standard_mask_img_nib          = nib.load(mask_path)
#     standard_mask_img              = standard_mask_img_nib.get_fdata()
#     mask_shape                     = standard_mask_img.shape
    
#     flair_image_img_nib            = nib.load(flair_image_path)
#     flair_image_img_ref            = flair_image_img_nib.get_fdata()
#     ref_shape                      = flair_image_img_ref.shape
    
#     if mask_shape == ref_shape:
#         print("same")
#         standard_mask_img_resized      = standard_mask_img
#     else:
#         print("not same")
#         standard_mask_img_resized      = skTrans.resize(standard_mask_img, ref_shape, order=1, preserve_range=True) 
        
#     mask_img_resized = nib.Nifti1Image(standard_mask_img_resized, standard_mask_img_nib.affine)
#     nib.save(mask_img_resized, standard_mask_path)
    
    
#     return standard_mask_path
def normalization_image(biascorrected_bet_image_path):
    
    biascorrected_nib =  nib.load(biascorrected_bet_image_path)    

    biascorrected_bet_image_normlized =  normalize_volume(biascorrected_nib.get_fdata()  )            
    biascorrected_normlized_nib = nib.Nifti1Image(biascorrected_bet_image_normlized, biascorrected_nib.affine)
    
    nib.save(biascorrected_normlized_nib, biascorrected_bet_image_path)
    
    return biascorrected_bet_image_path



def get_volume(path):
    
    output_all_matter = run(["fslstats",
                    path,
                    "-V"], 
                    capture_output=True)
            
    result_str = str(output_all_matter.stdout)
    all_matter_result = result_str.split(" ")

    return float(all_matter_result[1])


def normalize_volume(image):
    """normalize 3D Volume image"""
    
    
    dim     = image.shape
    volume = image.reshape(-1, 1)
    
    rvolume = np.divide(volume - volume.min(axis=0), (volume.max(axis=0) - volume.min(axis=0)))*1
    rvolume = rvolume.reshape(dim)
    
    return rvolume


def resize_niftii_image(flair_biascorrected_image_path, dim):
    """
    

    Parameters
    ----------
    flair_biascorrected_image_path : TYPE
        DESCRIPTION.
    dim : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """


    flair_image_img_d            = nib.load(flair_biascorrected_image_path)
    flair_image_img_original     = flair_image_img_d.get_fdata()
        
    flair_image_img_resized      = skTrans.resize(flair_image_img_original, dim, order=1, preserve_range=True) 
    flair_img = nib.Nifti1Image(flair_image_img_resized, flair_image_img_d.affine)
    nib.save(flair_img, flair_biascorrected_image_path)

    
            
def cosize_images(standard_mask, flair_biascorrected_bet_image_path ):
    """
    

    Parameters
    ----------
    standard_mask : TYPE
        DESCRIPTION.
    flair_biascorrected_bet_image_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    standard_mask_img_d            = nib.load(standard_mask)
    standard_mask_img              = standard_mask_img_d.get_fdata()
    mask_shape                     = standard_mask_img.shape
    
    flair_image_img_ref            = nib.load(flair_biascorrected_bet_image_path)
    flair_image_img_ref            = flair_image_img_ref.get_fdata()
    ref_shape                      = flair_image_img_ref.shape
      
    standard_mask_img_resized      = skTrans.resize(standard_mask_img, ref_shape, order=1, preserve_range=True) 
    mask_img_resized = nib.Nifti1Image(standard_mask_img_resized, standard_mask_img_d.affine)
    nib.save(mask_img_resized, standard_mask)
                
    
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
        

        new_image[:,:,i] = cv2.putText(image_normal_ascontiguousarray, text, org, font, 
                           fontScale, color, thickness, cv2.LINE_AA, True)
        
        
        # plt.imshow(image_normal_ascontiguousarray)
        # plt.show()
        
        
        # plt.imshow(new_image[:,:,i])
        # plt.show()
        
    return normalize_volume(new_image)