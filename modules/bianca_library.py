#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:43:43 2022

@author: temuuleu
"""

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


import subprocess
from subprocess import run,call    
import pandas as pd
import numpy as np

from modules.path_utils import get_all_dirs,  plot_tree, clear, copy_file,copy_dir
from modules.path_utils import create_dir
from modules.path_utils import create_result_dir,create_dir,get_dat_paths
from modules.bids_library import get_main_paths
from os.path import join, relpath
from os import system

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



from modules.path_utils import get_all_dirs,  plot_tree, clear, copy_file,copy_dir
from modules.path_utils import create_result_dir,create_dir,get_dat_paths
from modules.bids_library import get_main_paths
from modules.path_utils import  get_feature_paths
from os.path import join, relpath
import os
import re
import nibabel as nib
from nipype import Node, Workflow
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl
from nipype.interfaces.fsl import SliceTimer, MCFLIRT, Smooth, BET,ApplyXFM,ConvertXFM
from nipype.interfaces.fsl import Reorient2Std
from nipype.interfaces.ants import N4BiasFieldCorrection
import shutil

def copy_mask_to_workingdir(flair,standard_mask,prep_session_dir,standard_mask_image_name,biascorrected_bet_image_path):
    
    try:
        standard_mask_image_path          =  flair["mask"]
        
        has_roi                           = 1
    except:
        standard_mask_image_path          =  standard_mask
        has_roi                           = 0
        
    mask_path                             =  join(prep_session_dir,standard_mask_image_name)
    
    """copy mask to working directory"""
    copy_mask(standard_mask_image_path,mask_path, biascorrected_bet_image_path)
    
    
    return has_roi


def exctrac_bmatters(biascorrected_bet_image_path,prep_session_dir):

    flair_name = os.path.basename(biascorrected_bet_image_path).split(".")[0]
    
    #with mni matrix
    output_segmentation                   = run(["fast","-t","2",
                                                biascorrected_bet_image_path], 
                                                capture_output=True)
    
    pve_0                                 =  f"{flair_name}_pve_0.nii.gz" 	#wm
    pve_0_path                            =  join(prep_session_dir,pve_0)
    
    filtered_wm                           = excract_brain_matter(pve_0_path)
    
    #volume exctract  grey matter
    pve_1                                 =  f"{flair_name}_pve_1.nii.gz" 	#gm
    pve_1_path                            =  join(prep_session_dir,pve_1)
    filtered_gm                           =  excract_brain_matter(pve_1_path)
    
    return filtered_wm,filtered_gm, pve_0_path, pve_1_path

    
def preprocess(flair_image_path,prep_session_dir):
    
    paths_dict= create_paths(prep_session_dir)
    

    #copy_file(flair_image_path, base_image_path, False)
    wf                                    = Workflow(name="prep_flair")
    reorient                              = Node(Reorient2Std(), name="flair")
    reorient.inputs.in_file               = flair_image_path
    reorient.inputs.out_file              = paths_dict["reoriented_image_path"]
    reorient.run() 
                      
    biascorr                              = Node(N4BiasFieldCorrection(save_bias=False, 
                                                                       num_threads=-1),
                                                                       name="biascorr")
    
    biascorr.inputs.input_image           = paths_dict["reoriented_image_path"]
    biascorr.inputs.output_image          = paths_dict["biascorrected_image_path"]
    biascorr.run()

    skullstrip                            = Node(BET(mask=False), 
                                                 name="skullstrip")
    skullstrip.inputs.in_file             = paths_dict["biascorrected_image_path"]
    skullstrip.inputs.out_file            = paths_dict["biascorrected_bet_image_path"]
    skullstrip.run()
    
    return paths_dict["biascorrected_bet_image_path"], paths_dict

def create_paths(prep_session_dir):
    
    paths_dict = {}

    MNI_flair_xfm                         =  f"flair_to_MNI.mat" 	        #transformation matrix from template to MNI
    base_name                             =  f"flair.nii.gz" 	#brain extracted
    reoriented_image_name                 =  f"flair_reoriented.nii.gz" 	#reoriented_corrected
    biascorrected_image_name              =  f"flair_bias_corrected.nii.gz" 	#bias_corrected
    biascorrected_bet_image_name          =  f"flair_bet.nii.gz" 	#brain extracted
    biascorrected_bet_normalized          =  f"flair_normalized.nii.gz" 	#brain extracted
    
    MNI_xfm                               =  join(prep_session_dir,MNI_flair_xfm)
    base_image_path                       =  join(prep_session_dir,base_name)
    reoriented_image_path                 =  join(prep_session_dir,reoriented_image_name)
    biascorrected_image_path              =  join(prep_session_dir,biascorrected_image_name)
    biascorrected_bet_image_path          =  join(prep_session_dir,biascorrected_bet_image_name)
    biascorrected_normalized_image_path   =  join(prep_session_dir,biascorrected_bet_normalized)
    
    standard_mask_image_name_path         =  join(prep_session_dir,"WMHmask.nii.gz" )
    
    
    paths_dict["MNI_xfm"]                             = MNI_xfm
    paths_dict["base_image_path"]                     = base_image_path
    paths_dict["reoriented_image_path"]               = reoriented_image_path
    paths_dict["biascorrected_image_path"]            = biascorrected_image_path
    paths_dict["biascorrected_bet_image_path"]        = biascorrected_bet_image_path
    paths_dict["biascorrected_normalized_image_path"] = biascorrected_normalized_image_path
    paths_dict["standard_mask_image_name_path"]       =  standard_mask_image_name_path
    
    paths_dict["standard_mask_image_name"]             =  "WMHmask.nii.gz" 
    
    return paths_dict

def write_text_on_mri_2dimage(img, text="",    thickness = 2,  
                              fontScale = 0.5,    org = (10, 10) , flip = 0):
    
    #img = tp
    
    color = (255, 255, 255)   
    
    font  = cv2.FONT_HERSHEY_SIMPLEX

    #black image
    new_image = np.zeros(img.shape)
    #brain image
    image_normal = normalize_volume(img[:,:])*255
    image_normal_ascontiguousarray = np.ascontiguousarray(image_normal,
                                                          dtype=np.uint8)
    
    
    
    plt.imshow(image_normal_ascontiguousarray)
    
    
    
    black_image_with_text =  cv2.putText(new_image, str(text), org, font, 
                       fontScale, color, thickness, cv2.LINE_AA, True)
    
    
    plt.imshow(black_image_with_text)
    
    #flip the imaage
    #black_image_with_text = cv2.flip(black_image_with_text, flip)
    
    image_text= np.logical_or(image_normal_ascontiguousarray, black_image_with_text)
    
    black_image_with_text = 0
    
    
    plt.imshow(image_text)
    
    return image_text
    


def create_confusion_matrix(segmentation,ground_truth_bin ,org = (0,0), flip = 0):
        
    tp_3d = np.zeros(segmentation.shape)
    fp_3d = np.zeros(segmentation.shape)
    fn_3d = np.zeros(segmentation.shape)
    
    channels  = segmentation.shape[2]
    
    dice_score_list = {}
    
    # plt.imshow(tp_text)
    # plt.imshow(tp)
    
    for channel in range(channels):
        
        #channel=17
        
        segmentation_image = segmentation[:,:,channel]
        ground_truth_image = ground_truth_bin[:,:,channel]
        
        tp = ((segmentation_image == 1) & (ground_truth_image == 1)).astype(np.float64) 
        tp_text = write_text_on_mri_2dimage(tp, text = str(sum(tp)),org = org, thickness=0 , flip = flip)
        tp_3d[:,:,channel] = tp_text
        
        fp = ((segmentation_image == 1) & (ground_truth_image == 0)).astype(np.float64) 
        fp_text = write_text_on_mri_2dimage(fp, text = str(sum(fp)),org = org, thickness=0 , flip = flip)
        fp_3d[:,:,channel] = fp_text
        
        fn = ((segmentation_image == 0) & (ground_truth_image == 1)).astype(np.float64) 
        fn_text = write_text_on_mri_2dimage(fn, text = str(sum(fn)),org = org, thickness=0 , flip = flip)
        fn_3d[:,:,channel] = fn_text
        
        dice_score     = round((2*sum(tp)) / (2*sum(tp)+sum(fp)+sum(fn)),2) 
        sum_image          = np.logical_or(segmentation_image,ground_truth_image )
        
        dice_score_list[channel] = dice_score

    return tp_3d, fp_3d, fn_3d ,dice_score_list



def print_run_bianca(project_name, prepared_dir,workin_dict):
    #print
    print("starting bianca pipeline")
    print(f"directory name       : {project_name}")
    print(f"Result directory     : {prepared_dir}")
    print(f"Subject Count        : {len(workin_dict)}")
    print("")


def create_new_result_directory(prepared_dir,temp_dataset_directory,dataset_directory ):

    temp_prepared_dir         = create_result_dir(join(temp_dataset_directory,
                                        relpath(prepared_dir,
                                                dataset_directory))+"/","temp")

    if os.path.exists(prepared_dir):
        yes = 1
        
        while(yes):
            try:
                create_dir(prepared_dir) 
                shutil.rmtree(prepared_dir)
                create_dir(prepared_dir) 
                yes = 0
            except:
                print("error")
                
    return temp_prepared_dir
            

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

    
def cosize_images(mask_path, flair_biascorrected_bet_image_path ):
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

    standard_mask_img_d            = nib.load(mask_path)
    standard_mask_img              = standard_mask_img_d.get_fdata()
    mask_shape                     = standard_mask_img.shape
    
    flair_image_img_ref            = nib.load(flair_biascorrected_bet_image_path)
    flair_image_img_ref            = flair_image_img_ref.get_fdata()
    ref_shape                      = flair_image_img_ref.shape
      
    standard_mask_img_resized      = skTrans.resize(standard_mask_img, ref_shape, order=1, preserve_range=True) 
    mask_img_resized = nib.Nifti1Image(standard_mask_img_resized, standard_mask_img_d.affine)
    nib.save(mask_img_resized, mask_path)
                
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



def get_dice_score(segmentation_path,mask_path):
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

def make_threshholde(output_mask_original , th):
    #th_str = str(th).split(".")[1]
    output_mask = np.copy(output_mask_original)
    output_mask[output_mask >=th] = 1
    output_mask[output_mask <th] = 0
    return output_mask


def excract_brain_matter(pve_path):
    
    output_wm       = run(["fslstats",
                 pve_path,
                "-V"], 
                  capture_output=True)
    
    result_str       = str(output_wm.stdout)
    wm_result        = result_str.split(" ")
    filtered_wm      = float(wm_result[1])
    
    return filtered_wm


def delete_brain_parts(biascorrected_bet_image_path,
                       atlas_path,
                       prep_session_dir,
                       cerebellum_labels = [15,8 ,38, 34,14,37,11,12,16],
                       ):

    image_normal_path                      =  join(prep_session_dir,"flair_normal.nii.gz") 
    image_without_cerebellum_normal_path   =  join(prep_session_dir,"flair_wc_normal.nii.gz") 
    image_without_cerebellum_path          =  join(prep_session_dir,"flair_wc.nii.gz") 
    
    output_1                               = run(["flirt",
                                                  "-ref",atlas_path,
                                                  "-in", biascorrected_bet_image_path,
                                                  "-out",image_normal_path], 
                                                    capture_output=True)
    
    image_normal                         = nib.load(image_normal_path).get_fdata()
    atlas_d                              = nib.load(atlas_path)
    atlas_array                          = atlas_d.get_fdata()

    for l in cerebellum_labels:
        atlas_array[(atlas_array == l)  |  (atlas_array == 999)] = 999
        atlas_array[(atlas_array == l)] =  0
        
    atlas_array[atlas_array != 999] = 0
    atlas_array[atlas_array == 999] = 1
    
    image_withou_cerebellum = image_normal * np.logical_not(atlas_array)
    
    atlas_array_d = nib.Nifti1Image(image_withou_cerebellum, atlas_d.affine)
    nib.save(atlas_array_d, image_without_cerebellum_normal_path)
    
    output_1 = run(["flirt",
                    "-ref",biascorrected_bet_image_path,
                    "-in", image_without_cerebellum_normal_path,
                    "-out",image_without_cerebellum_path], 
                    capture_output=True)
    
    return image_without_cerebellum_path

def create_master_file_for_bianca(prepared_train_subjects_list,prep_session_dir, image_identifier = "bet", mni_bool=True):
        
    master_file_text_lines  = []
    trainstring       = ""
    test_list         = []
    test_dict         = {}
    row_number        = 0
    
    flair_path = ""
    mask_path = ""
    flair_to_mni = ""

    for train_index,train_path in enumerate(prepared_train_subjects_list):
 
        for subdir, dirs, files in os.walk(train_path):
            for file in files:
                
                if "flair.nii.gz" in file.lower():
                    #print(os.path.join(subdir, file))
                    flair_path = os.path.join(subdir, file)
                    print(f"train flair file : {file}")
                    
                if "wmhmask" in file.lower():
                    #print(os.path.join(subdir, file))
                    mask_path = os.path.join(subdir, file)
                    print(f"train mask file : {file}")
                    
                if mni_bool:
 
                    if "flair_to_mni" in file.lower():
                        #print(os.path.join(subdir, file))
                        flair_to_mni = os.path.join(subdir, file)
                        print(f"train  mni file : {file}")
 
                    
        if flair_path or flair_to_mni or mask_path:
            row_number+=1
            trainstring+=str(row_number)+","
                                
            master_file_train=    flair_path+" " +flair_to_mni+" "+mask_path
            master_file_text_lines.append(master_file_train)
        
    trainstring = trainstring[:-1]
    
    #print("            ",end="")
    images_list  = get_dat_paths(prep_session_dir)

    flair_path = ""
    mask_path = ""
    flair_to_mni = ""
    
    for image_idx,file in enumerate(images_list):
        b_file = os.path.basename(file).lower()
        if image_identifier in  b_file:
            flair_path =file
            print(f"test  mni file : {flair_path}")
            
        if "wmhmask" in b_file:
            mask_path = file
            print(f"test  mask file : {mask_path}")
            
        if mni_bool:
            if f"to_mni.mat" in  b_file:
                flair_to_mni = file   
                print(f"test  mni file : {flair_to_mni}")                       

    if flair_path or flair_to_mni or mask_path:
        row_number+=1
        test_list.append(row_number)
        master_file_test=    flair_path+" " +flair_to_mni+" "+mask_path
        master_file_text_lines.append(master_file_test)
        
    master_file_path = join(prep_session_dir,"masterfile.txt") 
    with open(master_file_path, 'w') as f:
        f.write('\n'.join(master_file_text_lines))
        
    return master_file_path,trainstring,row_number

    

def create_master_file(prepared_train, prepared_test=""):
    """
    

    Parameters
    ----------
    prepared_train : TYPE
        DESCRIPTION.
    prepared_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #temp_prepared_train_dir         =     create_result_dir(prepared_train+"_temp/","temp")         
    prepare_flair_train             =     prepared_train 
    #create_dir(prepared_train)
    prepared_train_subjects_list = get_all_dirs(prepare_flair_train)
    
    
    master_file_text_lines  = []
    trainstring       = ""
    test_list         = []
    test_dict         = {}
    row_number        = 0
    
    for train_index,train_path in enumerate(prepared_train_subjects_list):
 
        for subdir, dirs, files in os.walk(train_path):
            for file in files:
                
                print("train")
                

                if "flair.nii.gz" in file.lower():
                    #print(os.path.join(subdir, file))
                    flair_path = os.path.join(subdir, file)
                    print(f"file : {file}")
                    
                if "wmhmask" in file.lower():
                    #print(os.path.join(subdir, file))
                    mask_path = os.path.join(subdir, file)
                    print(f"file : {file}")
 
                if "flair_to_mni" in file.lower():
                    #print(os.path.join(subdir, file))
                    flair_to_mni = os.path.join(subdir, file)
                    
        if flair_path or flair_to_mni or mask_path:
            row_number+=1
            trainstring+=str(row_number)+","
                                
            master_file_train=    flair_path+" " +flair_to_mni+" "+mask_path
            master_file_text_lines.append(master_file_train)
        
    trainstring = trainstring[:-1]
    
    if prepared_test:
        
        prepare_flair_test              =     prepared_test 
        #create_dir(prepare_flair_test)
        prepared_test_subjects_list = get_all_dirs(prepare_flair_test)
        
                
        for test_index,test_path in enumerate(prepared_test_subjects_list):

            
            #if "07" in test_path.split("/")[-1]:  break
            
            prepared_test_session_list = get_all_dirs(test_path)

            for session_path in prepared_test_session_list:

                #print("            ",end="")
                images_list  = get_dat_paths(session_path)
                session = session_path.split("/")[-1]
                
                for index_of_image in range(1,20):

                    flair_path = ""
                    mask_path = ""
                    flair_to_mni = ""
                    
                    for image_idx,file in enumerate(images_list):
                        b_file = os.path.basename(file).lower()
                        
                        #print(b_file)
                        
                        if f"flair_{index_of_image}_bet.nii.gz" ==  b_file:
                            #print(os.path.join(subdir, file))
                            flair_path =file
                            #print(flair_path)
                            
                        if "wmhmask" in b_file:
                            #print(os.path.join(subdir, file))
                            mask_path = file
                            #print(mask_path)
                            #print(os.path.join(subdir, file))

                        if f"flair_{index_of_image}_to_mni.mat" ==  b_file:
                            flair_to_mni = file                          
                            #print(flair_to_mni)
                            

                    if flair_path and flair_to_mni and mask_path:
                        
                        row_number+=1
                        
                        test_list.append(row_number)
                        
                        master_file_test=    flair_path+" " +flair_to_mni+" "+mask_path
                        
                        print(flair_path)
                        print(flair_to_mni)
                        print(mask_path)
                        
                        master_file_text_lines.append(master_file_test)
                        test_dict[row_number] = {}
                        test_dict[row_number][session] = {}
                        
                        test_dict[row_number][session]["basename"] = os.path.basename(test_path)
                        test_dict[row_number][session]["test_path"] = test_path   
                        test_dict[row_number][session]["flair_path"] = flair_path   

    master_file_path = prepared_train+"/masterfile.txt" 

    try:
        os.system("rm "+master_file_path)
    except:
        pass
    
    with open(master_file_path, 'w') as f:
        f.write('\n'.join(master_file_text_lines))
        
        
    return master_file_path,trainstring,test_dict
    

    
def get_sessions_info(list_of_sessions, types_of_image = ["flair"]):

    session_dict = {}
    #create_dir(subject_dir)
    for session in list_of_sessions:
        session_index_name = session.split("/")[-1]
        session_dict[session_index_name] = {}
        
        for image_fun in next(os.walk(session))[1]:
            image_fun_path = join(session,image_fun)
            for image_type in next(os.walk(image_fun_path))[1]:
                
                if image_type in types_of_image:
                    
                    image_type_path = join(image_fun_path,image_type)
                    list_of_images  =  get_feature_paths(image_type_path,['nii','gz'])
    
                    for idx,image in enumerate(list_of_images):
                        session_dict[session_index_name][image_type] = {}
                        session_dict[session_index_name][image_type]["idx"] = idx
                        session_dict[session_index_name][image_type]["image_path"] = image

    return session_dict


import re



def create_data_dictionary_for_manual(data_set_belove, subject_id = 1):

    list_files = [join(data_set_belove,directory)\
                 for directory in os.listdir(data_set_belove)]
           
    file_base_names = {}
     
    # len(file_base_names)
    # len(list_files)
    
    for f,file in enumerate(list_files):
        found_roi =''
        found_nii = ''
        
        #break

        file_name = file.split("/")[-1]
        if file_name.lower().endswith("_roi.nii") or file_name.lower().endswith("-roi.nii"):
            file_base_name = file_name[:-8]
            found_roi = file_name
            
        elif file_name.lower().endswith(".nii"):
            file_base_name = file_name[:-4]
            found_nii = file_name
            
        print(file_base_name)
            
        if not file_base_name in file_base_names.keys():
            file_base_names[file_base_name] = {}
            
            file_base_names[file_base_name]["image"] = found_nii
            file_base_names[file_base_name]["mask"]  = found_roi
            
        elif found_roi:
            file_base_names[file_base_name]["mask"]  = found_roi
        elif found_nii:
            file_base_names[file_base_name]["image"] = found_nii
            
          
    file_paths = {}
    #subject_id = first_index
    
    for f,file in enumerate(file_base_names.items()):
        print(file)
        
        key = file[0]
        
        subject_id += 1
        
        file_id = f +1
        
        file_paths[file_id] = {}
        file_paths[file_id]["subject_id"]   = int(subject_id)
        file_paths[file_id]["mask"]      = join(data_set_belove,file[1]["mask"])
        file_paths[file_id]["flair"]     = join(data_set_belove,file[1]["image"])
        
        try:
            nib.load(file_paths[file_id]["mask"] )
        except:
            print("failed to load mask")
            continue
        
        try:
            nib.load(file_paths[file_id]["flair"] )
        except:
            print("failed to load mask")
            continue

            
    new_file_paths = {}
    session    = f"ses-01"
    
    for key in file_paths.keys():
        
        #print(key)
 
        sub_index = file_paths[key]['subject_id']
        subject_index_name = f"sub-{str(key).zfill(2)}"

        new_file_paths[key] = {}
        new_file_paths[key]["subject_index_name"] = subject_index_name
        new_file_paths[key]["session"]            = session
        
        new_file_paths[key]["flair"]            = {}
        new_file_paths[key]["flair"]['idx']     = 1
        new_file_paths[key]["flair"]['image_path']  = file_paths[key]['flair']
        new_file_paths[key]["flair"]['mask']        = file_paths[key]['mask']

        #new_file_paths[file_paths[key]['subject_id']]['t1'] = file_paths[key]['t1']
        
    return new_file_paths


def create_working_dict(bids_proscis_auto):
    
    list_of_bids_subjects = get_all_dirs(bids_proscis_auto)
    workin_dict = {}
    work_index = 0
    
    for idx, sub_idx in enumerate(list_of_bids_subjects):
        #print(idx)
        subject_index_name          = sub_idx.split("/")[-1]
        #subject_dir                 = join(prepared_dir,subject_index_name)
        #print(subject_index_name)
        list_of_sessions = get_all_dirs(sub_idx) 
        if not list_of_sessions: continue
        session_dict = get_sessions_info(list_of_sessions)
        
        for session, item in session_dict.items():
            #if no item found in session continue skip
            if not item:continue
            work_index+=1
            workin_dict[work_index] = {}
            workin_dict[work_index]["subject_index_name"] = subject_index_name
            workin_dict[work_index]["session"]            = session
            
            for key,value in item.items():
                workin_dict[work_index][key]            = value
                
    return workin_dict