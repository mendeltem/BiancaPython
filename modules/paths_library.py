#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:07:29 2022

@author: temuuleu
"""

from os import listdir
from os.path import join,basename
import pandas as pd
import os
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype import Node, Workflow
from nipype.interfaces.fsl import Reorient2Std
from nipype.interfaces.fsl import SliceTimer, MCFLIRT, Smooth, BET,ApplyXFM,ConvertXFM

from nipype.interfaces.fsl import ApplyXFM
from nipype.interfaces.ants.segmentation import BrainExtraction

from os.path import join
from subprocess import run,call    
from os import listdir

import skimage.transform as skTrans
import nibabel as nib

from modules.path_utils import get_all_dirs


def create_master_file(train_subjects,
                       data_set_df,
                       null_subject_path,
                       test_flair_path,
                       test_flair_to_mni,
                       test_mask_path
                       ):
        

    master_file_text_lines  = []
    trainstring       = ""
    test_list         = []
    test_dict         = {}
    row_number        = 0
    
    flair_path = ""
    mask_path = ""
    flair_to_mni = ""
    
    for train_index,subject in enumerate(train_subjects):
        
        patient_df                =  data_set_df.loc[data_set_df['subject'] == subject,:]
        flair_path                =  patient_df['biascorrected_bet_image_path'].values[0]
        flair_to_mni              =  patient_df['MNI_xfm_path'].values[0]
        mask_path_n               =  os.path.join(os.path.dirname(flair_path),"WMHmask.nii")
        mask_path_g               =  os.path.join(os.path.dirname(flair_path),"WMHmask.nii.gz")
        
        if os.path.isfile(mask_path_n):
            mask_path = mask_path_n
        
        elif os.path.isfile(mask_path_g):
            mask_path = mask_path_g
             
        
        if flair_path or flair_to_mni or mask_path:
            row_number+=1
            trainstring+=str(row_number)+","
                                
            master_file_train=    flair_path+" " +flair_to_mni+" "+mask_path
            master_file_text_lines.append(master_file_train)
            
    trainstring = trainstring[:-1]
             
    if test_flair_path or test_flair_to_mni or test_mask_path:
        row_number+=1
        test_list.append(row_number)
        master_file_test=    test_flair_path+" " +test_flair_to_mni+" "+test_mask_path
        master_file_text_lines.append(master_file_test)
        
    master_file_path = join(null_subject_path,"masterfile.txt") 
    with open(master_file_path, 'w') as f:
        f.write('\n'.join(master_file_text_lines))
        
        
    return  master_file_path,trainstring,row_number


def create_challange_table(dataset_path,index_for_patient):
    
      
    hlm_list =     get_all_dirs(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    
    single_dataset_df =  pd.DataFrame()
        
    for hlm_i,hlm_path in enumerate(hlm_list):
        
        file_base_names = {}

        
        base_name = os.path.basename(hlm_path)
        
        print(base_name)
        
        list_files = [ os.path.join(hlm_path,directory)\
                     for directory in os.listdir(hlm_path)]
    

        list_subjects   =     get_all_dirs(hlm_path)
        
        
        
        for f,files in enumerate(list_subjects):
            
            directory_id = os.path.basename(files)
            
            
            patient_df = pd.DataFrame(index=(int(index_for_patient),))
            
            index_for_patient +=1
            
            list_files = [os.path.join(files,directory)\
                         for directory in os.listdir(files) if '.' in os.path.basename(directory).lower()]
                
            
            for file_path in list_files: 
                if 'kv' in basename(file_path).lower():
                    
                    mask_image = file_path
    
                elif '.nii' in basename(file_path).lower():
                    mask_image = file_path
                
    
            list_dirs  = get_all_dirs(files)
            
            for image_dir in list_dirs:
    
                if 'pre' in image_dir:
                    
                    flair_image = [join(image_dir,directory)\
                                 for directory in listdir(image_dir) if 'flair.' in basename(directory).lower()][0]
                        
    
                    t1_image = [join(image_dir,directory)\
                                 for directory in listdir(image_dir) if 't1.nii.gz' == basename(directory).lower() or\
                                     't1.nii' == basename(directory).lower() ][0]
                        
            patient_df["patient_id"] = directory_id
            patient_df["mask_path"]  = mask_image
            patient_df["flair_image_path"]  = flair_image
            patient_df["t1_image_path"]  = t1_image
            patient_df["dataset_name"] = dataset_name
            patient_df["lesion_concentration"] = base_name
            
            
           
            single_dataset_df=pd.concat([single_dataset_df,patient_df])
        
    return single_dataset_df,index_for_patient


def save_df(data_set_df,data_pp_path):
    
    try:
        data_set_df.to_excel(data_pp_path)
    except:
        print("resource busy:")
    

def create_belove_table(dataset_path,index_for_patient):

    hlm_list =     get_all_dirs(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    
    
    single_dataset_df =  pd.DataFrame()

    
    for hlm_i,hlm_path in enumerate(hlm_list):
        
        file_base_names = {}

        
        base_name = os.path.basename(hlm_path)
        
        print(base_name)
        
        list_files = [ os.path.join(hlm_path,directory)\
                     for directory in os.listdir(hlm_path)]
            

        for f,file in enumerate(list_files):

            found_roi =''
            found_nii = ''
            
            dname = os.path.dirname(file)


            file_name = file.split("/")[-1]
            if file_name.lower().endswith("_roi.nii") or file_name.lower().endswith("-roi.nii"):
                file_base_name = file_name[:-8]
                found_roi = file_name
                 
            elif file_name.lower().endswith(".nii"):
                file_base_name = file_name[:-4]
                found_nii = file_name
                
            #print(file_base_name)
                
            if not file_base_name in file_base_names.keys():
                file_base_names[file_base_name] = {}
                
                file_base_names[file_base_name]["image"] = found_nii
                file_base_names[file_base_name]["mask"]  = found_roi
                
            elif found_roi:
                file_base_names[file_base_name]["mask"]  = found_roi
            elif found_nii:
                file_base_names[file_base_name]["image"] = found_nii
        
     
        for f_key, f_value in file_base_names.items():
            
            if not len(f_value['mask']) >=1 or not len(f_value['image']) >=1 :
                continue
            patient_df = pd.DataFrame(index=(int(index_for_patient),))
            index_for_patient +=1
        
            patient_df["patient_id"] = f_value['image'][:-8]
            patient_df["dataset_name"] = dataset_name
            
            patient_df["lesion_concentration"] = base_name
            patient_df["flair_image_path"] = join(dname,f_value['image'])
            patient_df["t1_image_path"] = ""
            patient_df["mask_path"] = join(dname,f_value['mask'])
    
            single_dataset_df=pd.concat([single_dataset_df,patient_df])
        
    return single_dataset_df,index_for_patient



def create_tabel_for_all_subjects(all_patient_dataset_df,challange_df,subject_index):
    
    if challange_df.empty:
        return all_patient_dataset_df,subject_index
    

    subject_list = list(challange_df.loc[:,"patient_id"])

    for subject in subject_list[:]:
        
        subject_index+=1
        subject_index_name                      = f"sub-{str(subject_index).zfill(2)}"
        
        
        table_df = pd.DataFrame(index=(int(subject_index),))
        
        table_df["subject"] = subject_index_name
        

        pd_series = challange_df[challange_df["patient_id"] == subject]
        
        
        for c in challange_df.columns:
            table_df[c] = pd_series[c].values[0]


        all_patient_dataset_df = pd.concat([all_patient_dataset_df, table_df])
    
    return all_patient_dataset_df,subject_index



# def create_null_images(biascorrected_bet_image_path,
#                        mask_path, 
#                        null_subject_path,max_number =10 ):

#     flair_nib                                = nib.load(biascorrected_bet_image_path)
#     flair_3d_im                              = flair_nib.get_fdata()
    
 
#     mask_nib                                 = nib.load(mask_path)
#     mask_3d_im                               = mask_nib.get_fdata()
    

#     mask_sum_list  = [ np.sum(mask_3d_im[:,:,i]) for i in range( mask_3d_im.shape[-1])]
#     top_index_list = [position for rank,value,position in  rank_list(mask_sum_list,max_number)]
    
#     random_mask_index                               = np.random.choice(top_index_list)
    

#     for ti,top_index in enumerate(top_index_list):
        
#         null_maske_3d = np.zeros(mask_3d_im.shape)
#         random_mask                                        = mask_3d_im[:,:,top_index]
#         inverted_random_mask                               = 1 - random_mask
        
#         null_maske_3d[:,:,random_mask_index]               = random_mask
        
#         null_flair_3d_im                                = normalize_volume(flair_3d_im.copy())
        
#         null_slice  = null_flair_3d_im[:,:,random_mask_index] *  inverted_random_mask
        
#         null_flair_3d_im[:,:,random_mask_index]         = null_slice
        
#         ni_img          = nib.Nifti1Image(null_flair_3d_im ,
#                                             flair_nib.affine)
        
#         nib.save(ni_img, f"{null_subject_path}/flair_{ti}_null.nii.gz")
        
#         ni_img          = nib.Nifti1Image(null_maske_3d ,
#                                             flair_nib.affine)
        
#         nib.save(ni_img, f"{null_subject_path}/mask_{ti}_null.nii.gz")
        
    
def rank_list(lst, n):
    "gives me the top n rank and the list in dictionary "
    ranked_lst = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)[:n]
    rank_info = []
    for i, (idx, val) in enumerate(ranked_lst):
        rank_info.append((i+1, val, idx))
    return rank_info
    

def normalize_volume(image):
    """normalize 3D Volume image"""
    
    
    dim     = image.shape
    volume = image.reshape(-1, 1)
    
    rvolume = np.divide(volume - volume.min(axis=0), (volume.max(axis=0) - volume.min(axis=0)))*1
    rvolume = rvolume.reshape(dim)
    
    return rvolume