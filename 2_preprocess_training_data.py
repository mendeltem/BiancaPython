#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:12:14 2023

@author: temuuleu
"""

import os
import pandas as pd

from pathlib import Path
import multiprocessing as mp
from multiprocessing import Manager, Pool
from concurrent.futures import ThreadPoolExecutor
from skimage.filters import threshold_otsu
#load personal libraries
os.chdir("../libraries")

from path_utils import create_result_dir,create_dir,get_main_paths,get_all_dirs
from paths_library import create_challange_table,create_belove_table,get_volume
from paths_library import create_tabel_for_all_subjects
from bianca_post_process_lib import create_null_image


import matplotlib.pyplot as plt


def create_null_mask(path_null_image,path_flair_image,null_mask_image_path ):

    try:
        controll_nib = nib.load(path_flair_image)
        controll_img = controll_nib.get_fdata()
        
        null_img = nib.load(path_null_image).get_fdata()
        
        img_diff = np.abs(controll_img - null_img)
        
        
        controll_thresholded_img = np.greater(controll_img, 0.001).astype(np.uint8) 
        null_thresholded_img = np.greater(null_img, 0.001).astype(np.uint8) 
        

        null_mask = np.abs(controll_img - null_img)
        
        output_mask_copy = np.copy(null_mask)
        
    
        output_mask_copy[output_mask_copy >=0.01] = 1
        output_mask_copy[output_mask_copy <0] = 0
        

        null_image_nib = nib.Nifti1Image(output_mask_copy, controll_nib.affine)
        nib.save(null_image_nib, null_mask_image_path)
        return 0
    except:
        return 1



import nibabel as nib
import numpy as np
import subprocess

manager = Manager()
global_dict = manager.dict()


os.chdir("../")

config_dict =get_main_paths()

#standard images
prepared_bianca                       =     config_dict['STANDARD-PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['STANDARD-PATHS']["standard_space_flair"]

#loading original datsets
label_datasets_list                   =   [config_dict['LABEL-DATA-SET-PATHS'][dset] for dset in list(config_dict['LABEL-DATA-SET-PATHS'])]

#loading original datsets
cohort_studies_list                   =   [config_dict['COHORT-DATA-SET-PATHS'][dset] for dset in list(config_dict['COHORT-DATA-SET-PATHS'])]

processed_paths_dict                  =   {dset: config_dict['PROCESSED-SET-PATHS'][dset] for dset in config_dict['PROCESSED-SET-PATHS']}
found_set                             =    [(dset,os.path.isdir(path)) for dset,path in processed_paths_dict.items()]

print(found_set)

datasets_prepared_path                =   processed_paths_dict['datasets_prepared']
if os.path.isdir(datasets_prepared_path):
    print(f"found {datasets_prepared_path}")
    
    
    
num_processes = 16
    
cohort_datasets_dir = Path(datasets_prepared_path) / "cohort_datasets"
cohort_datasets_dir.mkdir(parents=True, exist_ok=True)

data_location_table_path = cohort_datasets_dir / "data_original_location.xlsx"

data_set = cohort_datasets_dir / "set_1"
data_set.mkdir(parents=True, exist_ok=True)

data_preprocessed_set = data_set / "preprocessed"
data_preprocessed_set.mkdir(parents=True, exist_ok=True)


preprocessed_table_path  =data_set/"preprocessed.xlsx"


data_location_table_df   = pd.read_excel(data_location_table_path, index_col=0)


subject_list = list(data_location_table_df["subject"])


null_image_name ="flair_null.nii.gz"

controll_image_name ="flair_controll.nii.gz"

null_mask_name ="mask_null.nii.gz"

def preprocess_subject(data):
    
    subject,data_location_table_df=data

    subject_df = pd.DataFrame(index=(1,))
    
    subject_row   = data_location_table_df[data_location_table_df["subject"] == subject]
         
    subject = subject_row["subject"].values[0]
    original_patien_name = subject_row["original_patien_name"].values[0]

    name_mask = subject_row["name_mask"].values[0]
    path_mask = subject_row["path_mask"].values[0]
    name_null_image = subject_row["name_null_image"].values[0]
    path_null_image = subject_row["path_null_image"].values[0]
    name_flair_image = subject_row["name_flair_image"].values[0]
    path_flair_image = subject_row["path_flair_image"].values[0]
    mprage_pp = subject_row["mprage_pp"].values[0]
    mprage_volume = subject_row["mprage_volume"].values[0]
    path_mprage_pp = subject_row["path_mprage_pp"].values[0]
    mprage = subject_row["mprage"].values[0]
    path_mprage = subject_row["path_mprage"].values[0]

    # Get column names with NaN values
    nan_columns = subject_row.columns[subject_row.isna().any()].tolist()
    
    subject_dir = data_preprocessed_set / subject
    # subject_dir.mkdir(parents=True, exist_ok=True)
    
    null_image_path         = subject_dir / null_image_name
    controll_image_path     = subject_dir / controll_image_name
    
    null_mask_image_path     = subject_dir / null_mask_name
    
    subject_df["subject"] = subject
        
        
    # #if null image is not found create own with the help of the mask
    # if "name_null_image" in  nan_columns and not "name_mask" in nan_columns:
    #     print(f"NaN values detected in {nan_columns} for {subject}")
    #     create_null_image(path_mask,path_flair_image,null_image_path )
          
    # elif not "name_null_image" in  nan_columns:
        
    #     subject_dir = data_preprocessed_set / subject
    #     subject_dir.mkdir(parents=True, exist_ok=True)
    #     subprocess.run(["cp", path_null_image, null_image_path], check=True)
        
    # subject_df["name_null_image"] = null_image_path

    # #check for missing rage
    # #if copy the controll flair image
    # if not "path_flair_image" in  nan_columns:
    #     print(f"NaN values detected in {nan_columns} for {subject}")
    #     subprocess.run(["cp", path_flair_image, controll_image_path], check=True)
        
        
        
    # #check for missing rage
    # #if copy the controll flair image
    # if not "path_mask" in  nan_columns:
    #     print(f"NaN values detected in {nan_columns} for {subject}")
    #     subprocess.run(["cp", path_mask, null_mask_image_path], check=True)
        
    #check for missing rage
    #if copy the controll flair image
    if "path_mask" in  nan_columns and not "name_null_image" in  nan_columns:
        print(f"NaN values detected in {nan_columns} for {subject}")
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        
        create_null_mask(path_null_image,path_flair_image,null_mask_image_path )
        
        
        
        #subprocess.run(["cp", path_mask, null_mask_image_path], check=True) 
        
        
        
    #check for missing rage
    #if null image is not found create own with the help of the mask
    if not "mprage_volume" in  nan_columns:
        print(f"NaN values detected in {nan_columns} for {subject}")
        subject_df["mprage_volume"] = mprage_volume
        
        
    return pd.DataFrame(subject_df)
  




cohort_subject_dataset_df = pd.DataFrame()


#subject_list = subject_list[:10]
subjects_len = subject_list

#for pi,patien_dir in enumerate(patient_dir_list):
    
print(f"number of patients {    len(subjects_len)}  should be")

num_processes = 8
      
# Calculate the size of each sub-list
sub_list_size = len(subject_list) // num_processes

# Split the original list into sub-lists
sub_lists = [subject_list[i:i+num_processes] for i in range(0, len(subject_list), num_processes)]


# Print out each sub-list
for i, sub_list in enumerate(sub_lists):
    number_of_data = 0
    
    #create pool for multiprocess
    pool = mp.Pool(processes=num_processes)
    
    # create an empty list to store the concatenated dataframes
    concatenated_dfs = []
    
    # loop through the values and concatenate the dataframes for each value
    for value in sub_list:
        concatenated_dfs.append((value,data_location_table_df))
        
     
    for df in concatenated_dfs:
        print(df)
        result = preprocess_subject(df)
        
    #result = pool.map(preprocess_subject, concatenated_dfs)

    if result.shape[0] == 1: continue 
    print(pd.concat(result).shape)
    
    
    cohort_subject_dataset_df = pd.concat([cohort_subject_dataset_df, pd.concat(result)], ignore_index=True)
    


print(cohort_subject_dataset_df.info())
print(cohort_subject_dataset_df.head())

cohort_subject_dataset_df.to_excel(preprocessed_table_path)


# def copy_mask(mask_name,flair_image_path,prep_subject_dir):

    
#     if mask_name.endswith(".nii.gz"):
#         standard_mask_name                    =  f"WMHmask.nii.gz" 	#brain extracted
#     elif mask_name.endswith(".nii"):
#         standard_mask_name                    =  f"WMHmask.nii" 	#brain extracted
    
    
#     standard_mask_path   =  join(prep_subject_dir,standard_mask_name)
    
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




    
    