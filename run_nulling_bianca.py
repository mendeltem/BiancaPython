#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:52:58 2023

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


from bianca_preprocess import preprocess_image,segment_image,create_null_mask
from bianca_preprocess import create_null_image,preprocess_subject,create_masterfile
from bianca_preprocess import run_bianca
import matplotlib.pyplot as plt


import nibabel as nib
import numpy as np
import subprocess


#pip install scikit-image

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
    
output_dir                =   processed_paths_dict['output_directory']
if os.path.isdir(output_dir):
    print(f"found {output_dir}")
     
    
                 
# create preprocessing directory for cohort    
num_processes = 16
    
cohort_datasets_dir = Path(datasets_prepared_path) / "cohort_datasets"
cohort_datasets_dir.mkdir(parents=True, exist_ok=True)

data_location_table_path = cohort_datasets_dir / "data_original_location.xlsx"


cohort_output_datasets_dir = Path(output_dir) / "cohort_datasets"
cohort_output_datasets_dir.mkdir(parents=True, exist_ok=True)

restul_set = cohort_output_datasets_dir / "result_1"
restul_set.mkdir(parents=True, exist_ok=True)

bianca_result_set = restul_set / "bianca"
bianca_result_set.mkdir(parents=True, exist_ok=True)

bianca_result_table_path  = bianca_result_set/"bianca_result.xlsx"


data_set = cohort_datasets_dir / "set_1"
data_set.mkdir(parents=True, exist_ok=True)


preprocessed_table_path  = data_set/"preprocessed.xlsx"

if os.path.isfile(preprocessed_table_path):
    print(f"Found {preprocessed_table_path}")
    preprocessed_table_df    = pd.read_excel(preprocessed_table_path,index_col=0)
    
    subject_list = list(preprocessed_table_df["subject"])
else:
    raise ValueError("File not found at specified path.")



print(f"number of patients {    len(subject_list)}  should be")


def run_bianca_subject(arg):
    
    subject, dataframe,data_dict = arg
    
    subject_row   = dataframe[dataframe["subject"] == subject]
    

    bianca_result_set  = data_dict["bianca_result_set"]
    
    subject_dir = bianca_result_set / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    
    train_masterfile_path  = data_dict["train_masterfile_path"]
    standard_mask  = data_dict["standard_mask"]
    
    
    null_pp_path = subject_row["null_pp_path"].values[0]
    mni_matrix = subject_row["mni_matrix"].values[0]
    controll_pp_path = subject_row["controll_pp_path"].values[0]
    
    
    standard_space_ventrikel = data_dict["standard_space_ventrikel"]
    
    ventrikel_array = nib.load(standard_space_ventrikel).get_fdata()
    
    bianca_result_df = pd.DataFrame(index=(1,))
    
    
    bianca_result_df = subject_row
    
    #bianca_result_df["subject"] = subject
    
    
    output_mask                             = "seg_prob_bianca_controll.nii.gz"
    
    output_mask_mni                         = 'seg_prob_bianca_controll_mni.nii.gz'
    
    output_mask_mni_path                    = os.path.join(subject_dir,output_mask_mni)
    
    output_mask_mni_corrected                = 'seg_prob_bianca_controll_mni_corrected.nii.gz'

    output_mask_mni_corrected_path          = os.path.join(subject_dir,output_mask_mni_corrected)

    new_master_file_path,trainstring,row_number  = create_masterfile(train_masterfile_path,
                                    controll_pp_path,
                                    mni_matrix,
                                    standard_mask,
                                    subject_dir
                                    )
    
    print(new_master_file_path)
    output_mask_path               =    run_bianca(new_master_file_path,  
                                            str(subject_dir),
                                            trainstring,row_number,
                                            brainmaskfeaturenum       = 1,
                                            spatialweight             = 2,
                                            labelfeaturenum           = 3,
                                            trainingpts               = 10000,
                                            selectpts                 = "noborder",
                                            nonlespts                 = 10000,
                                            output_mask  = output_mask)
    bianca_result_df["controll_bianca"] = output_mask_path
    
    if not os.path.isfile(output_mask_mni_path):
    
        normal_mask_cmd = f"flirt -in {output_mask_path}  -ref {standard_space_flair}  -out {output_mask_mni_path} -init {mni_matrix} -applyxfm"
        os.system(normal_mask_cmd)
    
    bianca_result_df["controll_bianca_mni"] = output_mask_mni_path
    
    
    if not os.path.isfile(output_mask_mni_corrected_path):
        bianca_normalized_nib = nib.load(output_mask_mni_path)
        bianca_normalized_array = bianca_normalized_nib.get_fdata()
        
        bianca_ventrikel_corrected_array = (np.logical_not(ventrikel_array[:,:,:])) * bianca_normalized_array[:,:,:]
        null_image_nib = nib.Nifti1Image(bianca_ventrikel_corrected_array, bianca_normalized_nib.affine)
        nib.save(null_image_nib, output_mask_mni_corrected_path)
        
    bianca_result_df["controll_bianca_mni_ven_corrected"] = output_mask_mni_path
    
    
    output_mask                             = "seg_prob_bianca_null.nii.gz"
    
    output_mask_mni                         = 'seg_prob_bianca_null_mni.nii.gz'

    output_mask_mni_path                    = os.path.join(subject_dir,output_mask_mni)
    
    output_mask_mni_corrected                = 'seg_prob_bianca_null_mni_corrected.nii.gz'

    output_mask_mni_corrected_path          = os.path.join(subject_dir,output_mask_mni_corrected)

    new_master_file_path,trainstring,row_number  = create_masterfile(train_masterfile_path,
                                    null_pp_path,
                                    mni_matrix,
                                    standard_mask,
                                    subject_dir
                                    )
    
    
    print(new_master_file_path)
    
    
    output_mask_path               =    run_bianca(new_master_file_path,  
                                            str(subject_dir),
                                            trainstring,row_number,
                                            brainmaskfeaturenum       = 1,
                                            spatialweight             = 2,
                                            labelfeaturenum           = 3,
                                            trainingpts               = 10000,
                                            selectpts                 = "noborder",
                                            nonlespts                 = 10000,
                                            output_mask  = output_mask)
    
    bianca_result_df["null_bianca"] = output_mask_path
    
    
    if not os.path.isfile(output_mask_mni_path):
        normal_mask_cmd = f"flirt -in {output_mask_path}  -ref {standard_space_flair}  -out {output_mask_mni_path} -init {mni_matrix} -applyxfm"
        os.system(normal_mask_cmd)
        
    bianca_result_df["null_bianca_mni"] = output_mask_mni_path
    
    if not os.path.isfile(output_mask_mni_corrected_path):
        
        bianca_normalized_nib = nib.load(output_mask_mni_path)
        bianca_normalized_array = bianca_normalized_nib.get_fdata()
        bianca_ventrikel_corrected_array = (np.logical_not(ventrikel_array[:,:,:])) * bianca_normalized_array[:,:,:]
        null_image_nib = nib.Nifti1Image(bianca_ventrikel_corrected_array, bianca_normalized_nib.affine)
        nib.save(null_image_nib, output_mask_mni_corrected_path)

    bianca_result_df["null_bianca_mni_ven_corrected"] = output_mask_mni_corrected_path
    
    #standard_space_ventrikel
    
    return pd.DataFrame(bianca_result_df)
    
    
num_cpus = mp.cpu_count()
print("Number of CPUs: ", num_cpus)

num_processes = num_cpus

# Calculate the size of each sub-list
sub_list_size = len(subject_list) // num_processes

# Split the original list into sub-lists
sub_lists = [subject_list[i:i+num_processes] for i in range(0, len(subject_list), num_processes)]


train_masterfile_path = '/home/temuuleu/CSB_NeuroRad/temuuleu/Projekts/BELOVE/Bianca_Project/bianca_version_3/master_files/all_subjects_train_master_file.txt'

os.path.isfile(train_masterfile_path)

standard_space_ventrikel     = '/home/temuuleu/CSB_NeuroRad/temuuleu/DATASETS/STANDARD/standard_templates/HarvardOxford-1mm-latvent-dilated.nii.gz'

os.path.isfile(standard_space_ventrikel)  
  
  
data_dict = {"bianca_result_set": bianca_result_set,
             "train_masterfile_path" : train_masterfile_path,
             "standard_mask" : prepared_bianca,
             "standard_space_ventrikel" : standard_space_ventrikel
             }


bianca_results = pd.DataFrame()


# Print out each sub-list
for i, sub_list in enumerate(sub_lists):

    #create pool for multiprocess
    pool = mp.Pool(processes=num_processes)
    
    # create an empty list to store the concatenated dataframes
    concatenated_dfs = []
    
    # loop through the values and concatenate the dataframes for each value
    for value in sub_list:
        concatenated_dfs.append((value,preprocessed_table_df,data_dict))
    
    # for df in concatenated_dfs:
    #     print(df)
    #     result = run_bianca_subject(df)
        
        
    result = pool.map(run_bianca_subject, concatenated_dfs)
    
    
    bianca_results = pd.concat([bianca_results, pd.concat(result)], ignore_index=True)
    
    print(bianca_results.info())
    print(bianca_results.head())
    
    bianca_results.to_excel(bianca_result_table_path)
    