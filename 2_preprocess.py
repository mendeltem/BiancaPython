#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:12:14 2023

@author: temuuleu
"""
import os
os.chdir("/home/temuuleu/CSB_NeuroRad/temuuleu/Projekts/BIANCA/libraries")

import pandas as pd

from pathlib import Path
import multiprocessing as mp
from multiprocessing import Manager, Pool
from concurrent.futures import ThreadPoolExecutor
from skimage.filters import threshold_otsu
#load personal libraries


from path_utils import create_result_dir,create_dir,get_main_paths,get_all_dirs

#from paths_library import create_challange_table,create_belove_table,get_volume
#from paths_library import create_tabel_for_all_subjects


from bianca_preprocess import preprocess_image,segment_image,create_null_mask
from bianca_preprocess import create_null_image,preprocess_subject

import matplotlib.pyplot as plt


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
    
    
    
# create preprocessing directory for cohort    
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


    
data_dict = {"data_preprocessed_set": data_preprocessed_set,
             "standard_space_flair" : standard_space_flair
             }


cohort_subject_dataset_df = pd.DataFrame()


#subject_list = subject_list[:17]
subjects_len = subject_list

    
print(f"number of patients {    len(subjects_len)}  should be")

num_processes = 16

print(f"number of patients {    len(subjects_len)}  should be")
      
# Calculate the size of each sub-list
sub_list_size = len(subject_list) // num_processes

# Split the original list into sub-lists
sub_lists = [subject_list[i:i+num_processes] for i in range(0, len(subject_list), num_processes)]


# Print out each sub-list
for i, sub_list in enumerate(sub_lists):

    #create pool for multiprocess
    pool = mp.Pool(processes=num_processes)
    
    # create an empty list to store the concatenated dataframes
    concatenated_dfs = []
    
    # loop through the values and concatenate the dataframes for each value
    for value in sub_list:
        concatenated_dfs.append((value,data_location_table_df,data_dict))
        
        
    # for df in concatenated_dfs:
    #     print(df)
    #     result = preprocess_subject(df)
        
    result = pool.map(preprocess_subject, concatenated_dfs)

    #if len(result) == 1: continue 
    print(pd.concat(result).shape)
    
    cohort_subject_dataset_df = pd.concat([cohort_subject_dataset_df, pd.concat(result)], ignore_index=True)
    
    print(cohort_subject_dataset_df.info())
    print(cohort_subject_dataset_df.head())
    
    cohort_subject_dataset_df.to_excel(preprocessed_table_path)






    
    