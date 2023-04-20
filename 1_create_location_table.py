#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:00:46 2023

@author: temuuleu
"""

import os
import pandas as pd

from pathlib import Path
import configparser
import multiprocessing as mp
from multiprocessing import Manager
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


from libraries.paths_library import create_challange_table,create_belove_table,get_volume
from libraries.paths_library import create_tabel_for_all_subjects


manager = Manager()
global_dict = manager.dict()

#pip install nipype
#pip install scikit-image

#load personal libraries

config_dict = configparser.ConfigParser()
config_dict.read('param.ini')   

#standard images
prepared_bianca                       =     config_dict['STANDARD-PATHS']["standard_mask"]

if os.path.isfile(prepared_bianca):
    print(f"found {prepared_bianca}")
else:
    raise ValueError("Error: standard mask not found")


standard_space_flair                  =     config_dict['STANDARD-PATHS']["standard_space_flair"]

if os.path.isfile(standard_space_flair):
    print(f"found {standard_space_flair}")
else:
    raise ValueError("Error: standard standard_space_flair not found")

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
else:
    os.mkdir(datasets_prepared_path)
    
    
num_processes = 16
    
# label data sets 
cohort_datasets_dir  =   Path(datasets_prepared_path) / "cohort_datasets" 
#create directory
if not os.path.isdir(cohort_datasets_dir): os.mkdir(cohort_datasets_dir)
    
#create_study preparations
data_location_table_path  = cohort_datasets_dir / "data_original_location.xlsx"


if os.path.isfile(data_location_table_path):
    print(f"found {data_location_table_path}")
else:
    raise ValueError("Error: ata_original_location.xlsx not found")
    
    
    

def create_location_data(patien_dir):

    original_patien_name    = patien_dir.split('/')[-1]
    
    patient_df = pd.DataFrame(index=(1,))
    patient_df["original_patien_name"] = original_patien_name

    #print(patien_dir)
    files_list = os.listdir(patien_dir)
    
    for file in files_list:
        
        #print(file)
        
        if "bin" in file.lower():
            pass
        elif file.lower().endswith("-roi.nii") or file.lower().endswith("-roi.nii.gz"):
            patient_df["name_mask"]            = file
            patient_df["path_mask"]       = os.path.join(patien_dir,file)
        elif file.lower().endswith("_roi.nii") or file.lower().endswith("_roi.nii.gz"):
            patient_df["name_mask"]            = file
            patient_df["path_mask"]       = os.path.join(patien_dir,file)
            
        elif "mul" in file.lower() and not "roi" in file.lower():
            patient_df["name_null_image"]      = file
            patient_df["path_null_image"] = os.path.join(patien_dir,file)
            
        elif "fs_tra" in file.lower() or "tra_pre" in file.lower() or "flair" in file.lower() and not "roi" in file.lower():
            patient_df["name_flair_image"]      = file
            patient_df["path_flair_image"] = os.path.join(patien_dir,file)
            

        if "mprage" in file.lower() and "corrected" in file.lower() or \
            "mprage" in file.lower() and "cp" in file.lower() or \
                "mprage" in file.lower() and "cip" in file.lower():
            patient_df["mprage_pp"] = file
            patient_df["mprage_volume"] = round(get_volume( os.path.join(patien_dir,file)) ,2)

            patient_df["path_mprage_pp"] = os.path.join(patien_dir,file)
            
        elif "mprage" in file.lower():
            patient_df["mprage"] = file
            patient_df["path_mprage"] = os.path.join(patien_dir,file)
            
             
        if "bin" in file.lower():
            patient_df["type"] = file
            patient_df["path_type"] = os.path.join(patien_dir,file)
            
    
    return pd.DataFrame(patient_df)



cohort_patient_dataset_df = pd.DataFrame()

# """creates """
for d_i, dataset_path in enumerate(cohort_studies_list):

    base_name = os.path.basename(dataset_path)
    
    patient_dir_list = [os.path.join(dataset_path, p) for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, p))]

    #for pi,patien_dir in enumerate(patient_dir_list):
        
    print(f"number of patients {    len(patient_dir_list)}  should be")
          
    # Calculate the size of each sub-list
    sub_list_size = len(patient_dir_list) // num_processes

    # Split the original list into sub-lists
    sub_lists = [patient_dir_list[i:i+num_processes] for i in range(0, len(patient_dir_list), num_processes)]
        
    # Print out each sub-list
    for i, sub_list in enumerate(sub_lists):
        number_of_data = 0

        pool = mp.Pool(processes=num_processes)
        result = pool.map(create_location_data, sub_list)
        
        print(pd.concat(result).shape)
        cohort_patient_dataset_df = pd.concat([cohort_patient_dataset_df, pd.concat(result)], ignore_index=True)
        
        
    cohort_patient_dataset_df["subject"] = ""
    for i in range(len(cohort_patient_dataset_df)):
        cohort_patient_dataset_df.at[i, "subject"] = f"{cohort_patient_dataset_df.at[i, 'subject']}{i+1}"
        
      
    cohort_patient_dataset_df['subject'] = cohort_patient_dataset_df['subject'].astype(int).apply(lambda x: f"sub-{x:02d}") 
    
    cohort_patient_dataset_df = cohort_patient_dataset_df[["subject"]+[l for l in list(cohort_patient_dataset_df.columns) if not "subject" in l]]
    
    number_of_subjects = len(cohort_patient_dataset_df)
    
    print(f"number_of_subjects is located : {number_of_subjects}")
    cohort_patient_dataset_df.to_excel(data_location_table_path)  



# label data sets 
label_datasets_dir  =   Path(datasets_prepared_path) / "label_datasets" 
#create directory
if not os.path.isdir(label_datasets_dir): os.mkdir(label_datasets_dir)
    
#create_study preparations
data_location_table_path              =    label_datasets_dir / "data_original_location.xlsx"

index_for_patient = 1
subject_index = 0

label_patient_dataset_df =  pd.DataFrame()


"""creates """
for d_i, dataset_path in enumerate(label_datasets_list):

    base_name = os.path.basename(dataset_path)
     
    if "challenge" in  base_name.lower():
        
        print(base_name)
        challange_df,index_for_patient = create_challange_table(dataset_path,index_for_patient)
        print(challange_df.columns)
        
        label_patient_dataset_df,subject_index= create_tabel_for_all_subjects(label_patient_dataset_df,challange_df,subject_index)
        
    if "belove" in  base_name.lower():
        
        print(base_name)
        
        belove_df,index_for_patient = create_belove_table(dataset_path,index_for_patient)
        
        label_patient_dataset_df,subject_index= create_tabel_for_all_subjects(label_patient_dataset_df,
                                                                                         belove_df,
                                                                                         subject_index)
        
        
label_patient_dataset_df.to_excel(data_location_table_path)          

