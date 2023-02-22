#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:11:57 2023

@author: temuuleu
"""
import os
import numpy as np
import pandas as pd
from modules.path_utils import create_dir,get_main_paths,get_all_dirs
from modules.paths_library import create_null_images
from os.path import join

config_dict =get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]
datasets_list                         =     get_all_dirs(datasets_original_path)
data_location_table_path              =     os.path.join(datasets_prepared_path,"data_location.xlsx")
data_location_table_df                =     pd.read_excel(data_location_table_path)
subject_list                          =     list(data_location_table_df.loc[:,"subject"])
data_pp_path                          =     os.path.join(datasets_prepared_path,"data_prepared_data.xlsx")
subject_list                          =     list(data_location_table_df.loc[:,"subject"])
data_pp_path                          =     os.path.join(datasets_prepared_path,"data_prepared_data.xlsx")
data_set_df                           =     pd.read_excel(data_pp_path,index_col=None)


Set_1_path                            =     os.path.join(datasets_prepared_path,"Set_1")

nulled_images_path                    =     os.path.join(datasets_prepared_path,"nulled_images")
create_dir(nulled_images_path)
data_pp_path                          =     os.path.join(datasets_prepared_path,"data_prepared_data.xlsx")


for subject in subject_list[48:]:
    
    print(subject)
    
    patient_df                                = data_set_df.loc[data_set_df['subject'] == subject,:]
    
    patient_id                                =  patient_df["patient_id"].values[0]
    dataset_name                              =  patient_df["dataset_name"].values[0]

    biascorrected_bet_image_path              =  patient_df["biascorrected_bet_image_path"].values[0]
    biascorrected_normalized_image_path       =  patient_df["biascorrected_normalized_image_path"].values[0]
    MNI_xfm_path                              =  patient_df["MNI_xfm_path"].values[0]
    
    mask_path_n                                =  os.path.join(os.path.dirname(biascorrected_bet_image_path),"WMHmask.nii")
    mask_path_g                                =  os.path.join(os.path.dirname(biascorrected_bet_image_path),"WMHmask.nii.gz")
    
    if os.path.isfile(mask_path_n):
        mask_path = mask_path_n
    
    elif os.path.isfile(mask_path_g):
        mask_path = mask_path_g
         
    else: 
        print("continue")
        continue

    mask_normalized_path                     =  patient_df["mask_normalized_path"].values[0]
    
    
    null_subject_path                        =  join(nulled_images_path,subject)
    
    prep_flair_path                        =  join(null_subject_path,    os.path.basename(biascorrected_bet_image_path))
    prep_mask_path                         =  join(null_subject_path,    os.path.basename(mask_path))
    prep_MNI_xfm_path                      =  join(null_subject_path,    os.path.basename(MNI_xfm_path))
    
    create_dir(null_subject_path)

    os.system(f"cp {biascorrected_bet_image_path} {prep_flair_path}")
    os.system(f"cp {mask_path} {prep_mask_path}")
    os.system(f"cp {MNI_xfm_path} {prep_MNI_xfm_path}")
    
    create_null_images(biascorrected_bet_image_path,
                           mask_path, 
                           null_subject_path )

            

    
