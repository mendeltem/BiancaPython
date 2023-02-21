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

from modules.path_utils import create_result_dir,create_dir,get_main_paths,get_all_dirs


def create_challange_table(dataset_path):

    single_dataset_df =  pd.DataFrame()
    list_subjects   =     get_all_dirs(dataset_path)
    for f,files in enumerate(list_subjects):
        
        directory_id = os.path.basename(files)
        
        
        patient_df = pd.DataFrame(index=(int(directory_id),))
        
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
                    
     
        patient_df["mask_path"]  = mask_image
        patient_df["flair_image_path"]  = flair_image
        patient_df["t1_image_path"]  = t1_image
        
        
        single_dataset_df=pd.concat([single_dataset_df,patient_df])
        
    return single_dataset_df

