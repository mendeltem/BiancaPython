#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:21:27 2023

@author: temuuleu


Prepare the dataset for bianca training

"""


import os
import pandas as pd

from modules.path_utils import create_result_dir,create_dir,get_main_paths,get_all_dirs
from modules.paths_library import create_challange_table


config_dict =get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]

datasets_list                         =     get_all_dirs(datasets_original_path)



for d_i, dataset_path in enumerate(datasets_list):
    

    base_name = os.path.basename(dataset_path)
    if "challenge" in  base_name.lower():
        
        challange_df = create_challange_table(dataset_path)
        
        
    if "klinik" in  base_name.lower():
        
        print(klinik)
        
        challange_df = create_challange_table(dataset_path)
    
    

                
                        
                        

        
        
        
        
        
        
        
        
        
        
        
        

    
    
    
    
    