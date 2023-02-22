#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:21:27 2023

Prepare the dataset for bianca training


Create Data Location Table where collect all the necesarry data location and 
save it in an excel table.

"""

import os
import pandas as pd

from modules.path_utils import create_result_dir,create_dir,get_main_paths,get_all_dirs
from modules.paths_library import create_challange_table,create_belove_table
from modules.paths_library import create_tabel_for_all_subjects

config_dict =get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]

datasets_list                         =     get_all_dirs(datasets_original_path)


data_location_table_path              =   os.path.join(datasets_prepared_path,"data_location.xlsx")


index_for_patient = 1
subject_index = 0


all_patient_dataset_df =  pd.DataFrame()

"""creates """
for d_i, dataset_path in enumerate(datasets_list):
    
    base_name = os.path.basename(dataset_path)
     
    if "challenge" in  base_name.lower():

        challange_df,index_for_patient = create_challange_table(dataset_path,index_for_patient)
        print(challange_df.columns)
        
        all_patient_dataset_df,subject_index= create_tabel_for_all_subjects(all_patient_dataset_df,challange_df,subject_index)
        
  
    if "belove" in  base_name.lower():

        belove_df,index_for_patient = create_belove_table(dataset_path,index_for_patient)
        
        all_patient_dataset_df,subject_index= create_tabel_for_all_subjects(all_patient_dataset_df,
                                                                                         belove_df,
                                                                                         subject_index)
        
               
all_patient_dataset_df.to_excel(data_location_table_path)
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        

    
    
    
    
    