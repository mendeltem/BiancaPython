#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:44:44 2023

@author: temuuleu
"""

import os
import numpy as np
import pandas as pd

from os.path import join
import subprocess
from modules.path_utils import get_main_paths,get_all_dirs

 


config_dict =get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]


data_pp_path                          =     os.path.join(datasets_prepared_path,"data_prepared_data.xlsx")
data_set_df                           =     pd.read_excel(data_pp_path,index_col=None)


nulled_images_path                    =     os.path.join(datasets_prepared_path,"nulled_images")



bianca_result_path                    =     os.path.join(datasets_prepared_path,"bianca_result_seed_1")



nulled_images_paths_list    = get_all_dirs(nulled_images_path)
bianca_result_paths_list    = get_all_dirs(bianca_result_path)


for bri, subject_null_image_path in enumerate(nulled_images_paths_list):


    print(subject_null_image_path)
    
    
    flair_image               = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "bet" in f][0]
    mask_image               = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "WMHmask" in f][0]
 
    flair_null_images        = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "null" in f  and "flair" in f]
    mask_null_images         = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "null" in f  and "mask" in f] 

    
    #create csb fluid_grey_matter
    
    csb_fluid_new_name          = f"segment_csb_fluid.nii.gz"
    white_matter_new_name       = f"segment_white_matter.gz"
    grey_matter_new_name        = f"segment_grey_matter.nii.gz"

    
    csb_fluid_new_path                 = os.path.join(subject_null_image_path,csb_fluid_new_name)
    white_matter_new_path              = os.path.join(subject_null_image_path,white_matter_new_name)
    grey_matter_new_name               = os.path.join(subject_null_image_path,grey_matter_new_name)

    
    output_segmentation      = subprocess.run(["fast","-t","2",
                                                flair_image], 
                                                capture_output=True)
    
    
    csb_fluid_old_path            = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "pve_2" in f][0]
    white_matter_old_path         = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "pve_1" in f][0]
    grey_matter_old_path          = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "pve_0" in f][0]
    
    os.system(f"mv {csb_fluid_old_path} {csb_fluid_new_path}")
    os.system(f"mv {white_matter_old_path} {white_matter_new_path}")
    os.system(f"mv {grey_matter_old_path} {grey_matter_new_name}")
    
        
    mixeltype_old_path          = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "mixeltype" in f][0]
    seg_old_path                = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "seg" in f][0]
    os.system(f"rm  {mixeltype_old_path}")
    os.system(f"rm  {seg_old_path}")
    

    
    
