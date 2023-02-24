#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:43:27 2023

@author: temuuleu
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.path_utils import create_result_dir,create_dir,get_main_paths,get_all_dirs
from modules.paths_library import create_challange_table,create_belove_table
from modules.paths_library import create_tabel_for_all_subjects

from nipype.interfaces.ants import N4BiasFieldCorrection

from nipype import Node, Workflow
from nipype.interfaces.fsl import Reorient2Std
from nipype.interfaces.fsl import SliceTimer, MCFLIRT, Smooth, BET,ApplyXFM,ConvertXFM

from nipype.interfaces.fsl import ApplyXFM
from nipype.interfaces.ants.segmentation import BrainExtraction

from os.path import join
from subprocess import run,call   
import subprocess 
from os import listdir

import skimage.transform as skTrans
import nibabel as nib


def normalize_volume(image):
    """normalize 3D Volume image"""
    
    
    dim     = image.shape
    volume = image.reshape(-1, 1)
    
    rvolume = np.divide(volume - volume.min(axis=0), (volume.max(axis=0) - volume.min(axis=0)))*1
    rvolume = rvolume.reshape(dim)
    
    return rvolume



config_dict =get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]
datasets_list                         =     get_all_dirs(datasets_original_path)



Set_1_path                            =     os.path.join(datasets_prepared_path,"Set_1")

data_location_table_path              =     os.path.join(Set_1_path,"data_location.xlsx")
data_pp_path                          =     os.path.join(Set_1_path,"data_prepared_data.xlsx")

data_location_table_df                =     pd.read_excel(data_location_table_path,index_col=None)

subject_list                          =     list(data_location_table_df.loc[:,"subject"])


if os.path.isfile(data_pp_path):
    data_set_df                           =     pd.read_excel(data_pp_path,index_col=None)
else:
    data_set_df                           =     pd.DataFrame()


for subject in subject_list[:]:
    
    print(subject)
    
    single_patient_data = data_location_table_df.loc[data_location_table_df['subject'] == subject,:]
    
    
    single_patient_data.columns
