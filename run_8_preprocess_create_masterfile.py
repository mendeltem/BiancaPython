#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:02:16 2023

@author: temuuleu
"""


import os
import numpy as np
import pandas as pd
from modules.path_utils import create_dir,get_main_paths,get_all_dirs
from modules.paths_library import create_master_file,save_df


from modules.bianca_library import create_training_master_file,create_test_subjects_for_bianca_table
from modules.bianca_library import MasterCreator

from os.path import join
from sklearn.model_selection import train_test_split

import subprocess


config_dict =get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]


Set_1_path                            =     os.path.join(datasets_prepared_path,"Set_1")
data_pp_path                          =     os.path.join(Set_1_path,"data_prepared_data.xlsx")

data_set_df                           =     pd.read_excel(data_pp_path,index_col=0)

bianca_master_files_directory         =     os.path.join(datasets_prepared_path,"bianca_master_files")

#all subjects
subject_list                          =     list(data_set_df.loc[:,"subject"])

MasterCreator_1 = MasterCreator(bianca_master_files_directory, data_set_df)


seeds=10

for seed in range(1,seeds):

    #randomly select
    train_subjects,  test_subjects              = train_test_split(subject_list, random_state=seed)
    
    
    master_file_name = f"bianca_train_random_{seed}"
    MasterCreator_1.create_master_directory(train_subjects,test_subjects,master_file_name)


#only low
middle_low_subjects = list(data_set_df.loc[(data_set_df["lesion_concentration"] == 'middle') | (data_set_df["lesion_concentration"] == 'low'),"subject"])
high_subjects = list(data_set_df.loc[ (data_set_df["lesion_concentration"] == 'high'),"subject"])


train_middle_low_subjects,  test_middle_low_subjects  = train_test_split(middle_low_subjects, random_state=seed)

# train_high_subjects,  test_high_subjects  = train_test_split(high_subjects, random_state=seed)
train_subjects = train_middle_low_subjects
test_subjects = test_middle_low_subjects+high_subjects


master_file_name = f"bianca_train_middle_low"
MasterCreator_1.create_master_directory(train_subjects,test_subjects,master_file_name)


#only Belove
belove_subjects = list(data_set_df.loc[(data_set_df["dataset_name"] == "Belove_Kersten") ,"subject"])
train_subjects,  belove_test_subjects              = train_test_split(belove_subjects, random_state=seed)

other_subjects = list(data_set_df.loc[(data_set_df["dataset_name"] != "Belove_Kersten") ,"subject"])

test_subjects  = other_subjects +belove_test_subjects
master_file_name = f"bianca_train_belove"
MasterCreator_1.create_master_directory(train_subjects,test_subjects,master_file_name)



#only Challenge_set
challenge_set_subjects = list(data_set_df.loc[(data_set_df["dataset_name"] == "Challenge_set") ,"subject"])
train_subjects,  challenge_test_subjects              = train_test_split(challenge_set_subjects, random_state=seed)

test_subjects  = challenge_test_subjects +belove_subjects
master_file_name = f"bianca_train_challange"
MasterCreator_1.create_master_directory(train_subjects,test_subjects,master_file_name)



