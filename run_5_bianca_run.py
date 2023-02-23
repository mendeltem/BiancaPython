#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 19:33:16 2023

@author: temuuleu
"""

import os
import numpy as np
import pandas as pd
from modules.path_utils import create_dir,get_main_paths,get_all_dirs
from modules.paths_library import create_null_images,create_master_file
from os.path import join
from sklearn.model_selection import train_test_split

import subprocess


def start_bianca(train_subjects,
                 data_set_df,
                 bianca_result_subject_path,
                 test_flair_path,
                 test_flair_to_mni,
                 test_mask_path,
                 output_mask_path
                 ):


    master_file_path,trainstring,row_number  =  create_master_file(train_subjects,
                           data_set_df,
                           bianca_result_subject_path,
                           test_flair_path,
                           test_flair_to_mni,
                           test_mask_path
                           )


    output_mask_path                        =    run_bianca(master_file_path,  
                                                            output_mask_path,
                                                            trainstring,row_number)

def run_bianca(master_file_path,
               output_mask_path,
               trainstring,
               row_number,
                brainmaskfeaturenum       = 1,
                spatialweight             = 2,
                labelfeaturenum           = 3,
                trainingpts               = 10000,
                selectpts                 = "noborder",
                nonlespts                 = 10000,
               ):

    train_bianca_commands                   = "bianca --singlefile="+master_file_path+\
                                              " --labelfeaturenum="+str(labelfeaturenum)+\
                                              " --brainmaskfeaturenum="+str(brainmaskfeaturenum)+\
                                              " --matfeaturenum="+str(spatialweight)+\
                                              " --trainingpts="+str(trainingpts)+\
                                              " --querysubjectnum="+str(row_number)+" --nonlespts="+str(nonlespts)+\
                                              " --trainingnums="+trainstring+" -o "+output_mask_path+" -v"  
    print("train_bianca_commands:")                                         
    print(train_bianca_commands)
    print("")
                                              # use subprocess to run the command

    try:
        os.system(train_bianca_commands)
        
        #subprocess.Popen(train_bianca_commands, shell=True)
    except:
        print("bianca command failed")
        
    return output_mask_path



config_dict =get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]


data_pp_path                          =     os.path.join(datasets_prepared_path,"data_prepared_data.xlsx")
data_set_df                           =     pd.read_excel(data_pp_path,index_col=None)


nulled_images_path                    =     os.path.join(datasets_prepared_path,"nulled_images")



subject_list                          =     list(data_set_df.loc[:,"subject"])


seeds  = 10
seeds_list = [seed for seed in range(1,seeds)]


for seed in seeds_list:
    
    bianca_result_path                    =     os.path.join(datasets_prepared_path,f"bianca_result_seed_{seed}")


    # Split the subjects into a training set and a test set
    train_subjects, test_subjects = train_test_split(subject_list, test_size=0.2, random_state=seed)
    
    
    for test_subject in test_subjects:
        
        test_patient_df                                = data_set_df.loc[data_set_df['subject'] == test_subject,:]
    
        test_flair_path    = test_patient_df['biascorrected_bet_image_path'].values[0]
        test_flair_to_mni  = test_patient_df['MNI_xfm_path'].values[0]
        
        test_mask_path_n                                =  os.path.join(os.path.dirname(test_flair_path),"WMHmask.nii")
        test_mask_path_g                                =  os.path.join(os.path.dirname(test_flair_path),"WMHmask.nii.gz")
        
        if os.path.isfile(test_mask_path_n):
            test_mask_path = test_mask_path_n
        
        elif os.path.isfile(test_mask_path_g):
            test_mask_path = test_mask_path_g
            
            
        null_subject_path = os.path.join(nulled_images_path, test_subject)
    
        flair_null_images = [join(null_subject_path,f) for f in os.listdir(null_subject_path) if "null" in f  and "flair" in f]
        mask_null_images  = [join(null_subject_path,f) for f in os.listdir(null_subject_path) if "null" in f  and "mask" in f] 
        
        
        bianca_result_subject_path = os.path.join(bianca_result_path, test_subject)
        create_dir(bianca_result_subject_path)
            
        output_mask                             = f"seg_prob_bianca.nii.gz"
        output_mask_path                        = join(bianca_result_subject_path,output_mask)
        
        
        start_bianca(train_subjects,
                         data_set_df,
                         bianca_result_subject_path,
                         test_flair_path,
                         test_flair_to_mni,
                         test_mask_path,
                         output_mask_path
                         )
    
    
        for i, (flair_null_image,mask_null_images) in enumerate(zip(flair_null_images,mask_null_images)):
            
            print(i)
            
            output_mask                             = f"seg_prob_bianca_{i}_null.nii.gz"
            output_mask_path                        = join(bianca_result_subject_path,output_mask)
            
            
            start_bianca(train_subjects,
                             data_set_df,
                             bianca_result_subject_path,
                             flair_null_image,
                             test_flair_to_mni,
                             mask_null_images,
                             output_mask_path
                             )
    
            
        
    
