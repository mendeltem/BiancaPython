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

from os.path import join
from sklearn.model_selection import train_test_split

import subprocess


import os
import numpy as np
import pandas as pd
from modules.path_utils import create_dir,get_main_paths,get_all_dirs
from modules.paths_library import save_df


from modules.bianca_library import create_training_master_file,create_test_subjects_for_bianca_table
from modules.bianca_library import MasterCreator

from os.path import join
from sklearn.model_selection import train_test_split

import subprocess



def create_masterfile(master_file_path,
                                biascorrected_bet_image_path,
                                MNI_xfm_path,
                                mask_path
                                ):

    dirname = os.path.dirname(master_file_path)

    with open(master_file_path, 'r') as file:
        master_file_txt = file.read()
        
        
    row_number =master_file_txt.count("flair_to_mni.mat")+1

    trainstring = "".join([str(r)+"," for r in range(1,row_number)])[:-1]
            
    master_file_test=    biascorrected_bet_image_path+" " +MNI_xfm_path+" "+mask_path


    all_master_file_txt  = master_file_txt + "\n"+master_file_test+"\n"

    new_master_file_path = join(dirname,"master_file.txt")
    
    
    with open(new_master_file_path, 'w') as f:
        f.write(all_master_file_txt)
        
    return    new_master_file_path,trainstring,row_number
        


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


Set_1_path                            =     os.path.join(datasets_prepared_path,"Set_1")
data_pp_path                          =     os.path.join(Set_1_path,"data_prepared_data.xlsx")

data_set_df                           =     pd.read_excel(data_pp_path,index_col=0)

bianca_master_files_directory         =     os.path.join(datasets_prepared_path,"bianca_master_files")



bianca_master_files_directory_list    = get_all_dirs(bianca_master_files_directory)



for directory in bianca_master_files_directory_list[:]:
    
    print(directory)
    
    master_file_path         = [ join(directory,l) for l in os.listdir(directory)  if "master_file.txt" in l] [0]
    master_file_df_path      = [ join(directory,l) for l in os.listdir(directory)  if "master_file.xlsx" in l] [0]
    test_subjects_df_path    = [ join(directory,l) for l in os.listdir(directory)  if "test_subjects.xlsx" in l] [0]
    
    test_subjects_df         =  pd.read_excel(test_subjects_df_path, index_col=0)
    
    
    subject_list                          =     list(test_subjects_df.loc[:,"subject"])
    
    
    for subject in subject_list[:]:
        
        print(subject)
        
        single_patient_data = test_subjects_df.loc[test_subjects_df['subject'] == subject,:]
    

        patient_id                  =  single_patient_data["patient_id"].values[0]
        dataset_name                =  single_patient_data["dataset_name"].values[0]
        lesion_concentration        =  single_patient_data["lesion_concentration"].values[0]
        
        biascorrected_bet_image_path  =  single_patient_data["biascorrected_bet_image_path"].values[0]
        MNI_xfm_path                  =  single_patient_data["MNI_xfm_path"].values[0]
        mask_path                     =  single_patient_data["mask_path"].values[0]

        new_master_file_path,trainstring,row_number= create_masterfile(master_file_path,
                                                                       biascorrected_bet_image_path,
                                                                       MNI_xfm_path,
                                                                       mask_path
                                                                       )
        
        subject_path   = os.path.join(directory,subject)
        
        create_dir(subject_path)
        
        
        output_mask                             = f"seg_prob_bianca.nii.gz"
        output_mask_path                        = join(subject_path,output_mask)
        
        
        output_mask_path                        =    run_bianca(new_master_file_path,  
                                                                output_mask_path,
                                                                trainstring,row_number)






        
    
    
