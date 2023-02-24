#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:46:11 2023

@author: temuuleu
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.path_utils import create_result_dir,create_dir,get_main_paths,get_all_dirs
from modules.paths_library import create_challange_table,create_belove_table
from modules.paths_library import create_tabel_for_all_subjects,save_df

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
from modules.mri_utils import normalize_volume,get_volume

config_dict                           =     get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]
datasets_list                         =     get_all_dirs(datasets_original_path)


Set_1_path                            =     os.path.join(datasets_prepared_path,"Set_1")
data_location_table_path              =     os.path.join(datasets_prepared_path,"data_original_location.xlsx")
data_pp_path                          =     os.path.join(Set_1_path,"data_prepared_data.xlsx")

data_pps_path                          =     os.path.join(Set_1_path,"data_prepared_data_segmented.xlsx")
data_location_table_df                =     pd.read_excel(data_location_table_path,index_col=None)
subject_list                          =     list(data_location_table_df.loc[:,"subject"])

if os.path.isfile(data_pp_path):
    data_set_df                           =     pd.read_excel(data_pp_path,index_col=0)
else:
    data_set_df                           =     pd.DataFrame()


bool_new = False

for subject in subject_list[:]:
    
    print(subject)

    single_patient_data = data_set_df.loc[data_set_df['subject'] == subject,:]

    patient_id                  =  single_patient_data["patient_id"].values[0]
    dataset_name                =  single_patient_data["dataset_name"].values[0]
    lesion_concentration        =  single_patient_data["lesion_concentration"].values[0]
    
    
    #single_patient_data.columns
    biascorrected_bet_image_path = single_patient_data['biascorrected_bet_image_path'].values[0]
    biascorrected_normalized_image_path = single_patient_data['biascorrected_normalized_image_path'].values[0]

    prep_subject_dir                            =   os.path.join(Set_1_path,subject)

    #create csb fluid_grey_matter
    
    csb_fluid_new_name          = f"segment_csb_fluid.nii.gz"
    white_matter_new_name       = f"segment_white_matter.gz"
    grey_matter_new_name        = f"segment_grey_matter.nii.gz"
    
    
    csb_fluid_new_name_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if csb_fluid_new_name in l] 
    white_matter_new_name_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if white_matter_new_name in l] 
    grey_matter_new_name_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if grey_matter_new_name in l] 
    
    
    if csb_fluid_new_name_paths and white_matter_new_name_paths and grey_matter_new_name_paths and not bool_new:
         
        csb_fluid_new_path    = csb_fluid_new_name_paths[0]
        white_matter_new_path = white_matter_new_name_paths[0]
        grey_matter_new_path  = grey_matter_new_name_paths[0]
        
        print(f"found csb_fluid_new_name_paths  {csb_fluid_new_path}")
        
    else:

        print(f"didn't segmentations create new ones")
        csb_fluid_new_path                 = os.path.join(prep_subject_dir,csb_fluid_new_name)
        white_matter_new_path              = os.path.join(prep_subject_dir,white_matter_new_name)
        grey_matter_new_path               = os.path.join(prep_subject_dir,grey_matter_new_name)
        
   
        output_segmentation      = subprocess.run(["fast","-t","2",
                                                    biascorrected_bet_image_path], 
                                                    capture_output=True)
        
        
        csb_fluid_old_path            = [join(prep_subject_dir,f) for f in os.listdir(prep_subject_dir) if "pve_2" in f][0]
        white_matter_old_path         = [join(prep_subject_dir,f) for f in os.listdir(prep_subject_dir) if "pve_1" in f][0]
        grey_matter_old_path          = [join(prep_subject_dir,f) for f in os.listdir(prep_subject_dir) if "pve_0" in f][0]
        
        os.system(f"mv {csb_fluid_old_path} {csb_fluid_new_path}")
        os.system(f"mv {white_matter_old_path} {white_matter_new_path}")
        os.system(f"mv {grey_matter_old_path} {grey_matter_new_path}")
        
            
        mixeltype_old_path          = [join(prep_subject_dir,f) for f in os.listdir(prep_subject_dir) if "mixeltype" in f][0]
        seg_old_path                = [join(prep_subject_dir,f) for f in os.listdir(prep_subject_dir) if "seg" in f][0]
        os.system(f"rm  {mixeltype_old_path}")
        os.system(f"rm  {seg_old_path}")
        

    csb_fluid_new_name_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if csb_fluid_new_name in l] 
    white_matter_new_name_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if white_matter_new_name in l] 
    grey_matter_new_name_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if grey_matter_new_name in l] 
    


    data_set_df.loc[data_set_df['subject'] == subject,["csb_fluid_new_name_path"]] = csb_fluid_new_path
    data_set_df.loc[data_set_df['subject'] == subject,["white_matter_new_name_path"]] = white_matter_new_path
    data_set_df.loc[data_set_df['subject'] == subject,["grey_matter_new_name_path"]] = grey_matter_new_path
    

    save_df(data_set_df,data_pps_path)
    
    
    
    
        
        
        
        
        

        # biascorrected_bet_3d                                  = normalize_volume(nib.load(biascorrected_bet_image_path).get_fdata()  )  
        # biascorrected_normalized_3d                           = normalize_volume(nib.load(biascorrected_normalized_image_path).get_fdata())   
        
        # mask_3d                                               = nib.load(mask_path).get_fdata()    
        # mask_normalized_3d                                    = nib.load(mask_normalized_path).get_fdata()  
        





        # small_image_counts              = biascorrected_bet_3d.shape[2]
        # image_counts                   = biascorrected_normalized_3d.shape[2]
        # image_counts_list = [r for r in range(image_counts)]

        # full =  image_counts_list[-1]
        
        # part = round(full / small_image_counts)subject_list[:]
        
    
        # for r in image_counts_list:
        #     u =round(r/part)-1

        #     if u <=0:
        #         u = 0
        #     print(u)

        #     flair_normal_image_2d                = np.zeros(biascorrected_normalized_3d[:,:,r].shape+(3,))
        #     flair_image_2d                       = np.zeros(biascorrected_bet_3d[:,:,u].shape+(3,))
            
            
        #     flair_2d_normal_con                  = flair_normal_image_2d.copy()
        #     flair_2d_con                         = flair_image_2d.copy()
            
        #     if  np.sum(mask_3d[:,:,u]) <= 10: continue
        
        #     print(np.sum(mask_3d[:,:,u]))

        #     flair_2d_normal_con[:,:,0]   =   biascorrected_normalized_3d[:,:,r]
        #     flair_2d_normal_con[:,:,1]   =   biascorrected_normalized_3d[:,:,r]
        #     flair_2d_normal_con[:,:,2]   =   mask_normalized_3d[:,:,r]
            
        #     flair_2d_con[:,:,0]          =   biascorrected_bet_3d[:,:,u]
        #     flair_2d_con[:,:,1]          =   biascorrected_bet_3d[:,:,u]
        #     flair_2d_con[:,:,2]          =   mask_3d[:,:,u]
            
        #     sub_title = f"patient"
  
        #     fig, axes =  plt.subplots(1,2,constrained_layout = True)
        #     fig.suptitle(sub_title, fontsize=40)  
        #     fig.set_figheight(15)
        #     fig.set_figwidth(30)
        #     axes[0].imshow(flair_2d_normal_con)
        #     axes[1].imshow(flair_2d_con)
            

        #     jpg_file = prep_subject_dir+"/"+str(u)+"_"+str(r)+".jpg"
            
        #     plt.savefig(jpg_file, 
        #              dpi=150,
        #              bbox_inches ="tight",
        #              pad_inches=1)
            
        #     #plt.show()
        #     plt.clf() 
            

            

            
            # for s in range(axes.shape[0]):
            #     for a in range(axes.shape[1]):
            #         axes[s][a].get_xaxis().set_visible(False)
            #         axes[s][a].get_yaxis().set_visible(False)
                    
                
            #axes.set_title(f'Flair', fontsize=25)          
            # axes.imshow(flair_2d)
            
            # axes[0][1].set_title(f'Normal', fontsize=25)     
            # axes[0][1].imshow(flair_normal_2d)
            

            

            
            
            