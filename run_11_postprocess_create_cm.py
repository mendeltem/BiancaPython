#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 19:58:29 2023

@author: temuuleu
"""

import os
import numpy as np
import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
import subprocess

from modules.bianca_library import create_training_master_file,create_test_subjects_for_bianca_table
from modules.bianca_library import MasterCreator
from modules.path_utils import create_dir,get_main_paths,get_all_dirs
from modules.paths_library import save_df

import time

import sys
import os 
from os import listdir
from os.path import join, basename
#from nipype import Node, Workflow
#from nipype.interfaces.fsl import Reorient2Std
#from nipype.interfaces.fsl import SliceTimer, MCFLIRT, Smooth, BET,ApplyXFM,ConvertXFM
import pandas as pd
from subprocess import run,call    
import nibabel as nib
import numpy as np
import skimage.transform as skTrans
# import cv2 
from itertools import combinations
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


def get_volume(path):
    
    output_all_matter = subprocess.run(["fslstats",
                    path,
                    "-V"], 
                    capture_output=True)
            
    result_str = str(output_all_matter.stdout)
    all_matter_result = result_str.split(" ")

    return float(all_matter_result[1])


def make_threshholde(output_mask_path , post_processed_path,th):
    
    
    thresholds_path  = os.path.join(post_processed_path,"thresholds")
    
    create_dir(thresholds_path)
    
    mask_d                                  = nib.load(output_mask_path)
    output_mask_array                       = mask_d.get_fdata()
    
    #np.unique(output_mask)

    output_mask_copy = np.copy(output_mask_array)
    threshhold_float = th / 100

    output_mask_copy[output_mask_copy >=threshhold_float] = 1
    output_mask_copy[output_mask_copy <threshhold_float] = 0

    # #th_str = str(th).split(".")[1]
    # output_mask = np.copy(output_mask_original)
    # output_mask[output_mask >=th] = 1
    # output_mask[output_mask <th] = 0
    
    th_str          =  str(th)
    th_name         =  f"bianca_output_thr_{th_str}.nii.gz"

    ni_img          =  nib.Nifti1Image(output_mask_copy, mask_d.affine)
    th_path         =  join(thresholds_path,th_name)
    
    nib.save(ni_img, th_path)
    

    return output_mask_copy,th_path


def create_threshold(output_mask_path,th):

    #create one threshhold
    mask_d                                  = nib.load(output_mask_path)
    output_mask_array                       = mask_d.get_fdata()
    
    #np.unique(output_mask)
    output_mask_bin_segmentation_array      = make_threshholde(output_mask_array ,th)
    
    #np.unique(output_mask_bin_segmentation)
    
    th_str          = str(th)
    th_name         =  f"bianca_output_thr_{th_str}.nii.gz"

    ni_img          =  nib.Nifti1Image(output_mask_bin_segmentation_array, mask_d.affine)
    th_path         =  join(os.path.dirname(output_mask_path),th_name)
    
    nib.save(ni_img, th_path)
    
    return th_path


def create_wm_distances(bianca_normal_cluster_path, csb_fluid_new_name_path ,th,minclustersize,distance_to_ventrickle   = 10.0001):
    
    
    normal_image_dir = os.path.dirname(bianca_normal_cluster_path)

    bianca_out_bin_distvent_normal_cluster_path = f"{normal_image_dir}/bianca_out_bin_distvent_{minclustersize}_th{th}.nii.gz"
    
    os.system(f"distancemap -i {csb_fluid_new_name_path} -m {bianca_normal_cluster_path} -o {bianca_out_bin_distvent_normal_cluster_path}")
    
    
    deepwm_map_normal_path   = f"{normal_image_dir}/deepwm_map_{round(distance_to_ventrickle)}mm_th{th}.nii.gz"
    perivent_map_normal_path = f"{normal_image_dir}/perivent_map_{round(distance_to_ventrickle)}mm_th{th}.nii.gz" 
    
    os.system(f"fslmaths {bianca_out_bin_distvent_normal_cluster_path} -thr {distance_to_ventrickle} -bin {deepwm_map_normal_path}")
    os.system(f"fslmaths {bianca_normal_cluster_path} -sub {deepwm_map_normal_path} {perivent_map_normal_path}")
    

    return deepwm_map_normal_path,perivent_map_normal_path


def create_analyse_bianca_results(bianca_normal_prob,
                                  flair_csb_new_fluidname_path,
                                  th = 99,
                                  distance_to_ventrickle = 10):

    thresh_normal_path          =  create_threshold(bianca_normal_prob,th)
    
    bianca_normal_cluster_path  =  perform_cluster_analysis(thresh_normal_path)
    
    deepwm_map_normal_path,perivent_map_normal_path  =    create_wm_distances(bianca_normal_cluster_path,
                                                          flair_csb_new_fluidname_path ,th,
                                                          distance_to_ventrickle   = distance_to_ventrickle)
    
    
    
    
    return deepwm_map_normal_path,perivent_map_normal_path,bianca_normal_cluster_path 

                
def perform_cluster_analysis(thresh_path, minclustersize = 150):
    
    normal_image_dir = os.path.dirname(thresh_path)
    
    # Generate output path for bianca output
    bianca_normal_cluster_path = f"{normal_image_dir}/bianca_out_mincluster_{minclustersize}.nii.gz"
    
    # Perform cluster analysis on thresholded image
    os.system(f"cluster --in={thresh_path} --thresh=0.05 \
    --oindex={bianca_normal_cluster_path} --connectivity=6 --no_table --minextent={minclustersize}")
    
    # Convert output to binary mask
    os.system(f"fslmaths {bianca_normal_cluster_path} -bin {bianca_normal_cluster_path}")
    
    return bianca_normal_cluster_path


def create_confusion_matrix(bianca_normal_cluster_path,mask_image,confusion_processed):
    
    confusion_matrix = {}

    print(f"minimum_clustersize :  {minimum_clustersize} ")
    
    th = os.path.basename(bianca_normal_cluster_path).split(".")[0].split("_")[-2]
    #print(th)
    
    
    minclustersize = os.path.basename(bianca_normal_cluster_path).split(".")[0].split("_")[-1]
    #print(minclustersize)
    
    
    bianca_out_nib                      = nib.load(bianca_normal_cluster_path)
    bianca_out_ms                       = bianca_out_nib.get_fdata()  

    ground_truth                        = nib.load(mask_image).get_fdata()  
   
    tp = ((bianca_out_ms == 1) & (ground_truth == 1)).astype(np.float64) 
    tn = ((bianca_out_ms == 0) & (ground_truth == 0)).astype(np.float64) 
    fp = ((bianca_out_ms == 1) & (ground_truth == 0)).astype(np.float64) 
    fn = ((bianca_out_ms == 0) & (ground_truth == 1)).astype(np.float64) 
    
    tp_path = os.path.join(confusion_processed,f"tp_{th}_{minclustersize}.nii")
    fp_path = os.path.join(confusion_processed,f"fp_{th}_{minclustersize}.nii")
    
    fn_path = os.path.join(confusion_processed,f"fn_{th}_{minclustersize}.nii")
         
    tp_nib = nib.Nifti1Image(tp, bianca_out_nib.affine)
    nib.save(tp_nib, tp_path)

    fp_nib = nib.Nifti1Image(fp, bianca_out_nib.affine)
    nib.save(fp_nib, fp_path)

    fn_nib = nib.Nifti1Image(fn, bianca_out_nib.affine)
    nib.save(fn_nib, fn_path)

    tp_value = np.sum(tp)
    tn_value = np.sum(tn)
    
    fp_value = np.sum(fp)
    fn_value = np.sum(fn)
    
    dice_score_cl     = round((2*(tp_value)) / (2*(tp_value)+(fp_value)+(fn_value)) ,2)
    
    print(f"dice_score_cl {dice_score_cl}")

    sensitivity_cl    = round( (tp_value) / ((tp_value) + (fn_value)),2)
    
    print(f"dice_score_cl {dice_score_cl}")
    print(f"sensitivity_cl {sensitivity_cl}")
    
    confusion_matrix["tp"] = {}
    confusion_matrix["tp"]["value"] = tp_value
    confusion_matrix["tp"]["path"] = tp_path
    
    confusion_matrix["fp"] = {}
    confusion_matrix["fp"]["value"] = fp_value
    confusion_matrix["fp"]["path"]  = fp_path

                    
    confusion_matrix["fn"] = {}
    confusion_matrix["fn"]["value"] = fn_value
    confusion_matrix["fn"]["path"]  = fn_nib
    
    return dice_score_cl,confusion_matrix


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



Set_1_path                            =     os.path.join(datasets_prepared_path,"Set_1")
data_pp_path                          =     os.path.join(Set_1_path,"data_prepared_data_segmented.xlsx")
data_set_df                           =     pd.read_excel(data_pp_path,index_col=0)



thresholds_list          = [97,99]

minimum_clustersizes     = [100,150]

distance_to_ventrickles = [10]


for directory in bianca_master_files_directory_list[:]:
    
    print(directory)
    
    subject_results_dir_list    = get_all_dirs(directory)
    
    analyse_result_excel_path = os.path.join( directory,"result.xlsx")
    
    result_paths_path         = os.path.join( directory,"result_pth.xlsx")
    
    df_result       = pd.DataFrame()
    df_paths_result = pd.DataFrame()
    
    for subject_results_dir in subject_results_dir_list:
        

        print(subject_results_dir)
        
        
        subject  = os.path.basename(subject_results_dir)
        
        
        single_patient_data = data_set_df.loc[data_set_df['subject'] == subject,:]

        patient_id                  =  single_patient_data["patient_id"].values[0]
        dataset_name                =  single_patient_data["dataset_name"].values[0]
        lesion_concentration        =  single_patient_data["lesion_concentration"].values[0]
        
        
        #single_patient_data.columns
        biascorrected_bet_image_path = single_patient_data['biascorrected_bet_image_path'].values[0]
        biascorrected_normalized_image_path = single_patient_data['biascorrected_normalized_image_path'].values[0]
        
        total_brain_volume             =     get_volume(biascorrected_bet_image_path)
        
        
        csb_fluid_new_name_path        =  single_patient_data["csb_fluid_new_name_path"].values[0]

        mask_path                      =  single_patient_data["mask_path"].values[0]
        
        
        seg_prob_bianca_paths = [ join(subject_results_dir,l) for l in os.listdir(subject_results_dir)  if "prob" in l] 
            

        prob_found = 0
        
        while not prob_found:
            try:
                bianca_normal_prob  = seg_prob_bianca_paths[0]
                prob_found = 1
                
            except:
                print("File not found. Waiting...")
                time.sleep(1)  # wait for 1 second before trying again
                
        
        confusion_processed      = os.path.join(subject_results_dir,"confusion")
        
        
        for threshold in thresholds_list:
            
            output_mask_copy,th_path = make_threshholde(bianca_normal_prob , confusion_processed,threshold)
            
            print(f"threshold :  {threshold} ")
            
            for minimum_clustersize in minimum_clustersizes:
                
                bianca_normal_cluster_path  =  perform_cluster_analysis(th_path,minimum_clustersize)
                
        
                dice_score_cl,confusion_matrix=create_confusion_matrix(bianca_normal_cluster_path,mask_path,confusion_processed)

                for distance_to_ventrickle in distance_to_ventrickles:
                    
                    
                    print(f"distance_to_ventrickle :  {distance_to_ventrickle} ")

                    deepwm_map_normal_path,perivent_map_normal_path  =    create_wm_distances(bianca_normal_cluster_path,
                                                                          csb_fluid_new_name_path ,
                                                                          threshold,
                                                                          minimum_clustersize,
                                                                          distance_to_ventrickle   = distance_to_ventrickle)
                        
                    deepwm_map_normal_volume               =     get_volume(deepwm_map_normal_path)
                    perivent_map_normal_volume             =     get_volume(perivent_map_normal_path)
                    
                
                    total_lesion_volume                    =     get_volume(bianca_normal_cluster_path) 
                    pvent_vol_10mm_volume                  =     get_volume(perivent_map_normal_path)  
                    deep_vol_10mm_volume                   =     get_volume(deepwm_map_normal_path)   

                    #dokumentation
                    df_result_temp = pd.DataFrame(index=(0,))
    
                    df_result_temp["patient_id"]            = patient_id
                    df_result_temp["dataset_name"]          = dataset_name
                    df_result_temp["subject_name"]          = subject
                    df_result_temp["threshold"]             = threshold
                    df_result_temp["minimum_clustersize"]   = minimum_clustersize
                    df_result_temp["lesion_concentration"]               = lesion_concentration
                    
                    df_result_temp["dice_score"]            = dice_score_cl
                    df_result_temp["distance_to_ventrickle"]= distance_to_ventrickle
                    
                    df_result_temp["total_lesion_volume"]   = np.round(total_lesion_volume /total_brain_volume,7)
                    df_result_temp["pvent_vol_10mm_volume"] = np.round(pvent_vol_10mm_volume/total_brain_volume,7)
                    df_result_temp["deep_vol_10mm_volume"]  = np.round(deep_vol_10mm_volume/total_brain_volume,7)
     
                    df_result_temp["brain_volume"]  = total_brain_volume
                    
                    
                    df_result_temp["tp"]  = confusion_matrix['tp']['value']
                    df_result_temp["fp"]  = confusion_matrix['fp']['value']
                    df_result_temp["fn"]  = confusion_matrix['fn']['value']
                    
                    df_result = pd.concat([df_result,df_result_temp]) 
                    
                    
                    try:
                        df_result.to_excel(analyse_result_excel_path)
                    except:
                        print("error saving")
            
                
                    
                    df_result_path_temp = pd.DataFrame(index=(0,))
                    df_result_path_temp["patient_id"]            = patient_id
                    df_result_path_temp["dataset_name"]          = dataset_name
                    df_result_path_temp["subject"]               = subject
                    df_result_path_temp["threshold"]             = threshold
                    df_result_path_temp["minimum_clustersize"]   = minimum_clustersize
                    
                    df_result_temp["lesion_concentration"]       = lesion_concentration
                    df_result_temp["dice_score"]                 = dice_score_cl
                    
                    df_result_path_temp["th_path"]               = th_path
                    
                    df_result_path_temp["bianca_normal_cluster_path"] = bianca_normal_cluster_path
                    df_result_path_temp["perivent_map_normal_path"]   = perivent_map_normal_path
                    df_result_path_temp["deepwm_map_normal_path"]     = deepwm_map_normal_path
                    
                    df_result_path_temp["mask_path"]       = mask_path
                    df_result_path_temp["flair_path"]     = biascorrected_bet_image_path
                        
             
                    df_result_path_temp["tp"]  = confusion_matrix['tp']['path']
                    df_result_path_temp["fp"]  = confusion_matrix['fp']['path']
                    df_result_path_temp["fn"]  = confusion_matrix['fn']['path']
                    
                    df_paths_result = pd.concat([df_paths_result,df_result_path_temp]) 
                    df_paths_result.to_excel(result_paths_path)
                    
                    try:
                        df_paths_result.to_excel(result_paths_path)
                    except:
                        print("error saving")
                    