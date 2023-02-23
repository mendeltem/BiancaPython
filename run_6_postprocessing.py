#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:40:30 2023

@author: temuuleu
"""

import os
import numpy as np
import pandas as pd

from os.path import join
from sklearn.model_selection import train_test_split

import subprocess
from subprocess import run,call  

import nibabel as nib

os.chdir("/home/temuuleu/CSB_NeuroRad/temuuleu/Projekts/BELOVE/Bianca_Project/bianca_version_3/")

from modules.path_utils import create_dir,get_main_paths,get_all_dirs
from modules.paths_library import create_null_images,create_master_file







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

def get_volume(path):
    
    output_all_matter = run(["fslstats",
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





def create_wm_distances(bianca_normal_cluster_path, flair_csb_new_fluidname_path ,distance_to_ventrickle   = 10.0001):
    
    
    normal_image_dir = os.path.dirname(bianca_normal_cluster_path)
    
        
    th = os.path.basename(bianca_normal_cluster_path).split(".")[0].split("_")[-2]
    #print(th)
    
    
    minclustersize = os.path.basename(bianca_normal_cluster_path).split(".")[0].split("_")[-1]
    #print(minclustersize)
    
    
    dist_v = round(distance_to_ventrickle)

    # os.path.isfile(bianca_normal_cluster_path)
    
    # os.path.isfile(flair_csb_new_fluidname_path)
    
    bianca_out_bin_distvent_normal_cluster_path = f"{normal_image_dir}/bianca_out_bin_distvent_{minclustersize}_th_{th}.nii.gz"
    
    os.system(f"distancemap -i {flair_csb_new_fluidname_path} -m {bianca_normal_cluster_path} -o {bianca_out_bin_distvent_normal_cluster_path}")
    
    
    deepwm_map_normal_path   = f"{normal_image_dir}/deepwm_map_{dist_v}mm_th{th}.nii.gz"
    perivent_map_normal_path = f"{normal_image_dir}/perivent_map_{dist_v}mm_th{th}.nii.gz" 
    
    os.system(f"fslmaths {bianca_out_bin_distvent_normal_cluster_path} -thr {distance_to_ventrickle} -bin {deepwm_map_normal_path}")
    os.system(f"fslmaths {bianca_normal_cluster_path} -sub {deepwm_map_normal_path} {perivent_map_normal_path}")
    

    return deepwm_map_normal_path,perivent_map_normal_path




def perform_cluster_analysis(thresh_path, minclustersize = 150):
    
    normal_image_dir = os.path.dirname(thresh_path)
    
    
    th = os.path.basename(thresh_path).split(".")[0].split("_")[-1]
    print(th)
    # Generate output path for bianca output
    bianca_normal_cluster_path = f"{normal_image_dir}/bianca_out_mincluster_{th}_{minclustersize}.nii.gz"
    
    # Perform cluster analysis on thresholded image
    os.system(f"cluster --in={thresh_path} --thresh=0.05 \
    --oindex={bianca_normal_cluster_path} --connectivity=6 --no_table --minextent={minclustersize}")
    
    # Convert output to binary mask
    os.system(f"fslmaths {bianca_normal_cluster_path} -bin {bianca_normal_cluster_path}")
    
    return bianca_normal_cluster_path

   
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




config_dict =get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]


data_pp_path                          =     os.path.join(datasets_prepared_path,"data_prepared_data.xlsx")
data_set_df                           =     pd.read_excel(data_pp_path,index_col=None)

thresholds_list          = [96,97,98,99]
distance_to_ventrickles  = [9,10,11]

minimum_clustersizes     = [100,125,150,175,200]


nulled_images_path                    =     os.path.join(datasets_prepared_path,"nulled_images")

bianca_results = [i for i in get_all_dirs(datasets_prepared_path) if "result" in i]

                
result_data_path           = os.path.join(datasets_prepared_path,"result_data.xlsx")
result_paths_path           = os.path.join(datasets_prepared_path,"result_paths.xlsx")
    
df_result = pd.DataFrame()      
df_paths_result = pd.DataFrame()

for bianca_result_path in bianca_results:

    nulled_images_paths_list    = get_all_dirs(nulled_images_path)
    bianca_result_paths_list    = get_all_dirs(bianca_result_path)
    

    for bri, bianca_result in enumerate(bianca_result_paths_list):
    
    
        print(bianca_result)
        
        post_processed  = os.path.join(bianca_result,"post_processed")
        create_dir(post_processed)
        
        
        subject_name = os.path.basename(bianca_result)
        
        test_patient_df                                = data_set_df.loc[data_set_df['subject'] == subject_name,:]
        
        test_patient_df.columns
        
        patient_id = test_patient_df['patient_id'].values[0]
        dataset_name = test_patient_df['dataset_name'].values[0]
    
        
        subject_null_image_path  = [i for i in nulled_images_paths_list if subject_name in i][0]
    
        
        flair_image              = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "bet" in f][0]
        mask_image               = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "WMHmask" in f][0]
        result_image             = [join(bianca_result,f) for f in os.listdir(bianca_result) if "nii" in f and not "null" in f ] [0]
        
        
        flair_csb_fluidname_path  = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "csb_fluid" in f] [0]
        
        
        flair_null_images        = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "null" in f  and "flair" in f]
        mask_null_images         = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "null" in f  and "mask" in f] 
        result_null_images       = [join(bianca_result,f) for f in os.listdir(bianca_result) if "null" in f  ] 
        
        main_processed           = os.path.join(post_processed,"main")
    
        
        confusion_processed      = os.path.join(main_processed,"confusion")
        
        brain_volume             = get_volume(flair_image)
        
        create_dir(main_processed)
        create_dir(confusion_processed)
        
        
        
        for threshold in thresholds_list:
            print(f"threshold :  {threshold} ")
            
            output_mask_copy,th_path = make_threshholde(result_image , main_processed,threshold)
            
            for minimum_clustersize in minimum_clustersizes:
                
                bianca_normal_cluster_path  =  perform_cluster_analysis(th_path,minimum_clustersize)
                
                dice_score_cl,confusion_matrix=create_confusion_matrix(bianca_normal_cluster_path,mask_image,confusion_processed)
                
                
                for distance_to_ventrickle in distance_to_ventrickles:
                    
                    print(f"distance_to_ventrickle :  {distance_to_ventrickle} ")
                    deepwm_map_normal_path,perivent_map_normal_path  =    create_wm_distances(bianca_normal_cluster_path,
                                                                          flair_csb_fluidname_path ,
                                                                          distance_to_ventrickle   = distance_to_ventrickle)
                    
                    total_lesion_volume                              = get_volume(bianca_normal_cluster_path) 
                    pvent_vol_10mm_volume                            = get_volume(perivent_map_normal_path)  
                    deep_vol_10mm_volume                             = get_volume(deepwm_map_normal_path)    
                    
    
                    
                    df_result_temp = pd.DataFrame(index=(bri,))
    
                    df_result_temp["patient_id"]            = patient_id
                    df_result_temp["dataset_name"]          = dataset_name
                    df_result_temp["subject_name"]          = subject_name
                    df_result_temp["null_type"]             = 0
                    df_result_temp["threshold"]             = threshold
                    df_result_temp["minimum_clustersize"]   = minimum_clustersize
                    
                    df_result_temp["dice_score_cl"]         = dice_score_cl
                    df_result_temp["distance_to_ventrickle"]= distance_to_ventrickle
                    
                    df_result_temp["total_lesion_volume"]   = np.round(total_lesion_volume /brain_volume,7)
                    df_result_temp["pvent_vol_10mm_volume"] = np.round(pvent_vol_10mm_volume/brain_volume,7)
                    df_result_temp["deep_vol_10mm_volume"]  = np.round(deep_vol_10mm_volume/brain_volume,7)
     
                    df_result_temp["brain_volume"]  = brain_volume
                    
                    
                    df_result_temp["tp"]  = confusion_matrix['tp']['value']
                    df_result_temp["fp"]  = confusion_matrix['fp']['value']
                    df_result_temp["fn"]  = confusion_matrix['fn']['value']
                    
                    df_result = pd.concat([df_result,df_result_temp]) 
                    
                    df_result.to_excel(result_data_path)
                    
                    df_result_path_temp = pd.DataFrame(index=(bri,))
                    df_result_path_temp["patient_id"]            = patient_id
                    df_result_path_temp["dataset_name"]          = dataset_name
                    df_result_path_temp["subject_name"]          = subject_name
                    df_result_path_temp["null_type"]             = 0
                    df_result_path_temp["threshold"]             = threshold
                    df_result_path_temp["minimum_clustersize"]   = minimum_clustersize
                    
                    df_result_path_temp["th_path"]      = th_path
                    
                    df_result_path_temp["bianca_normal_cluster_path"] = bianca_normal_cluster_path
                    df_result_path_temp["perivent_map_normal_path"]   = perivent_map_normal_path
                    df_result_path_temp["deepwm_map_normal_path"]     = deepwm_map_normal_path
                    
                                    
                    df_result_path_temp["tp"]  = confusion_matrix['tp']['path']
                    df_result_path_temp["fp"]  = confusion_matrix['fp']['path']
                    df_result_path_temp["fn"]  = confusion_matrix['fn']['path']
                    
                    df_paths_result = pd.concat([df_paths_result,df_result_path_temp]) 
                    df_paths_result.to_excel(result_paths_path)
            
        
        
        flair_null_images        = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "null" in f  and "flair" in f]
        mask_null_images         = [join(subject_null_image_path,f) for f in os.listdir(subject_null_image_path) if "null" in f  and "mask" in f] 
        result_null_images       = [join(bianca_result,f) for f in os.listdir(bianca_result) if "null" in f  ] 
        
        
        for null_in, (flair_null_image, mask_null_image,result_image) in enumerate(zip(flair_null_images,mask_null_images,result_null_images)):
            
            
            #print(flair_null_image)
            #print(mask_null_image)
            #print(result_image)
            
            null_in +=1
            
            print(null_in)
    
            main_processed           = os.path.join(post_processed,f"null_{null_in}")
            
            confusion_processed      = os.path.join(main_processed,f"confusion")
                
            create_dir(main_processed)
            create_dir(confusion_processed)
            
            
            for threshold in thresholds_list:
                print(f"threshold :  {threshold} ")
                
                output_mask_copy,th_path = make_threshholde(result_image , main_processed,threshold)
                
                for minimum_clustersize in minimum_clustersizes:
                    
                    bianca_normal_cluster_path  =  perform_cluster_analysis(th_path,minimum_clustersize)
                    
                    dice_score_cl,confusion_matrix=create_confusion_matrix(bianca_normal_cluster_path,mask_image,confusion_processed)
                    
                    
                    for distance_to_ventrickle in distance_to_ventrickles:
                        
                        print(f"distance_to_ventrickle :  {distance_to_ventrickle} ")
                        deepwm_map_normal_path,perivent_map_normal_path  =    create_wm_distances(bianca_normal_cluster_path,
                                                                              flair_csb_fluidname_path ,
                                                                              distance_to_ventrickle   = distance_to_ventrickle)
                        
                        total_lesion_volume                              = get_volume(bianca_normal_cluster_path) 
                        pvent_vol_10mm_volume                            = get_volume(perivent_map_normal_path)  
                        deep_vol_10mm_volume                             = get_volume(deepwm_map_normal_path)    
                        
    
                        df_result_temp = pd.DataFrame(index=(bri,))
    
                        df_result_temp["patient_id"]            = patient_id
                        df_result_temp["dataset_name"]          = dataset_name
                        df_result_temp["subject_name"]          = subject_name
                        df_result_temp["null_type"]             = null_in
                        df_result_temp["threshold"]             = threshold
                        df_result_temp["minimum_clustersize"]   = minimum_clustersize
                        
                        df_result_temp["dice_score_cl"]         = dice_score_cl
                        df_result_temp["distance_to_ventrickle"]= distance_to_ventrickle
                        
    
                        
                        df_result_temp["total_lesion_volume"]   = np.round(total_lesion_volume /brain_volume,7)
                        df_result_temp["pvent_vol_10mm_volume"] = np.round(pvent_vol_10mm_volume/brain_volume,7)
                        df_result_temp["deep_vol_10mm_volume"]  = np.round(deep_vol_10mm_volume/brain_volume,7)
         
                        df_result_temp["brain_volume"]  = brain_volume
                        
                        
                        df_result_temp["tp"]  = confusion_matrix['tp']['value']
                        df_result_temp["fp"]  = confusion_matrix['fp']['value']
                        df_result_temp["fn"]  = confusion_matrix['fn']['value']
                        
                        df_result = pd.concat([df_result,df_result_temp]) 
                        
                        df_result.to_excel(result_data_path)
                        
                        df_result_path_temp = pd.DataFrame(index=(bri,))
                        
                        df_result_path_temp["patient_id"]            = patient_id
                        df_result_path_temp["dataset_name"]          = dataset_name
    
                        df_result_path_temp["subject_name"]          = subject_name
                        df_result_path_temp["null_type"]             = null_in
                        df_result_path_temp["threshold"]             = threshold
                        df_result_path_temp["minimum_clustersize"]   = minimum_clustersize
                        
                        df_result_path_temp["th_path"]      = th_path
                        
                        df_result_path_temp["bianca_normal_cluster_path"] = bianca_normal_cluster_path
                        df_result_path_temp["perivent_map_normal_path"]   = perivent_map_normal_path
                        df_result_path_temp["deepwm_map_normal_path"]     = deepwm_map_normal_path
                        
                                        
                        df_result_path_temp["tp"]  = confusion_matrix['tp']['path']
                        df_result_path_temp["fp"]  = confusion_matrix['fp']['path']
                        df_result_path_temp["fn"]  = confusion_matrix['fn']['path']
                        
                        df_paths_result = pd.concat([df_paths_result,df_result_path_temp]) 
                        df_paths_result.to_excel(result_paths_path)
                
    


