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
from modules.paths_library import create_tabel_for_all_subjects

from nipype.interfaces.ants import N4BiasFieldCorrection

from nipype import Node, Workflow
from nipype.interfaces.fsl import Reorient2Std
from nipype.interfaces.fsl import SliceTimer, MCFLIRT, Smooth, BET,ApplyXFM,ConvertXFM

from nipype.interfaces.fsl import ApplyXFM
from nipype.interfaces.ants.segmentation import BrainExtraction

from os.path import join
from subprocess import run,call    
from os import listdir

import skimage.transform as skTrans
import nibabel as nib


def get_volume(path):
    
    output_all_matter = run(["fslstats",
                    path,
                    "-V"], 
                    capture_output=True)
            
    result_str = str(output_all_matter.stdout)
    all_matter_result = result_str.split(" ")

    return float(all_matter_result[1])


def normalize_volume(image):
    """normalize 3D Volume image"""
    
    
    dim     = image.shape
    volume = image.reshape(-1, 1)
    
    rvolume = np.divide(volume - volume.min(axis=0), (volume.max(axis=0) - volume.min(axis=0)))*1
    rvolume = rvolume.reshape(dim)
    
    return rvolume


def copy_mask(mask_name,flair_image_path,prep_subject_dir):

    
    if mask_name.endswith(".nii.gz"):
        standard_mask_name                    =  "WMHmask.nii.gz" 	#brain extracted
    elif mask_name.endswith(".nii"):
        standard_mask_name                    =  "WMHmask.nii" 	#brain extracted
    
    
    standard_mask_path   =  join(prep_subject_dir,standard_mask_name)
    
    os.system(f"cp {mask_path} {standard_mask_path} ")
    
    standard_mask_img_nib          = nib.load(mask_path)
    standard_mask_img              = standard_mask_img_nib.get_fdata()
    mask_shape                     = standard_mask_img.shape
    
    flair_image_img_nib            = nib.load(flair_image_path)
    flair_image_img_ref            = flair_image_img_nib.get_fdata()
    ref_shape                      = flair_image_img_ref.shape
    
    if mask_shape == ref_shape:
        print("same")
        standard_mask_img_resized      = standard_mask_img
    else:
        print("not same")
        standard_mask_img_resized      = skTrans.resize(standard_mask_img, ref_shape, order=1, preserve_range=True) 
        
    mask_img_resized = nib.Nifti1Image(standard_mask_img_resized, standard_mask_img_nib.affine)
    nib.save(mask_img_resized, standard_mask_path)
    
    
    return standard_mask_path


config_dict =get_main_paths()

datasets_original_path                =     config_dict['PATHS']["datasets_original"]
datasets_prepared_path                =     config_dict['PATHS']["datasets_prepared"]
prepared_bianca                       =     config_dict['PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['PATHS']["standard_space_flair"]
datasets_list                         =     get_all_dirs(datasets_original_path)
data_location_table_path              =     os.path.join(datasets_prepared_path,"data_location.xlsx")
Set_1_path                            =     os.path.join(datasets_prepared_path,"Set_2")
data_location_table_df                =     pd.read_excel(data_location_table_path)





data_pp_path                        =     os.path.join(datasets_prepared_path,"data_prepared_data.xlsx")



subject_list                          =     list(data_location_table_df.loc[:,"subject"])


if os.path.isfile(data_pp_path):
    data_set_df                           =     pd.read_excel(data_pp_path)
else:
    data_set_df                           =     pd.DataFrame()


subject_index = int(list(data_set_df["subject"])[-1].split("-")[-1])

for subject in subject_list[:36]:
    
    
    subject_index +=1
    
    print(subject)
    
    single_patient_data = data_location_table_df.loc[data_location_table_df['subject'] == subject,:]

    
    patient_id          =  single_patient_data["patient_id"].values[0]
    dataset_name        =  single_patient_data["dataset_name"].values[0]


    
    mask_path           =  single_patient_data["mask_path"].values[0]
    flair_image_path    =  single_patient_data["flair_image_path"].values[0]
    t1_image_path       =  single_patient_data["t1_image_path"].values[0]
    
    
    subject_index_name                      = f"sub-{str(subject_index).zfill(2)}"
    
    print(dataset_name)
    
        
    patient_df = pd.DataFrame(index=(subject_index_name,))
    
    patient_df["subject"]          = subject_index_name
    
    patient_df["patient_id"]       = patient_id
    patient_df["dataset_name"]     = dataset_name
    

    prep_subject_dir                            =   os.path.join(Set_1_path,subject_index_name)
    create_dir(prep_subject_dir)
    if flair_image_path:

        image_name = "flair"
    
    
        MNI_flair_xfm                         =  f"{image_name}_to_mni.mat" 	        #transformation matrix from template to MNI
        base_name                             =  f"{image_name}.nii.gz" 	#brain extracted
        reoriented_image_name                 =  f"{image_name}_reoriented.nii.gz" 	#reoriented_corrected
        biascorrected_image_name              =  f"{image_name}_bias_corrected.nii.gz" 	#bias_corrected
        biascorrected_bet_image_name          =  f"{image_name}_bet.nii.gz" 	#brain extracted
        biascorrected_bet_normalized          =  f"{image_name}_normalized.nii.gz" 	#brain extracted
        
        
        
        if "gz" in mask_path:
            mask_name                             =  f"WMHmask.nii.gz" 
            mask_normalized                       =  f"WMHmask_normalized.nii.gz" 	#brain extracted
            mask_registered_name                  =  f"WMHmask_registered.nii.gz" 
        else:    
            mask_name                             =  f"WMHmask.nii" 
            mask_normalized                       =  f"WMHmask_normalized.nii.gz" 	#brain extracted
            mask_registered_name                  =  f"WMHmask_registered.nii.gz" 

        
        MNI_xfm_path                          =  join(prep_subject_dir,MNI_flair_xfm)
        base_image_path                       =  join(prep_subject_dir,base_name)
        reoriented_image_path                 =  join(prep_subject_dir,reoriented_image_name)
        biascorrected_image_path              =  join(prep_subject_dir,biascorrected_image_name)
        biascorrected_bet_image_path          =  join(prep_subject_dir,biascorrected_bet_image_name)
        biascorrected_normalized_image_path   =  join(prep_subject_dir,biascorrected_bet_normalized)
        
        mask_registered_path                  =  join(prep_subject_dir,mask_registered_name)
        mask_normalized_path                  =  join(prep_subject_dir,mask_normalized)
        mask_n_path                           =  join(prep_subject_dir,mask_name)
        
        
        os.system(f"cp {mask_path} {mask_n_path}")
        
        patient_df["biascorrected_bet_image_path"]            = biascorrected_bet_image_path
        patient_df["biascorrected_normalized_image_path"]     = biascorrected_normalized_image_path
        patient_df["MNI_xfm_path"]                            = MNI_xfm_path
        patient_df["mask_registered_path"]                    = mask_registered_path
        patient_df["mask_normalized_path"]                    = mask_normalized_path
        
        
    
        reoriented_image_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if reoriented_image_name in l.lower() ] 
    
        if reoriented_image_paths:
            print("found reoriented_image_path")
            reoriented_image_path = reoriented_image_paths[0]
        else:
            print("run reoriented_image_path")
            wf                                    = Workflow(name="prep_flair")
            reorient                              = Node(Reorient2Std(), name="flair")
            reorient.inputs.in_file               = flair_image_path
            reorient.inputs.out_file              = reoriented_image_path
            reorient.run() 
                      
    
        #biascorrection
        biascorrected_image_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if biascorrected_image_name in l.lower() ] 
        
        
        if biascorrected_image_paths:
            biascorrected_image_path = biascorrected_image_paths[0]
            print("found biascorrected_image_path")
        else:
            print("run biascorrected_image")
            biascorr                              = Node(N4BiasFieldCorrection(save_bias=False, 
                                                                               num_threads=-1),
                                                                               name="biascorr")
            
            biascorr.inputs.input_image           = reoriented_image_path
            biascorr.inputs.output_image          = biascorrected_image_path
            biascorr.run()
            
            
        #skullstrip
        biascorrected_bet_image_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if biascorrected_bet_image_name in l.lower() ] 
        
        
        if biascorrected_bet_image_paths:
            print("found biascorrected_bet_image_path")
            biascorrected_bet_image_path = biascorrected_bet_image_paths[0]
        else:
            print("run biascorrected_bet")
            #skullstrip
            skullstrip                            = Node(BET(mask=False), 
                                                         name="skullstrip")
            
            #-B
            #skullstrip.inputs.reduce_bias = True
            #-f
            skullstrip.inputs.frac = 0.5
            #-g
            #skullstrip.inputs.vertical_gradient = (1,)

              
            """FSL BET wrapper for skull stripping

            For complete details, see the `BET Documentation.
            <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide>`_

            Examples
            --------
            >>> from nipype.interfaces import fsl
            >>> btr = fsl.BET()
            >>> btr.inputs.in_file = 'structural.nii'
            >>> btr.inputs.frac = 0.7
            >>> btr.inputs.out_file = 'brain_anat.nii'
            >>> btr.cmdline
            'bet structural.nii brain_anat.nii -f 0.70'
            >>> res = btr.run() # doctest: +SKIP

            """
            
            
            skullstrip.inputs.in_file             = biascorrected_image_path
            skullstrip.inputs.out_file            = biascorrected_bet_image_path
            skullstrip.run()
            
            
            
        # #mask normalization
        # mask_normalized_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if mask_normalized in l.lower() ] 
        
        # if mask_normalized_paths:
        #     print("found MNI_xfm_paths  {MNI_xfm_paths}")
        #     mask_normalized_path = mask_normalized_paths[0]
            
        #     print(f"found mask normalization  {mask_normalized_path}")
        # else:
        #     print("run mask normalization")

        #     output_mask_normalization             = run(["flirt",
        #                                         "-ref",biascorrected_bet_image_path,
        #                                         "-in", mask_path,
        #                                         "-out",mask_registered_path], 
        #                                         capture_output=True)


        MNI_flair_xfm_name = MNI_flair_xfm.split(".")[0]
        #flair normalization
        MNI_xfm_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if MNI_flair_xfm_name in l.lower() ] 
        
        if MNI_xfm_paths:
            print("found MNI_xfm_paths  {MNI_xfm_paths}")
            MNI_xfm_path = MNI_xfm_paths[0]
            
            print(f"found MNI_xfm_path  {MNI_flair_xfm_name}")
        else:
            print("run flair normalization")
            output_mask_normalization             = run(["flirt",
                                                "-ref",standard_space_flair,
                                                "-in", biascorrected_bet_image_path,
                                                "-omat",MNI_xfm_path,
                                                "-out",biascorrected_normalized_image_path], 
                                                capture_output=True)
            

        #mask normalization
        mask_normalized_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if mask_normalized in l.lower() ] 
        
        if mask_normalized_paths:
            print("found MNI_xfm_paths  {MNI_xfm_paths}")
            mask_normalized_path = mask_normalized_paths[0]
            
            print(f"found mask normalization  {mask_normalized_path}")
        else:
            print("run mask normalization")

            # output_mask_normalization             = run(["flirt",
            #                                     "-ref",standard_space_flair,
            #                                     "-in", mask_path,
            #                                     "-out",mask_normalized_path], 
            #                                     capture_output=True)
            output_mask_normalization             = run(["flirt",
                                                "-ref",biascorrected_normalized_image_path,
                                                "-in", mask_path,
                                                "-out",mask_normalized_path,
                                                "-interp","nearestneighbour",
                                                "-applyxfm",
                                                "-init",MNI_xfm_path], 
                                                capture_output=True)
            
            
        biascorrected_bet_3d                                  = normalize_volume(nib.load(biascorrected_bet_image_path).get_fdata()  )  
        biascorrected_normalized_3d                           = normalize_volume(nib.load(biascorrected_normalized_image_path).get_fdata())   
        
        mask_3d                                               = nib.load(mask_path).get_fdata()    
        mask_normalized_3d                                    = nib.load(mask_normalized_path).get_fdata()  
        

        data_set_df = pd.concat([data_set_df , patient_df])
        
        data_set_df.to_excel(data_pp_path)

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
            

            

            
            
            