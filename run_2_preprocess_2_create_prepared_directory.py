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
data_location_table_df                =     pd.read_excel(data_location_table_path,index_col=None)
subject_list                          =     list(data_location_table_df.loc[:,"subject"])

# if os.path.isfile(data_pp_path):
#     data_set_df                           =     pd.read_excel(data_pp_path,index_col=None)
# else:
#     data_set_df                           =     pd.DataFrame()

data_set_df                           =     pd.DataFrame()
bool_new              = False
bool_run_segmentation = False

for subject in subject_list[:]:

    print(subject)
    
    single_patient_data = data_location_table_df.loc[data_location_table_df['subject'] == subject,:]

    
    patient_id                  =  single_patient_data["patient_id"].values[0]
    dataset_name                =  single_patient_data["dataset_name"].values[0]
    lesion_concentration        =  single_patient_data["lesion_concentration"].values[0]

    mask_path                   =  single_patient_data["mask_path"].values[0]
    flair_image_path            =  single_patient_data["flair_image_path"].values[0]
    t1_image_path               =  single_patient_data["t1_image_path"].values[0]
    
    
    flair_base_name = os.path.basename(flair_image_path)
    
    
    subject_index_name                      = subject
    
    patient_df = pd.DataFrame(index=(subject_index_name,))
    
    patient_df["subject"]                    = subject_index_name
    
    patient_df["patient_id"]                 = patient_id
    patient_df["dataset_name"]               = dataset_name
    
    patient_df["lesion_concentration"]       = lesion_concentration

    print(list(single_patient_data))
    
    

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
            mask_name                             =  f"{image_name}_WMHmask.nii.gz" 
            mask_normalized                       =  f"{image_name}_WMHmask_normalized.nii.gz" 	#brain extracted
        else:    
            mask_name                             =  f"{image_name}_WMHmask.nii" 
            mask_normalized                       =  f"{image_name}_WMHmask_normalized.nii.gz" 	#brain extracted
  

        MNI_xfm_path                          =  join(prep_subject_dir,MNI_flair_xfm)
        base_image_path                       =  join(prep_subject_dir,base_name)
        reoriented_image_path                 =  join(prep_subject_dir,reoriented_image_name)
        biascorrected_image_path              =  join(prep_subject_dir,biascorrected_image_name)
        biascorrected_bet_image_path          =  join(prep_subject_dir,biascorrected_bet_image_name)
        biascorrected_normalized_image_path   =  join(prep_subject_dir,biascorrected_bet_normalized)
        
        mask_normalized_path                  =  join(prep_subject_dir,mask_normalized)
        mask_n_path                           =  join(prep_subject_dir,mask_name)
        
        
        maske_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if mask_name in l ] 
        os.system(f"cp {mask_path} {mask_n_path}")
        

        reoriented_image_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if reoriented_image_name in l.lower() ] 
        
         
        if reoriented_image_paths and not bool_new:
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
        
        
        if biascorrected_image_paths and not bool_new:
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
        
        
        if biascorrected_bet_image_paths and not bool_new:
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
        
        if MNI_xfm_paths and not bool_new:
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
        
        if mask_normalized_paths and not bool_new:
            print("found MNI_xfm_paths  {MNI_xfm_paths}")
            mask_normalized_path = mask_normalized_paths[0]
            
            print(f"found mask normalization  {mask_normalized_path}")
        else:
            print("run mask normalization")

            output_mask_normalization             = run(["flirt",
                                                "-ref",biascorrected_normalized_image_path,
                                                "-in", mask_path,
                                                "-out",mask_normalized_path,
                                                "-interp","nearestneighbour",
                                                "-applyxfm",
                                                "-init",MNI_xfm_path], 
                                                capture_output=True)
            
            
        #create csb fluid_grey_matter
        
        patient_df["biascorrected_bet_image_path"]            = biascorrected_bet_image_path
        patient_df["biascorrected_normalized_image_path"]     = biascorrected_normalized_image_path
        patient_df["MNI_xfm_path"]                            = MNI_xfm_path
        patient_df["mask_path"]                               = mask_path
        patient_df["mask_normalized_path"]                    = mask_normalized_path
        
        if bool_run_segmentation:
            
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
            
            patient_df["csb_fluid_new_name_path"]                 = csb_fluid_new_path
            patient_df["white_matter_new_name_path"]              = white_matter_new_path
            patient_df["grey_matter_new_name_path"]               = grey_matter_new_path
            
        else:
            patient_df["csb_fluid_new_name_path"]                 = ""
            patient_df["white_matter_new_name_path"]              = ""
            patient_df["grey_matter_new_name_path"]               = ""
            

        

        data_set_df = pd.concat([data_set_df , patient_df])
        save_df(data_set_df,data_pp_path)
        
        
        

        
        
        
        
        

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
            

            

            
            
            