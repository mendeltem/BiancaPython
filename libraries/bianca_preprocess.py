#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:36:57 2023

@author: temuuleu
"""

import os 
from os import listdir
from os.path import join, basename
from nipype import Node, Workflow
from nipype.interfaces.fsl import Reorient2Std
from nipype.interfaces.fsl import SliceTimer, MCFLIRT, Smooth, BET,ApplyXFM,ConvertXFM
from nipype.interfaces.fsl import ApplyXFM


from nipype.interfaces.ants.segmentation import BrainExtraction

import pandas as pd
from subprocess import run,call    
import nibabel as nib
import numpy as np
import skimage.transform as skTrans
from scipy.ndimage import rotate

from pathlib import Path

import matplotlib.pyplot as plt
from nipype.interfaces.ants import N4BiasFieldCorrection

import subprocess


def run_bianca(master_file_path,
               prep_session_dir,
               trainstring,
               row_number,
                brainmaskfeaturenum       = 1,
                spatialweight             = 2,
                labelfeaturenum           = 3,
                trainingpts               = 10000,
                selectpts                 = "noborder",
                nonlespts                 = 10000,
                output_mask               = "seg_prob_bianca.nii.gz"
               ):

 
    
    output_mask_path                        = join(prep_session_dir,output_mask)
    
    if not os.path.isfile(output_mask_path):
    

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
                                                  
        try:
            os.system(train_bianca_commands)
        except:
            print("bianca command failed")
        
        
    return output_mask_path


def create_masterfile(master_file_path,
                                biascorrected_bet_image_path,
                                MNI_xfm_path,
                                mask_path,
                                subject_dir
                                ):

    #dirname = os.path.dirname(master_file_path)

    with open(master_file_path, 'r') as file:
        master_file_txt = file.read()
        
        
    row_number =master_file_txt.count("flair_to_mni.mat")+1

    trainstring = "".join([str(r)+"," for r in range(1,row_number)])[:-1]
            
    master_file_test=    biascorrected_bet_image_path+" " +MNI_xfm_path+" "+mask_path


    all_master_file_txt  = master_file_txt + "\n"+master_file_test+"\n"

    new_master_file_path = join(subject_dir,"master_file.txt")
    
    
    with open(new_master_file_path, 'w') as f:
        f.write(all_master_file_txt)
        
    return    new_master_file_path,trainstring,row_number



def preprocess_subject(data):
    
    null_image_name     ="flair_null.nii.gz"
    controll_image_name ="flair_controll.nii.gz"
    null_mask_name      ="mask_null.nii.gz"
    
    
    subject,data_location_table_df,data_dict=data
    
    data_preprocessed_set  = data_dict["data_preprocessed_set"]
    standard_space_flair  = data_dict["standard_space_flair"]
    

    subject_df = pd.DataFrame(index=(1,))
    
    subject_row   = data_location_table_df[data_location_table_df["subject"] == subject]
         
    subject = subject_row["subject"].values[0]
    original_patien_name = subject_row["original_patien_name"].values[0]

    name_mask = subject_row["name_mask"].values[0]
    path_mask = subject_row["path_mask"].values[0]
    name_null_image = subject_row["name_null_image"].values[0]
    path_null_image = subject_row["path_null_image"].values[0]
    name_flair_image = subject_row["name_flair_image"].values[0]
    path_flair_image = subject_row["path_flair_image"].values[0]
    mprage_pp = subject_row["mprage_pp"].values[0]
    mprage_volume = subject_row["mprage_volume"].values[0]
    path_mprage_pp = subject_row["path_mprage_pp"].values[0]
    mprage = subject_row["mprage"].values[0]
    path_mprage = subject_row["path_mprage"].values[0]

    # Get column names with NaN values
    nan_columns = subject_row.columns[subject_row.isna().any()].tolist()
    
    subject_dir = data_preprocessed_set / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    null_image_path         = subject_dir / null_image_name
    controll_image_path     = subject_dir / controll_image_name
    
    null_mask_image_path     = subject_dir / null_mask_name
    
    
    subject_df  = subject_row
    
    #subject_df["subject"] = subject
        
        
    #if null image is not found create own with the help of the mask
    if "name_null_image" in  nan_columns and not "name_mask" in nan_columns:
        #print(f"NaN values detected in {nan_columns} for {subject}")
        create_null_image(path_mask,path_flair_image,null_image_path )
          
    elif not "name_null_image" in  nan_columns:
        subject_dir = data_preprocessed_set / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", path_null_image, null_image_path], check=True)
        
   
    subject_df["null_image_path"] = null_image_path

    #check for missing rage
    #if copy the controll flair image
    if not "path_flair_image" in  nan_columns:
        #print(f"NaN values detected in {nan_columns} for {subject}")
        subprocess.run(["cp", path_flair_image, controll_image_path], check=True)
        
        
    subject_df["controll_image_path"] = controll_image_path
        
        
    #check for missing rage
    #if copy the controll flair image
    if not "path_mask" in  nan_columns:
        #print(f"NaN values detected in {nan_columns} for {subject}")
        subprocess.run(["cp", path_mask, null_mask_image_path], check=True)
        
    #check for missing rage
    #if copy the controll flair image
    if "path_mask" in  nan_columns and not "name_null_image" in  nan_columns:
        #print(f"NaN values detected in {nan_columns} for {subject}")
        create_null_mask(path_null_image,path_flair_image,null_mask_image_path )
        
        

    #check for missing rage
    #if null image is not found create own with the help of the mask
    if not "mprage_volume" in  nan_columns:
        #print(f"NaN values detected in {nan_columns} for {subject}")
        subject_df["mprage_volume"] = mprage_volume
        
        
    
    controll_pp_path,flair_MNI_xfm_path   =  preprocess_image(controll_image_path,
                                                           subject_dir,
                                                           "controll",
                                                           standard_space_flair)
    
    null_pp_path,flair_MNI_xfm_path   =  preprocess_image(null_image_path,
                                                          subject_dir,
                                                          "null",
                                                          standard_space_flair)
    
    subject_df["controll_pp_path"] = controll_pp_path
    subject_df["null_pp_path"]     = null_pp_path
    subject_df["mni_matrix"]       = flair_MNI_xfm_path
    
    flair_segmentation             = segment_image(controll_pp_path)
    subject_df["csb_fluid_path"]   = flair_segmentation["csb_fluid"]["path"]
    
    th_csb                    = 0.99
    minimum_clustersize_csb   = 3000
    th_csb_str                = int(minimum_clustersize_csb)
    
    flair_csb_clustered_path = Path(os.path.dirname(flair_segmentation["csb_fluid"]["path"])) / f"csb_clustered_cs_{th_csb_str}.nii.gz"
    
    os.system(f"fslmaths {flair_segmentation['csb_fluid']['path']} -thr {th_csb} -bin {flair_csb_clustered_path}")
    os.system(f"cluster --in={flair_csb_clustered_path} --thresh=0.05 --oindex={flair_csb_clustered_path} --connectivity=6 --no_table --minextent={minimum_clustersize_csb}")

    subject_df["csb_fluid_clustered_path"]   = flair_csb_clustered_path
    
    
    #to mni space
    
    null_image_mni_name                   = "null_pp_mni.nii.gz"
    null_image_mni_path                   = subject_dir / null_image_mni_name
    
    controll_image_mni_name               = "controll_pp_mni.nii.gz"
    controll_image_mni_path               = subject_dir / controll_image_mni_name
    
    
    flair_csb_clustered_path_mni_name     = "csb_fluid_pp_mni.nii.gz"
    flair_csb_clustered_path_mni_path     = subject_dir / flair_csb_clustered_path_mni_name
    
    
    
    null_mask_mni_name                   = "null_mask_mni.nii.gz"
    null_mask_mni_path                   = subject_dir / null_mask_mni_name
    
    
    if not os.path.isfile(null_mask_mni_path):
        normal_mask_cmd = f"flirt -in {null_mask_image_path}  -ref {standard_space_flair}  -out {null_mask_mni_path} -init {flair_MNI_xfm_path} -applyxfm"
        os.system(normal_mask_cmd)
    
    
    if not os.path.isfile(null_image_mni_path):
        normal_mask_cmd = f"flirt -in {null_pp_path}  -ref {standard_space_flair}  -out {null_image_mni_path} -init {flair_MNI_xfm_path} -applyxfm"
        os.system(normal_mask_cmd)
        
        
    if not os.path.isfile(controll_image_mni_path):
        normal_mask_cmd = f"flirt -in {controll_pp_path}  -ref {standard_space_flair}  -out {controll_image_mni_path} -init {flair_MNI_xfm_path} -applyxfm"
        os.system(normal_mask_cmd)
        
        
    if not os.path.isfile(flair_csb_clustered_path_mni_path):
        normal_mask_cmd = f"flirt -in {flair_csb_clustered_path}  -ref {standard_space_flair}  -out {flair_csb_clustered_path_mni_path} -init {flair_MNI_xfm_path} -applyxfm"
        os.system(normal_mask_cmd)
        
    
    subject_df["null_image_mni_path"]                = null_image_mni_path
    subject_df["controll_image_mni_path"]            = controll_image_mni_path
    subject_df["flair_csb_clustered_path_mni_path"]  = flair_csb_clustered_path_mni_path
    subject_df["null_mask_mni_path"]                 = null_mask_mni_path
    

    return pd.DataFrame(subject_df)
  



def create_null_image(path_mask,path_flair_image,null_image_path ):

    try:
        null_mask_img = nib.load(path_mask).get_fdata()
        flair_nib = nib.load(path_flair_image)
        flair_img = flair_nib.get_fdata()
        null_image = (np.logical_not(null_mask_img[:,:,:])) * flair_img[:,:,:]
        null_image_nib = nib.Nifti1Image(null_image, flair_nib.affine)
        nib.save(null_image_nib, null_image_path)
        return 0
    except:
        return 1
    
    
    
    

def create_null_mask(path_null_image,path_flair_image,null_mask_image_path ):

    try:
        controll_nib = nib.load(path_flair_image)
        controll_img = controll_nib.get_fdata()
        
        null_img = nib.load(path_null_image).get_fdata()
        
        img_diff = np.abs(controll_img - null_img)
        
        
        controll_thresholded_img = np.greater(controll_img, 0.001).astype(np.uint8) 
        null_thresholded_img = np.greater(null_img, 0.001).astype(np.uint8) 
        

        null_mask = np.abs(controll_img - null_img)
        
        output_mask_copy = np.copy(null_mask)
        
    
        output_mask_copy[output_mask_copy >=0.01] = 1
        output_mask_copy[output_mask_copy <0] = 0
        

        null_image_nib = nib.Nifti1Image(output_mask_copy, controll_nib.affine)
        nib.save(null_image_nib, null_mask_image_path)
        return 0
    except:
        return 1

def segment_image(flair_pp_path,image_type = ""):
    

    #create csb fluid, gm and wm from flair/brain
    
    bet_name = os.path.basename(flair_pp_path)
    
    prep_subject_dir  = '/'.join(flair_pp_path.split('/')[0:-1])
    

    #rename csbfluid
    csb_fluid_name_fast = bet_name.split(".")[0]+"_pve_2"+"."+bet_name.split(".")[1]+"."+ bet_name.split(".")[2]
    csb_new_fluidname = f"segment_{image_type}_csb_fluid.nii.gz"
    
    csb_fluid_name_fast_path = join(prep_subject_dir, csb_fluid_name_fast)
    csb_new_fluidname_path  = join(prep_subject_dir, csb_new_fluidname)


    #rename grey mater
    grey_matter_name_fast = bet_name.split(".")[0]+"_pve_0"+"."+bet_name.split(".")[1]+"."+ bet_name.split(".")[2]
    grey_matter_name_new_fluidname = f"segment_{image_type}_grey_matter.nii.gz"
    
    grey_matter_name_fast_path           = join(prep_subject_dir, grey_matter_name_fast)
    grey_matter_name_new_fluidname_path  = join(prep_subject_dir, grey_matter_name_new_fluidname)

    
    #rename white mater
    white_matter_name_fast = bet_name.split(".")[0]+"_pve_1"+"."+bet_name.split(".")[1]+"."+ bet_name.split(".")[2]
    white_matter_name_new_fluidname = f"segment_{image_type}_white_matter.nii.gz"
    
    white_matter_name_fast_path = join(prep_subject_dir, white_matter_name_fast)
    white_matter_name_new_fluidname_path  = join(prep_subject_dir, white_matter_name_new_fluidname)

    mixtype_name_fast = bet_name.split(".")[0]+"_mixeltype"+"."+bet_name.split(".")[1]+"."+ bet_name.split(".")[2]
    mixtype_name_fast_path = join(prep_subject_dir, mixtype_name_fast)
    
    pveseg_name_fast = bet_name.split(".")[0]+"_pveseg"+"."+bet_name.split(".")[1]+"."+ bet_name.split(".")[2]
    pveseg_new_name = f"segment_{image_type}_pveseg.nii.gz"
    pveseg_name_fast_path = join(prep_subject_dir, pveseg_name_fast)
    pveseg_new_path  = join(prep_subject_dir, pveseg_new_name)
    seg_name_fast = bet_name.split(".")[0]+"_seg"+"."+bet_name.split(".")[1]+"."+ bet_name.split(".")[2]
    pveseg_name_fast_path = join(prep_subject_dir, seg_name_fast)
    seg_new_name = f"segment_{image_type}_seg.nii.gz"
    seg_new_path  = join(prep_subject_dir, seg_new_name)
    
    
    grey_matter_name_new_fluidname_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if grey_matter_name_new_fluidname in l.lower() ] 
    white_matter_name_new_fluidname_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if white_matter_name_new_fluidname in l.lower() ] 
    csb_new_fluidname_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if csb_new_fluidname in l.lower() ] 

    
    if grey_matter_name_new_fluidname_paths and white_matter_name_new_fluidname_paths \
        and csb_new_fluidname_paths:
        print("found previous segmentation")
        grey_matter_name_new_fluidname_path = grey_matter_name_new_fluidname_paths[0]
        white_matter_name_new_fluidname_path = white_matter_name_new_fluidname_paths[0]
        csb_new_fluidname_path = csb_new_fluidname_paths[0]
        
    else:
        print("run fast segmentation")
        output_segmentation                   = run(["fast","-t","2",
                                                    flair_pp_path], 
                                                    capture_output=True)
    
        os.system(f"mv {csb_fluid_name_fast_path} {csb_new_fluidname_path}")
        os.system(f"mv {grey_matter_name_fast_path} {grey_matter_name_new_fluidname_path}")
        os.system(f"mv {white_matter_name_fast_path} {white_matter_name_new_fluidname_path}")
        os.system(f"rm  {mixtype_name_fast_path}")
        os.system(f"mv {pveseg_name_fast_path} {pveseg_new_path}")
        
    new_names = {}
    new_names["csb_fluid"] =  {}
    new_names["csb_fluid"]["name"]   = csb_new_fluidname
    #new_names["csb_fluid"]["volume"] = get_volume(csb_new_fluidname_path)
    new_names["csb_fluid"]["path"]   = csb_new_fluidname_path
    
    
    new_names["grey_matter"] =  {}
    new_names["grey_matter"]["name"]   = grey_matter_name_new_fluidname
    #new_names["grey_matter"]["volume"] = get_volume(grey_matter_name_new_fluidname_path)
    new_names["grey_matter"]["path"]   = grey_matter_name_new_fluidname_path
    
    
    new_names["white_matter"] =  {}
    new_names["white_matter"]["name"]   = white_matter_name_new_fluidname
    #new_names["white_matter"]["volume"] = get_volume(white_matter_name_new_fluidname_path)
    new_names["white_matter"]["path"]   = white_matter_name_new_fluidname_path
    
    new_names["pveseg"] =  {}
    new_names["pveseg"]["name"]   = pveseg_new_name
    #new_names["pveseg"]["volume"] = get_volume(pveseg_new_path)
    new_names["pveseg"]["path"]   = pveseg_new_path


    new_names["seg"] =  {}
    new_names["seg"]["name"]   = seg_new_name
    # new_names["seg"]["volume"] = get_volume(seg_new_path)
    new_names["seg"]["path"]   = seg_new_path
        
    return new_names
        

def preprocess_image(image_path,prep_subject_dir,image_name,standard_space_flair):
    

    MNI_flair_xfm                         =  f"{image_name}_to_mni.mat" 	        #transformation matrix from template to MNI
    base_name                             =  f"{image_name}.nii.gz" 	#brain extracted
    mprage_base_name                      =  f"mprage.nii.gz" 	#brain extracted
    reoriented_image_name                 =  f"{image_name}_reoriented.nii.gz" 	#reoriented_corrected
    biascorrected_image_name              =  f"{image_name}_bias_corrected.nii.gz" 	#bias_corrected
    biascorrected_bet_image_name          =  f"{image_name}_bet.nii.gz" 	#brain extracted
    biascorrected_bet_normalized          =  f"{image_name}_normalized.nii.gz" 	#brain extracted
    
 
    
    MNI_xfm_path                          =  join(prep_subject_dir,MNI_flair_xfm)
    base_image_path                       =  join(prep_subject_dir,base_name)
    reoriented_image_path                 =  join(prep_subject_dir,reoriented_image_name)
    biascorrected_image_path              =  join(prep_subject_dir,biascorrected_image_name)
    biascorrected_bet_image_path          =  join(prep_subject_dir,biascorrected_bet_image_name)
    biascorrected_normalized_image_path   =  join(prep_subject_dir,biascorrected_bet_normalized)
    
    
    reoriented_image_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if reoriented_image_name in l.lower() ] 

    if reoriented_image_paths:
        print("found reoriented_image_path")
        reoriented_image_path = reoriented_image_paths[0]
    else:
        print("run reoriented_image_path")
        wf                                    = Workflow(name="prep_flair")
        reorient                              = Node(Reorient2Std(), name="flair")
        reorient.inputs.in_file               = image_path
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
        
        
    #biascorrection
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
        
        
    #normalization_image(biascorrected_bet_image_path)
        

    MNI_flair_xfm_name = MNI_flair_xfm.split(".")[0]
    #biascorrection
    MNI_xfm_paths = [ join(prep_subject_dir,l) for l in listdir(prep_subject_dir)  if MNI_flair_xfm_name in l.lower() ] 
    
 
    if MNI_xfm_paths:
        print("found MNI_xfm_paths  {MNI_xfm_paths}")
        MNI_xfm_path = MNI_xfm_paths[0]
        
    else:
        print("run MNI_xfm")
        output_created_mni_matrix             = run(["flirt",
                                            "-ref",standard_space_flair,
                                            "-in", biascorrected_bet_image_path,
                                            "-omat",MNI_xfm_path], 
                                            capture_output=True)
            
    
    return biascorrected_bet_image_path,MNI_xfm_path