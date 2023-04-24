#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:12:27 2023

@author: temuuleu
"""


import os
from bianca_shell.config.master_config import create_masterfile


def run_bianca(train_masterfile_path, input_image, mni_matrix, output_file):
    """
    Runs the BIANCA function with the given input parameters.

    Args:
        train_masterfile_path (str): File path of the training master file.
        input_image (str): Input image file path.
        mni_matrix (str): MNI transformation file path.
        output_file (str): Output file name or file path.

    Returns:
        None
    """
    
 
    # Check if train_masterfile_path is provided and is a valid file path
    if not train_masterfile_path or not os.path.isfile(train_masterfile_path):
        print(train_masterfile_path)
        return "Error: Training master file not found."


    master_file_txt = "/home/temuuleu/CSB_NeuroRad/temuuleu/Projekts/BIANCA/tests/data_test/Masterfiles/small_masterfile.txt"
    master_file_txt = "/home/temuuleu/CSB_NeuroRad/temuuleu/Projekts/BIANCA/tests/data_test/Masterfiles/small_masterfile.txt"

    os.path.isfile(master_file_txt)


    # Check if train_masterfile_path has the correct file type
    if not train_masterfile_path.endswith(".txt"):
        return"Error: Incorrect training master file type. Expected .txt file."

    

    # Check if standard_mask file exists
    standard_mask = "standard_templates/WMHmask.nii.gz"
    
    BIANCA_DIR = os.environ.get('BIANCA_DIR')
    print("DIR environment variable:", BIANCA_DIR)
    
    standard_mask_abs_path = os.path.join( BIANCA_DIR,standard_mask)

    
    print("current direcotry")
    print(os.system("pwd"))
    
    if not os.path.isfile(standard_mask_abs_path):
        return "Error: Standard mask file not found."
    
    
    # Check if train_masterfile_path is provided and is a valid file path
    if not input_image or not os.path.isfile(input_image):
        print(input_image)
        return "Error: input_image file not found."

    
    # Check if train_masterfile_path has the correct file type
    if not input_image.endswith(".nii")  and  not input_image.endswith(".nii.gz"):
        return"Error: Incorrect input_image file type. Expected .txt file."
        

    # Check if output_file has another directory, if so, extract output_dir and output_file_name
    output_dir = os.path.dirname(output_file)
    output_file_name = os.path.basename(output_file)

    # If output directory is given, construct output_file path accordingly
    if output_dir:
        bianca_output = os.path.join(output_dir, output_file_name)
    else:
        bianca_output = output_file
        
         
    # Call create_masterfile function to create a new master file
    new_master_file_path, trainstring, row_number = create_masterfile(train_masterfile_path,
                                                                      input_image,
                                                                      mni_matrix,
                                                                      standard_mask,
                                                                      output_dir)
    print("BIANCA:")
    print(bianca_output)
    

    # Call run_bianca function with the new master file and other parameters
    output_mask_path = run_bianca_process(new_master_file_path,
                                  str(output_dir),
                                  trainstring, row_number,
                                  brainmaskfeaturenum=1,
                                  spatialweight=2,
                                  labelfeaturenum=3,
                                  trainingpts=10000,
                                  selectpts="noborder",
                                  nonlespts=10000,
                                  output_mask=output_file_name)
    
    
    
    
    return output_mask_path
    
    
    


def run_bianca_process(master_file_path,
               prep_session_dir,
               trainstring,
               row_number,
               brainmaskfeaturenum=1,
               spatialweight=2,
               labelfeaturenum=3,
               trainingpts=10000,
               selectpts="noborder",
               nonlespts=10000,
               output_mask="seg_prob_bianca.nii.gz"
               ):
    """
    Runs the Bianca command-line tool with the given parameters.

    Args:
        master_file_path (str): File path of the master file.
        prep_session_dir (str): Directory path for the prep session.
        trainstring (str): String containing training numbers.
        row_number (int): Query subject number.
        brainmaskfeaturenum (int, optional): Number of brain mask features. Defaults to 1.
        spatialweight (int, optional): Spatial weight. Defaults to 2.
        labelfeaturenum (int, optional): Number of label features. Defaults to 3.
        trainingpts (int, optional): Number of training points. Defaults to 10000.
        selectpts (str, optional): Method for selecting points. Defaults to "noborder".
        nonlespts (int, optional): Number of non-lesion points. Defaults to 10000.
        output_mask (str, optional): Output mask file name. Defaults to "seg_prob_bianca.nii.gz".

    Returns:
        str: File path of the output mask.
    """

    # Join the prep_session_dir and output_mask to create the output_mask_path
    output_mask_path = os.path.join(prep_session_dir, output_mask)

    # Check if the output mask file already exists
    if not os.path.isfile(output_mask_path):

        # Construct the bianca command with the input parameters
        train_bianca_commands = "bianca --singlefile=" + master_file_path + \
                                " --labelfeaturenum=" + str(labelfeaturenum) + \
                                " --brainmaskfeaturenum=" + str(brainmaskfeaturenum) + \
                                " --matfeaturenum=" + str(spatialweight) + \
                                " --trainingpts=" + str(trainingpts) + \
                                " --querysubjectnum=" + str(row_number) + " --nonlespts=" + str(nonlespts) + \
                                " --trainingnums=" + trainstring + " -o " + output_mask_path + " -v"
        print("train_bianca_commands:")
        print(train_bianca_commands)
        print("")

        try:
            # Execute the bianca command using os.system
            os.system(train_bianca_commands)
        except:
            print("bianca command failed")

    # Return the output_mask_path
    return output_mask_path




