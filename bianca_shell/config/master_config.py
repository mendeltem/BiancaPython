#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:53:49 2023

@author: temuuleu
"""

import os

def create_masterfile(master_file_path,
                      input_image,
                      MNI_xfm_path,
                      input_mask,
                      output_dir=""
                      ):
    """
    Creates a master file with the given input parameters.

    Args:
        master_file_path (str): File path of the existing master file.
        input_image (str): Input image file path.
        MNI_xfm_path (str): MNI transformation file path.
        input_mask (str): Input mask file path.
        output_dir (str): Output directory path.

    Returns:
        tuple: A tuple containing the new master file path, trainstring, and row number.
    """
    
    # Check if output_dir is empty and set it to current directory if so
    if not output_dir:
        output_dir = '.'
        
    
    if not os.path.isdir(output_dir):
        raise ValueError("The specified output directory does not exist.")


    # Read the contents of the existing master file
    with open(master_file_path, 'r') as file:
        master_file_txt = file.read()

    # Count the occurrences of "flair_to_mni.mat" to determine the row number
    row_number = master_file_txt.count("flair_to_mni.mat") + 1

    # Create a comma-separated string of training numbers
    trainstring = ",".join([str(r) for r in range(1, row_number)])

    # Construct the new master file entry
    master_file_test = input_image + " " + MNI_xfm_path + " " + input_mask

    # Append the new master file entry to the existing master file contents
    all_master_file_txt = master_file_txt + "\n" + master_file_test + "\n"

    # Define the path for the new master file
    new_master_file_path = os.path.join(output_dir, "master_file.txt")

    # Write the updated master file contents to the new master file
    with open(new_master_file_path, 'w') as f:
        f.write(all_master_file_txt)

    # Return the new master file path, trainstring, and row number as a tuple
    return new_master_file_path, trainstring, row_number
