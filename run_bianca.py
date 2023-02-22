#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 19:06:26 2023

@author: temuuleu
"""


import argparse
import subprocess

import os
from os.path import join

def train_bianca(master_file_path,
                 prep_session_dir,
                 trainstring,
                 row_number,
                 brainmaskfeaturenum       = 1,
                 spatialweight             = 2,
                 labelfeaturenum           = 3,
                 trainingpts               = 10000,
                 selectpts                 = "noborder",
                 nonlespts                 = 10000):
    
    output_mask                             = "seg_prob_bianca.nii.gz"
    output_mask_path = join(prep_session_dir, output_mask)
    
    
    train_bianca_commands = ['bianca', '--singlefile=' + master_file_path, '--labelfeaturenum=' + str(labelfeaturenum),
                             '--brainmaskfeaturenum=' + str(brainmaskfeaturenum), '--matfeaturenum=' + str(spatialweight),
                             '--trainingpts=' + str(trainingpts), '--querysubjectnum=' + str(row_number),
                             '--nonlespts=' + str(nonlespts), '--trainingnums=' + trainstring, '-o', output_mask_path, '-v']

    print("train_bianca_commands:")
    print(' '.join(train_bianca_commands))
    print("")

    try:
        subprocess.run(train_bianca_commands, check=True)
    except subprocess.CalledProcessError:
        print("bianca command failed")
        
        
#preprocess ing with Nipype
# def start_bianca(prepared_train,prep_subject_dir):


#     prepared_train_subjects_list          =     get_all_dirs(prepared_train)

#     """create master file with link for single subject"""
#     master_file_path,trainstring,row_number = create_master_file_for_bianca(prepared_train_subjects_list,prep_subject_dir)
    
#     output_mask_path                        =    train_bianca(master_file_path,  
#                                                         prep_subject_dir,
#                                                         trainstring,row_number)
    
#     return output_mask_path
        
        
def my_function():
    # Create a new ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of my function')

    # Define the command-line arguments that the function accepts
    parser.add_argument('-m', '--masterfile_path', help='Path to imaster_file_path')
    parser.add_argument('-o', '--output', help='Path to output file')
    parser.add_argument('-f', '--file', help='Path to input file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the command-line arguments
    masterfile_path = args.masterfile_path
    output_file_path = args.output
    
    
    print("masterfile : ",masterfile_path)
    print("masterfile : ",output_file_path)
    
my_function()

    # Your function code goes here, using the input_file_path and output_file_path variables as needed
# define the command to run
command = "nohup cat my_script.py &"

# use subprocess to run the command
subprocess.Popen(command, shell=True)
