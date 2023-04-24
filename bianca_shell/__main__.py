#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:40:30 2023

@author: temuuleu
"""

#from bianca_gui.single_gui import *
from bianca_gui.config.config import Config
from bianca_shell.bianca_shell import run_bianca

import os

#todo Logger
#from bianca_gui.logs import logger

import argparse

BIANCA_DIR = os.environ.get('BIANCA_DIR')
print("DIR environment variable:", BIANCA_DIR)


def main():  
    
    cfg = Config()
    print("run_bianca")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-image", type=str, help='path to input image')
    parser.add_argument("-mni", type=str, help='path to mni ')
    parser.add_argument("-masterfile", type=str, help='Path to Masterfile')
    parser.add_argument("-output", type=str, help='Path to Bianca Output')
    args = parser.parse_args()

    image_path = args.image
    mni_path = args.mni
    masterfile_path = args.masterfile
    output = args.output

    # print(f" image_path : {image_path}")
    # print(f" mni_path : {mni_path}")
    # print(f" masterfile_path : {masterfile_path}")
    
        
    master_file_path = os.path.join(BIANCA_DIR, masterfile_path)
    input_image = os.path.join(BIANCA_DIR, image_path)
    MNI_xfm_path = os.path.join(BIANCA_DIR, mni_path)
    
    
    print(f" image_path : {input_image}")
    print(f" mni_path : {MNI_xfm_path}")
    print(f" masterfile_path : {master_file_path}")
    
    print(f" output : {output}")
    
    output_mask_path =  run_bianca(master_file_path, input_image, MNI_xfm_path, output)
    print(output_mask_path)

    #run_bianca_sh.sh unittest/data_test/Masterfiles/small_masterfile.txt unittest/data_test/flair_image_bet.nii.gz
    
    # print(os.getcwdb())
    
    # BIANCA = "/home/temuuleu/CSB_NeuroRad/temuuleu/Projekts/BIANCA"
    # master_file_path = "unittest/data_test/Masterfiles/small_masterfile.txt"
    # input_image = "unittest/data_test/flair_image_bet.nii.gz"
    # MNI_xfm_path = "unittest/data_test/flair_to_mni.mat"
    # output_file  = "/home/temuuleu/bianca_output.nii.gz"

    # #output_file = os.path.join(os.getcwdb().decode(), output_file)

    # if not os.path.isfile(master_file_path):
    #     print(f"master_file_path {master_file_path}")
    
    # output_mask_path =  run_bianca(master_file_path, input_image, MNI_xfm_path, output_file)
    # print(output_mask_path)
    
    #print(cfg.master_file_path)
    #check_master_file_list_path() 
    #create_bianca_gui()
    #create_option_table()

if __name__ == "__main__":
    main()


#/home/§USER/.cache