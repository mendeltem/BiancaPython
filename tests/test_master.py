#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:45:00 2023

@author: temuuleu
"""

import os
current_dir = os.getcwd()
print(f"current_dir {current_dir}")

import unittest
from tempfile import TemporaryDirectory

#from bianca_gui.config.config import Config
from bianca_shell.config.master_config import create_masterfile
from bianca_shell.bianca_shell import run_bianca


class TestCreateMasterFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = TemporaryDirectory()

    def tearDown(self):
        # Clean up the temporary directory after testing
        self.temp_dir.cleanup()

    def test_create_masterfile(self):
        # Define test input parameters
        master_file_path = "data_test/Masterfiles/small_masterfile.txt"
        input_image = "data_test/flair_image_bet.gz"
        MNI_xfm_path = "data_test/flair_to_mni.mat"
        input_mask = "data_test/flair_WMHmask.nii.gz"

        # Call the function with test input parameters
        new_master_file_path, trainstring, row_number = create_masterfile(
            master_file_path,
            input_image,
            MNI_xfm_path,
            input_mask
        )

        # Check if the new master file was created in the correct directory
        self.assertEqual(new_master_file_path, "./master_file.txt")

        # Check if the trainstring and row number are updated correctly
        self.assertEqual(trainstring, "1,2,3,4,5,6,7,8,9")
        self.assertEqual(row_number, 10)

        # Check if the new master file contains the correct entry
        with open(new_master_file_path, 'r') as f:
            new_master_file_contents = f.read()
            expected_entry = "{} {} {}\n".format(input_image, MNI_xfm_path, input_mask)
            self.assertIn(expected_entry, new_master_file_contents)



    def test_run_bianca(self):
        
        master_file_path = "data_test/Masterfiles/small_masterfil"
        input_image = "data_test/flair_image_bet.nii.gz"
        MNI_xfm_path = "data_test/flair_to_mni.mat"
        output_file  = "data_test/bianca_output.nii.gz"
        
        output_mask_path =  run_bianca(master_file_path, input_image, MNI_xfm_path, output_file)
        

        # Check if the new master file was created in the correct directory
        self.assertEqual(output_mask_path, 'Error: Training master file not found.')
        
        master_file_path = "data_test/flair_image_bet.nii.gz"
        input_image = "data_test/flair_image_bet.nii.gz"
        MNI_xfm_path = "data_test/flair_to_mni.mat"
        output_file  = "data_test/bianca_output.nii.gz"
        
        output_mask_path =  run_bianca(master_file_path, input_image, MNI_xfm_path, output_file)
        
    
        # Check if the new master file was created in the correct directory
        self.assertEqual(output_mask_path, 'Error: Incorrect training master file type. Expected .txt file.')
        
        
        master_file_path = "data_test/Masterfiles/small_masterfile.txt"
        input_image = "data_test/flair_image_bet.nii.gz"
        MNI_xfm_path = "data_test/flair_to_mni.mat"
        output_file  = "data_test/bianca_output.nii.gz"
        
        
        print(output_file)
        
        output_mask_path =  run_bianca(master_file_path, input_image, MNI_xfm_path, output_file)
        
    
        # Check if the new master file was created in the correct directory
        self.assertEqual(output_mask_path, 'Error: Incorrect training master file type. Expected .txt file.')



if __name__ == '__main__':
    unittest.main()
