#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:32:31 2023

@author: temuuleu
"""

import os
from dotenv import load_dotenv
from colorama import Fore

load_dotenv(verbose=True)

from bianca_gui.config.singleton import Singleton


class Config(metaclass=Singleton):
    
    def __init__(self) -> None:
        self.masterfile_dir_path       =  os.getenv("masterfile_dir_path")
        self.masterfile_file_path      =  os.getenv("masterfile_file_path")
        self.standard_space_flair      =  os.getenv("standard_space_flair")
        self.standard_space_ventrikel  =  os.getenv("standard_space_ventrikel")
        self.standard_mask             =  os.getenv("standard_mask")
        self.data_test_dir_path        =  os.getenv("data_test_dir_path")
        self.data_out_test_dir_path        =  os.getenv("data_out_test_dir_path")
        
               

def check_master_file_list_path() -> None:
    """Check if the MASTERFILE_PATH is set in config.py or as an environment variable."""
    cfg = Config()
    if not cfg.master_file_path:
        print(
            Fore.RED
            + "Please set your MASTERFILE_PATH in .env or as an environment variable."
        )
        exit(1)
