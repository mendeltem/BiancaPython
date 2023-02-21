#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:42:27 2022

@author: temuuleu
"""

import os


def copy_file(file_path, to_copy_file_path):
    
    if "." in file_path:
        cmd = "cp "+file_path+" "+to_copy_file_path
    else:
        cmd = "cp -r "+file_path+" "+to_copy_file_path 
        
    os.system(cmd)
    
    
    
def get_all_dirs(data_set_manual):
    """
    
    Parameters
    ----------
    data_set_manual : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    list_of_subjects = [os.path.join(data_set_manual,directory)\
                 for directory in os.listdir(data_set_manual) if not '.' in directory]
        
    return list_of_subjects
