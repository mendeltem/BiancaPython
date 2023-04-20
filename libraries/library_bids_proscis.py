#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:46:53 2022

@author: temuuleu
"""

import os
import re
import os.path
import numpy as np
from os.path import join,basename
from os import listdir

#from bids_validator import BIDSValidator
#from library import DisplayablePath, clear,plot_tree

from path_utils import  get_feature_paths
from datetime import datetime

import operator
import nibabel as nib


from path_utils import get_all_dirs,  plot_tree, clear, copy_file,copy_dir
from path_utils import create_result_dir,create_dir


from library_bids_proscis import *
from os.path import join, relpath

import pandas as pd
import matplotlib.pyplot as plt


import nibabel as nib
import skimage.transform as skTrans

import collections

import subprocess
from subprocess import run,call    


def create_dict_from_belove(kersten_path, last_index = 1):

    list_files = [join(kersten_path,directory)\
                 for directory in listdir(kersten_path)]
        
    image_paths = {} 
    
    for roots in list_files:
        
        temp_dict,last_index = create_data_dictionary_for_manual(roots,last_index)
        image_paths.update(temp_dict)
          
    return image_paths,last_index
    
def create_data_dictionary_for_manual(data_set_manual,start_index  = 1):

    import re
    
    list_files = [join(data_set_manual,directory)\
                 for directory in listdir(data_set_manual)]

    file_paths = {}
    subject_id = 0
    
    f = start_index
         
    for o,file in enumerate(list_files):
    
        file_name_withending = file.split("/")[-1]

        file_name            = file_name_withending.split(".")[0]
        
        ending               = file_name_withending.split(".")[1]
        
        if not file_name in file_paths.values():
            print(file_name)
            file_paths[f] = {}
            
            if "ROI" in file_name:
                file_paths[f]["mask"] = join(data_set_manual,file_name+"."+ending)
            else:
                file_paths[f]["flair"] = join(data_set_manual,file_name+"."+ending) 
        f +=1
    
    return file_paths,f
            
            

def create_dict_from_challange(singapur_set, start_index = 1):

    list_files = [join(singapur_set,directory)\
                 for directory in listdir(singapur_set)]
        

    image_paths = {}    
    
    f = start_index
    
     
    for o,files in enumerate(list_files):

        directory_id = basename(files)
        
            
        image_paths[f] = {}    
        image_paths[f]['mask'] = {}
        image_paths[f]['image'] = {}
        image_paths[f]['id'] = int(directory_id)

        list_masks = [join(files,directory)\
                     for directory in listdir(files) if '.' in basename(directory).lower()]
            
        list_dirs = [join(files,directory)\
                     for directory in listdir(files) if not '.' in basename(directory).lower()]
            
        for sub in list_masks:
            
            if 'kv' in basename(sub).lower():
                
                image_paths[f]['mask']['man'] =  sub
            elif '.nii' in basename(sub).lower():
                image_paths[f]['mask']['original'] =  sub
                
        for image_dir in list_dirs:
            
            if 'orig' in image_dir:
                
                image_paths[f]['image']['original'] = {}
      
                list_image = [join(image_dir,directory)\
                             for directory in listdir(image_dir) if 'flair.' in basename(directory).lower()]
                    
                image_paths[f]['image']['original']['FLAIR'] =  list_image[0]
                
                list_image = [join(image_dir,directory)\
                             for directory in listdir(image_dir) if 't1.nii.gz' == basename(directory).lower() or\
                                 't1.nii' == basename(directory).lower() ]
                    
                image_paths[f]['image']['original']['T1'] =  list_image[0]
                
                    
            if 'pre' in image_dir:
      
                list_image = [join(image_dir,directory)\
                             for directory in listdir(image_dir) if 'flair.' in basename(directory).lower()]
                    
                image_paths[f]['image']['pre'] = {}
      
                list_image = [join(image_dir,directory)\
                             for directory in listdir(image_dir) if 'flair.' in basename(directory).lower()]
                    
                image_paths[f]['image']['pre']['FLAIR'] =  list_image[0]
                
                list_image = [join(image_dir,directory)\
                             for directory in listdir(image_dir) if 't1.nii.gz' == basename(directory).lower() or\
                                 't1.nii' == basename(directory).lower() ]
                    
                image_paths[f]['image']['pre']['T1'] =  list_image[0]
               
        f += 1      
    return image_paths,f

def get_nifti_files(sessions_info,patters_dict, subject_path):
    """
    
    Parameters
    ----------
    sessions_info : TYPE
        DESCRIPTION.
    patters_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    session_dict : TYPE
        DESCRIPTION.

    """
    
    all_nifti_files = get_feature_paths(subject_path,['.nii','gz'])

    session_dict = {}
    
    for session in sessions_info.keys():
        
        #print(f"session {session}")
        session_name  = sessions_info[session]['session_name']
        #print(f"session_name {session_name}")
        session_date  = sessions_info[session]['session_date'] 
        #print(f"session_date {session_date}")

        session_dict[session] = {}
        session_dict[session]["session_name"] = session_name 
        session_dict[session]["session_date"] = session_date 
        
        session_dict[session]["found_flair"]  = 0
        
        flair_counts = 0
        found_flair  = 0
        found_pattern_list = []
        
        for nifti_file in all_nifti_files:
            
            nifti_basename = os.path.basename(nifti_file.lower())
            
            if not session_name.lower() in nifti_file.lower(): continue  
        
            print(nifti_basename)
        
            found_pattern = 0

            for image_pattern in patters_dict["flair"]["positive"]:
                
                if image_pattern.lower() in nifti_basename: 
                    found_pattern = 1

            if found_pattern:

                found_pattern_list.append(nifti_file)
                    
                    
        found_pattern_list_with_wrong_pattern_check = []            
                    
        for nifti_file in found_pattern_list:       
            nifti_basename = os.path.basename(nifti_file.lower())
            res = [ele for ele in patters_dict["flair"]["negative"] if(ele in nifti_basename)]
            
            if not res:

                flair_image_path = nifti_file
                print("positive")
                #print("subjet-"+str(subject_idx) + " "+os.path.basename(flair_image_path))
                flair_counts+=1

                print(flair_counts)

                session_dict[session]["found_flair"] = 1 

                #session_dict[session]["flair_paths"] = {}
                
                session_dict[session].setdefault("flair_counts", 0)  
                session_dict[session]["flair_counts"]  = flair_counts

                found_pattern_list_with_wrong_pattern_check.append(flair_image_path)
                
        session_dict[session].setdefault("flair_paths", found_pattern_list_with_wrong_pattern_check)  
                
                
        #print(session_dict[session]["flair_paths"])

        #session_dict[session]["flair_paths"][flair_counts] = flair_image_path 
        #print(session_dict[session]["flair_paths"][flair_counts])
 
    return session_dict


def create_bids_dir(bids_proscis,original_data_dictionary , participants_df,temp_dir = ""):
    """
    

    Parameters
    ----------
    bids_proscis_auto : TYPE
        DESCRIPTION.
    original_data_dictionary : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #temp_dir = '../data/temp/'
    #bids_proscis = bids_proscis_man
    
    #bids_proscis=bids_proscis_auto
    create_dir(bids_proscis) 
    if os.path.exists(bids_proscis):
        call(["rm","-r",join(bids_proscis)])
    
    create_dir(bids_proscis) 
    
    
    #relative_prepared_flair_dir      = relpath(prepared_flair_dir, dataset_directory)
    temp_bids_dir          = create_result_dir(temp_dir,"bids")
    
    participants_df.to_excel(bids_proscis+"/participants.xlsx")
    participants_df.to_excel(temp_bids_dir+"/participants.xlsx")

    for subject_name, sub_dict in original_data_dictionary.items():

        print(subject_name)

        sub_dir_name = join(bids_proscis,subject_name)
        #if not found_flair: continue

        for ses_name, ses_dict in sub_dict.items():

            print("       ", end='')
            print(f"SESSION {ses_name}")
            
            ses_dir_name                 = join(sub_dir_name,ses_name,"anat")
            
            for pattern_name,pattern_info in ses_dict.items():
                
                print(pattern_name)
                
                if not pattern_info: continue
            
                patter_dir_path             = join(ses_dir_name,pattern_name)
                
                create_dir(patter_dir_path)
                

                for image_index, image_path in enumerate(pattern_info['found_images']):

                    flair_name               = f"-anat_acq-2D_{pattern_name}_{image_index+1}.nii"

                    new_image_name_ses_name  =  subject_name+"-"+ses_name+flair_name
                    
                    new_image_path           = join(patter_dir_path,new_image_name_ses_name)
                    
                    copy_file(image_path, new_image_path,False)
                    
                    
                    rel_new_flair_name_ses_name =    relpath(new_image_path,bids_proscis)
                    print(f"create  {rel_new_flair_name_ses_name} ")

    copy_dir(bids_proscis, temp_bids_dir)
    

    
    
    #plot_tree(standard_proscis)
    print(f"save to standard_proscis directory {bids_proscis}")
    
    


def get_session_with_images(subject_path, session_pattern ="CSB[_-]TRIO_\d+_\d+",image_endings = ['.nii','gz'] ):
    """
    

    Parameters
    ----------
    subject_path : TYPE
        DESCRIPTION.
    pattern : TYPE, optional
        DESCRIPTION. The default is "CSB_TRIO_\d+_\d+".
    image_endings : TYPE, optional
        DESCRIPTION. The default is ['.nii','gz'].

    Returns
    -------
    session_count : TYPE
        DESCRIPTION.
    new_session : TYPE
        DESCRIPTION.

    """

    #all_nifti_files = get_feature_paths(subject_path,image_endings)
    
    #get only directories
    all_nifti_files = [roots + '/'  for roots,dirs,files in os.walk(subject_path)]

    only_session_directorys  = []
    
    for nifti_file in all_nifti_files:
        
        found  = re.search(".*\/CSB[_-]TRIO_\d+_\d+", nifti_file)
        if found:
            only_session_directorys.append(found[0])
            
            
    only_session_directorys_set = list(set(only_session_directorys))  
            
    found_csbs_list = []
    
    for nifti_file in all_nifti_files:
        relative_image_path      = relpath(nifti_file, subject_path)
        paths_list               = relative_image_path.split("/")
        
        for i in range(len(paths_list)):
            file_names = paths_list[i]
            found_csb  = re.search(session_pattern, file_names)
            
            if found_csb:
                found_csb_name = found_csb[0]
                if not found_csb_name in found_csbs_list:
                    found_csbs_list.append(found_csb_name)
                    
    session_names = {}
                  
    for found_csbs  in found_csbs_list:

        for only in only_session_directorys_set:
            
            #found_csbs = "CSB-TRIO_20100325_094425"
            
            csb_pattern  = '.+?(?='+found_csbs+")"
            
            found  = re.search(csb_pattern, only)
            
            if found:
                
                found_path = found[0]   +    found_csbs    
                
                print(found_path)
                
                        
                current_session_name = found_csbs
                
                time = current_session_name.split("_")[-1]
                date = current_session_name.split("_")[-2]
                
        
                hour     = int(time[:2])
                minute   = int(time[2:4])
                
                day      = int(date[6:])
                month    = int(date[4:6])
                year     = int(date[:4])
                
                imrt_date = datetime(year, month, day, hour, minute)
                
                timestamp = int(imrt_date.timestamp())
                
                
                if not timestamp in session_names.keys():
                
                    session_names[timestamp] = {}
                    session_names[timestamp]["found_path"] = found_path
                    session_names[timestamp]["found_csbs"] = found_csbs
                    session_names[timestamp]["imrt_date"]  = imrt_date


    sorted_session_names = collections.OrderedDict(sorted(session_names.items()))

            
    new_session = {}
    
    for sess_index, sess in enumerate(sorted_session_names):
        
        session_name = "ses-"+str(sess_index+1).zfill(2)

        new_session[session_name] = {}
        new_session[session_name]["session_name"] = sorted_session_names[sess]["found_csbs"]
        new_session[session_name]["session_date"] = sorted_session_names[sess]["imrt_date"]
        new_session[session_name]["session_path"] = sorted_session_names[sess]["found_path"]

    return  new_session



def get_images_from_session(session_path, patters_dict):
    
    session_dict = {}
    
    #directories_session  = [session_path+"/"+di for di in next(os.walk(session_path))[1]]
    
    only_session_directorys  = {}
    
    for directory in next(os.walk(session_path))[1]: 
        only_session_directorys[directory] = session_path+"/"+directory
        

    # for nifti_file in all_nifti_files: 
            
    #     until_last_occurance_pattern  = '.*\/'
    #     found  = re.search(until_last_occurance_pattern, nifti_file)

    #     if found:
    #         path_name = os.path.basename(os.path.normpath(found[0]))
    #         only_session_directorys[path_name] = found[0]  
            
    found_pattern_list= {}   

    real_found_pattern_list = {}     
            
    for pattern in patters_dict:
        
        
        found_pattern_list[pattern] = {}
        real_found_pattern_list[pattern]= {}
        
        real_found_pattern_list[pattern]["found"] = {}
        
        
        index_paths = 0
        
        found_pattern_list[pattern]["found"] =[]
        
        for key,info_p in only_session_directorys.items():
            
            found_pattern = 0

            for positive_image_pattern in patters_dict[pattern]["positive"]:
                
                if positive_image_pattern.lower() in key.lower(): 
                    
                    #print(key)
                    index_paths+=1
                    
                    
                    found_pattern = 1
                
                
            if found_pattern:
                found_pattern_list[pattern]["found"].append(key)
                
                    
        for found in found_pattern_list[pattern]["found"]:
            
            not_found_pattern = 0
            for negative_image_pattern in patters_dict[pattern]["negative"]:   
                if negative_image_pattern.lower() in found.lower():
    
                    not_found_pattern = 1
                    
            if not not_found_pattern:
                real_found_pattern_list[pattern]["found"]["name"] = found
                real_found_pattern_list[pattern]["found"]["path"] = only_session_directorys[found]
            

    return real_found_pattern_list 
            

    
def create_proscis_data_dict(proscis_path, number_of_subjects=0,patters_dict={}):

    #proscis_path = proscis_data_path_manual    
    #proscis_path = proscis_data_path_auto
    
    list_of_subject = get_all_dirs(proscis_path)
    
    participants_df =   pd.DataFrame()
    subject_idx = 0
    original_data_dictionary = {}
    
    for sub_idx,subject_path in enumerate(list_of_subject):
        
        #if count >=0: break 

        print("")
        
        df = pd.DataFrame( index=(0,))  

        subject_idx+=1
        
        print(f"subject_idx {subject_idx}")
        #print("")
        
        directory_number                                             = subject_path.split("/")[-1]
        subject_name                                                 = "sub-"+ str(subject_idx).zfill(2)
        
        
        original_data_dictionary[subject_name] = {}
        
        df["subject"]                                                = subject_name
        df["directory_number"]                                       = directory_number
        
        
        print(f"directory_number {directory_number}")
        sessions_info                                                  = get_session_with_images(subject_path,
                                                                        session_pattern ="CSB[_-]TRIO_\d+_\d+",
                                                                        image_endings = ['.nii','gz'] )
        
        
        if number_of_subjects:
            if  sub_idx + 1   >= number_of_subjects:break
        
        subject_dict = {}
        
        #if len(sessions_info) > 2:break
        #if subject_idx > 100 :break

        for session in sessions_info:

            original_data_dictionary[subject_name][session] = {}
            df["session"]  = session
            #if "02" in session: break
            #if count >=3: break 
            print("")
            print(session)
            subject_dict[session] = {}

            session_path  =  sessions_info[session]['session_path']
            session_date  =  sessions_info[session]['session_date'] 
            session_name  =  sessions_info[session]['session_name']
            
            found_pattern_dict  =  get_images_from_session(session_path, patters_dict)
            
            for pattern,info in found_pattern_dict.items():

                
                df["pattern"]          = pattern
                original_data_dictionary[subject_name][session][pattern] = {}
                
                if info['found']:
                    found_pattern_directory_name = info['found']['name']
                    found_pattern_directory = info['found']['path']
                    
                    all_nifti_files = get_feature_paths(found_pattern_directory,['nii','gz'])
                    all_nifti_files_dict = {}
                    found_pattern_list= {}
                    
                    
                    for edx,nifile in enumerate(all_nifti_files):

                        found_pattern_list[edx] = {}

                        #print(nifile)
                        all_nifti_files_dict[edx] = {}
                        all_nifti_files_dict[edx]["name"] = os.path.basename(nifile)
                        all_nifti_files_dict[edx]["path"] = nifile
                        
                        
                        found_pattern = 0
                                    
                        for positive_image_pattern in patters_dict[pattern]["positive"]:
                            if positive_image_pattern.lower() in all_nifti_files_dict[edx]["name"].lower(): 
                                
                                found_pattern = 1
                                
                        #print(found_pattern)
                                            
                        if found_pattern:
                            found_pattern_list[edx] = {}
                            found_pattern_list[edx]['name'] =all_nifti_files_dict[edx]["name"]
                            found_pattern_list[edx]['path'] =all_nifti_files_dict[edx]["path"]
    
                    real_found_pattern_list = []            

                    for key in found_pattern_list.keys():

                        not_found_pattern = 0
                        
                        
                        if found_pattern_list[key]:
                        
                            for negative_image_pattern in patters_dict[pattern]["negative"]:   
                                if negative_image_pattern.lower() in found_pattern_list[key]['name'].lower():
                    
                                    not_found_pattern = 1
                                    
                            if not not_found_pattern:
                                real_found_pattern_list.append( found_pattern_list[key]['path'])
                        
                        
                    if real_found_pattern_list:
                        df["found_images_count"] = len(real_found_pattern_list)
                        for idx, image_p in enumerate(real_found_pattern_list):
                            df["found_images_"+str(idx)]=  os.path.basename(image_p)
                            

                        participants_df  = pd.concat([participants_df , df])
                        
                        for idx, image_p in enumerate(real_found_pattern_list):
                            df["found_images_"+str(idx)]=  np.NaN
                        
                        

                        
                        original_data_dictionary[subject_name][session][pattern]["directory_number"]    =  directory_number
                        original_data_dictionary[subject_name][session][pattern]["found_images"]        =  real_found_pattern_list
                        original_data_dictionary[subject_name][session][pattern]["found_images_count"]  =  len(real_found_pattern_list)
                        
                    else:
                        df["found_images"]=''
                        df["found_images_count"] = 0
                        
                else:
                    df["found_images"]=''
                    df["found_images_count"] = 0
                    

                

    return participants_df,original_data_dictionary





            
# def get_niftii_info(session_dict,image_type = 'flair_paths' ):
#     """
    

#     Parameters
#     ----------
#     session_dict : TYPE
#         DESCRIPTION.
#     image_type : TYPE, optional
#         DESCRIPTION. The default is 'flair_paths'.

#     Returns
#     -------
#     niftii_dict : TYPE
#         DESCRIPTION.
#     TYPE
#         DESCRIPTION.
#     found_flair : TYPE
#         DESCRIPTION.
#     list_of_images : TYPE
#         DESCRIPTION.

#     """

#     niftii_dict = {}
#     found_flair = 0
#     list_of_images = []
    
#     for session_number,info in session_dict.items():
        
#         pass
        
#         if not image_type in info.keys():continue
        
#         for f, flair_path in enumerate(info[image_type]):
            
#             print(flair_path)
            
#             flair_path_nib = nib.load(flair_path)

#             n1_header = flair_path_nib.header
        
#             niftii_dict["dim"]       = n1_header["dim"]
#             niftii_dict["descrip"]  = n1_header["descrip"]
            
            
#             list_of_images.append(flair_path)
            
#             found_flair  = 1
            
#             break    
        
#     return niftii_dict, list(niftii_dict.keys()) ,found_flair,list_of_images