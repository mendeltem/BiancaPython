#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import configparser

import nibabel as nib
import numpy as np
import subprocess
from pathlib import Path
import multiprocessing as mp


#load personal libraries

#pip install scikit-image

import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 10

matplotlib.use('Agg')
#Use TkAgg backend because it works better for some reason:
plt.ioff()

from libraries.bianca_post_process_lib import get_volume,make_threshholde
from libraries.bianca_post_process_lib import create_wm_distances,perform_cluster_analysis
from libraries.bianca_post_process_lib import MatplotlibClearMemory,normalize_volume



config_dict = configparser.ConfigParser()
config_dict.read('param.ini')   

#standard images
prepared_bianca                       =     config_dict['STANDARD-PATHS']["standard_mask"]
standard_space_flair                  =     config_dict['STANDARD-PATHS']["standard_space_flair"]
standard_ventrikel                  =     config_dict['STANDARD-PATHS']["standard_space_ventrikel"]

#loading original datsets
label_datasets_list                   =   [config_dict['LABEL-DATA-SET-PATHS'][dset] for dset in list(config_dict['LABEL-DATA-SET-PATHS'])]

#loading original datsets
cohort_studies_list                   =   [config_dict['COHORT-DATA-SET-PATHS'][dset] for dset in list(config_dict['COHORT-DATA-SET-PATHS'])]

processed_paths_dict                  =   {dset: config_dict['PROCESSED-SET-PATHS'][dset] for dset in config_dict['PROCESSED-SET-PATHS']}
found_set                             =    [(dset,os.path.isdir(path)) for dset,path in processed_paths_dict.items()]

print(found_set)

datasets_prepared_path                =   processed_paths_dict['datasets_prepared']
if os.path.isdir(datasets_prepared_path):
    print(f"found {datasets_prepared_path}")
    
output_dir                =   Path(processed_paths_dict['output_directory'])
if os.path.isdir(output_dir):
    print(f"found {output_dir}")
    

    
cohort_datasets_dir = Path(datasets_prepared_path) / "cohort_datasets"
cohort_datasets_dir.mkdir(parents=True, exist_ok=True)

data_location_table_path = cohort_datasets_dir / "data_original_location.xlsx"


cohort_output_datasets_dir = Path(output_dir) / "cohort_datasets"
cohort_output_datasets_dir.mkdir(parents=True, exist_ok=True)

result_set = cohort_output_datasets_dir / "result_1"
result_set.mkdir(parents=True, exist_ok=True)

bianca_result_set = result_set / "bianca"
bianca_result_set.mkdir(parents=True, exist_ok=True)


bianca_result_set.mkdir(parents=True, exist_ok=True)
bianca_result_table_path  = bianca_result_set/"bianca_result.xlsx"


analyse_result_set = result_set / "postprocessing"
analyse_result_set.mkdir(parents=True, exist_ok=True)


analyse_result_set = result_set / "postprocessing"
analyse_result_set.mkdir(parents=True, exist_ok=True)

pp_result_table_path  = analyse_result_set/"post_processing_result.xlsx"


if os.path.isfile(bianca_result_table_path):
    print(f"Found {bianca_result_table_path}")
    
    bianca_result_table_df = pd.read_excel(bianca_result_table_path, index_col=0)

    subject_list = list(bianca_result_table_df["subject"])
else:
    raise ValueError("File not found at specified path.")



def run_preprocessing(arg):
    
    #arg = df
    
    subject, dataframe,data_dict = arg
    
    
    pp_result_table_path = data_dict["pp_result_table_path"]
    thresholds_list      = data_dict["thresholds_list"]
    minimum_clustersizes = data_dict["minimum_clustersizes"]
    standard_ventrikel   = data_dict["standard_ventrikel"]
    image_controll_bool  = data_dict["image_controll_bool"]
    
    
    subject_path = Path(pp_result_table_path) / subject
    
    
    subject_row   = dataframe[dataframe["subject"] == subject]
    patient_id    = subject_row["original_patien_name"]
    
    
    subject_row   = subject_row.reset_index(drop=True)
    
    #print(subject_row)
    #print(data_dict)
    
    list_of_results = []
    
    
    pp_subject_result = pd.DataFrame()
    
    mprage_volume                      = subject_row["mprage_volume"]

    
    null_image_path                    = subject_row["null_image_path"]
    controll_image_path                = subject_row["controll_image_path"]
    
    # pp_result["null_image_path"]       = null_image_path
    # pp_result["controll_image_path"]   = controll_image_path
    
    flair_csb_clustered_path_mni_path  = subject_row["flair_csb_clustered_path_mni_path"].values[0]
    controll_image_mni_path            = subject_row["controll_image_mni_path"].values[0]
    null_image_mni_path                = subject_row["null_image_mni_path"].values[0]
    
    
    controll_bianca_mni_ven_corrected    = subject_row["controll_bianca_mni_ven_corrected"].values[0]
    null_bianca_mni_ven_corrected      = subject_row["null_bianca_mni_ven_corrected"].values[0]
    null_mask_mni_path                 = subject_row["null_mask_mni_path"].values[0]
    
    
    controll_dir   =  Path(subject_path) / "controll"
    null_dir   =  Path(subject_path) / "null"
    
    
    images_dir   =  Path(subject_path) / "images"
    
    
    for threshold in thresholds_list:
        
        th_controll_dir   =  Path(controll_dir) / f"th_{threshold}"
        th_controll_dir.mkdir(parents=True, exist_ok=True)
        
        th_null_dir   =  Path(null_dir) / f"th_{threshold}"
        th_null_dir.mkdir(parents=True, exist_ok=True)
        
        output_mask_normal,th_normal_path = make_threshholde(controll_bianca_mni_ven_corrected , th_controll_dir,threshold)
        output_mask_null,th_null_path = make_threshholde(null_bianca_mni_ven_corrected , th_null_dir,threshold)
        
        for minimum_clustersize in minimum_clustersizes[:]:
            
            # mc_controll_path = Path(th_controll_dir) / f"th_{minimum_clustersize}"
            # mc_controll_path.mkdir(parents=True, exist_ok=True)
            # mc_null_path = Path(th_controll_dir) / f"th_{minimum_clustersize}"
            # mc_null_path.mkdir(parents=True, exist_ok=True)
            
            
            
            bianca_normal_cluster_path  =  perform_cluster_analysis(th_normal_path,minimum_clustersize,threshold)
            bianca_null_cluster_path  =  perform_cluster_analysis(th_null_path,minimum_clustersize,threshold)
            
            
            
            deepwm_map_normal_path,perivent_map_normal_path  =    create_wm_distances(bianca_normal_cluster_path,
                                                      flair_csb_clustered_path_mni_path ,
                                                      threshold,
                                                      minimum_clustersize )
            
            

            deepwm_map_null_path,perivent_map_null_path  =    create_wm_distances(bianca_null_cluster_path,
                                                                  flair_csb_clustered_path_mni_path ,
                                                                  threshold,
                                                                  minimum_clustersize )
                

            normal_mincluster_volume            =  np.round(get_volume(bianca_normal_cluster_path))
            normal_deepwm_map_10mm_volume       =  np.round(get_volume(deepwm_map_normal_path))
            normal_perivent_map_10mm_volume     =  np.round(get_volume(perivent_map_normal_path))
                        
            null_mincluster_volume              =  np.round(get_volume(bianca_null_cluster_path))
            null_deepwm_map_10mm_volume         =  np.round(get_volume(deepwm_map_null_path))
            null_perivent_map_10mm_volume       =  np.round(get_volume(perivent_map_null_path))   
            
            pp_result = pd.DataFrame(index=(0,))
            
            pp_result["original_patien_name"]  = subject_row["original_patien_name"]
            pp_result["subject"]               = subject
            pp_result["clustersize"]           = minimum_clustersize
            pp_result["threshold"]             = threshold
            
            pp_result["mprage_volume"]         = mprage_volume
            
            pp_result["controll_volume"]       = normal_mincluster_volume/mprage_volume
            pp_result["controll_per_volume"]   = normal_perivent_map_10mm_volume/mprage_volume
            pp_result["controll_deep_volume"]  = normal_deepwm_map_10mm_volume/mprage_volume
            
            pp_result["null_volume"]           = null_mincluster_volume/mprage_volume
            pp_result["per_null_volume"]       = null_perivent_map_10mm_volume/mprage_volume
            pp_result["deep_null_volume"]      = null_deepwm_map_10mm_volume/mprage_volume
            
            #if not image_controll_bool: continue
            
            #create images to controll sanity
            
            # #input images
            # controll_image_array       = normalize_volume(nib.load(controll_image_mni_path).get_fdata()) 
            # null_image_array           = normalize_volume(nib.load(null_image_mni_path).get_fdata() )
            # flair_csb_clustered_array  = normalize_volume(nib.load(flair_csb_clustered_path_mni_path).get_fdata())
            # mask_array                 = nib.load(null_mask_mni_path).get_fdata()
            # ventrikel_array            = nib.load(standard_ventrikel).get_fdata()
            
            # #controll resultes
            bianca_normal_cluster_array   = nib.load(bianca_normal_cluster_path).get_fdata()
            # deepwm_map_normal_array       = nib.load(deepwm_map_normal_path).get_fdata()
            # perivent_map_normal_array     = nib.load(perivent_map_normal_path).get_fdata()
            
            # #null resultes
            bianca_null_cluster_array     = nib.load(bianca_null_cluster_path).get_fdata()
            # deepwm_map_null_array         = nib.load(deepwm_map_null_path).get_fdata()
            # perivent_map_null_array       = nib.load(perivent_map_null_path).get_fdata()
            
            # test_image_th_dir = images_dir / str(threshold)
            # test_image_dir = test_image_th_dir/ str(minimum_clustersize)
            
            # test_image_dir.mkdir(parents=True, exist_ok=True)
            
            # slizes = controll_image_array.shape[2]
            
            
            no_nu = bianca_normal_cluster_array  - bianca_null_cluster_array
            green  = int(np.sum( no_nu[no_nu > 0]))
                
            nu_no = bianca_null_cluster_array  - bianca_normal_cluster_array
            red    = int( np.sum(nu_no[nu_no > 0]))
            
            yellow =  int(np.sum(np.logical_and(bianca_normal_cluster_array,bianca_null_cluster_array)))
            
            full = green+red+yellow 
            
            
            if full:
                green_percent = np.round((green / full) * 100,2)
                red_percent = np.round((red / full) * 100,2)
                yellow_percent = np.round((yellow / full) * 100,2)
            else:
                green_percent = 0
                red_percent = 0
                yellow_percent = 0
                
            pp_result["removed"]           = green_percent
            pp_result["gained"]            = red_percent
            pp_result["no_change"]         = yellow_percent
            

            # image_template_2d      = np.zeros(bianca_normal_cluster_array[:,:,0].shape+(3,)) 
    
            # for slize in range(slizes):
                
            #     bianca_summ = np.sum(bianca_normal_cluster_array[:,:,slize])

            #     if  bianca_summ <= 50  or np.sum(mask_array) ==0: continue   
            
            #     #print(bianca_summ)
            
            
            #     fig, axes =  plt.subplots(2,4,constrained_layout = True)
                
            #     sub_title = f"patient_name : {subject}    ID : {patient_id}\nthreshold : {threshold}  Min Cluster  {minimum_clustersize}" 
            #     fig.suptitle(f" {sub_title}", fontsize=40)  
                
            #     fig.set_figheight(20)
            #     fig.set_figwidth(25)
                           
            #     for s in range(axes.shape[0]):
            #         for a in range(axes.shape[1]):
            #             axes[s][a].get_xaxis().set_visible(False)
            #             axes[s][a].get_yaxis().set_visible(False)
                        
                
            #     #Input image
                
            #     image_template_2d[:,:,0]       =    controll_image_array[:,:,slize]
            #     image_template_2d[:,:,1]       =    controll_image_array[:,:,slize]
            #     image_template_2d[:,:,2]       =    controll_image_array[:,:,slize]
            #     axes[0][0].imshow(image_template_2d)
            #     axes[0][0].set_title(f'controll_image', fontsize=25)   
                
            #     image_template_2d[:,:,0]       =    null_image_array[:,:,slize]
            #     image_template_2d[:,:,1]       =    null_image_array[:,:,slize]
            #     image_template_2d[:,:,2]       =    null_image_array[:,:,slize]
            #     axes[1][0].imshow(image_template_2d)
            #     axes[1][0].set_title(f'null_image', fontsize=25)   
                
                
            #     #Input BIANCA Result image
                
            #     image_template_2d[:,:,0]       =    ventrikel_array[:,:,slize]
            #     image_template_2d[:,:,1]       =    bianca_normal_cluster_array[:,:,slize]
            #     image_template_2d[:,:,2]       =    controll_image_array[:,:,slize]
            #     axes[0][1].imshow(image_template_2d)
            #     axes[0][1].set_title(f'Bianca Controll ', fontsize=25) 
                
                
                
            #     image_template_2d[:,:,0]       =    ventrikel_array[:,:,slize]
            #     image_template_2d[:,:,1]       =    bianca_null_cluster_array[:,:,slize]
            #     image_template_2d[:,:,2]       =    controll_image_array[:,:,slize]
            #     axes[1][1].imshow(image_template_2d)
            #     axes[1][1].set_title(f'Bianca Null ', fontsize=25)  
                
                
            #     #periventrikel vs deep deepwm_map_normal_array
            #     image_template_2d[:,:,0]       =    flair_csb_clustered_array[:,:,slize]
            #     image_template_2d[:,:,1]       =    perivent_map_normal_array[:,:,slize]
            #     image_template_2d[:,:,2]       =    deepwm_map_normal_array[:,:,slize]
            #     axes[0][2].imshow(image_template_2d)
            #     axes[0][2].set_title(f'Controll Image: \n Perivascular=Green: {normal_perivent_map_10mm_volume}   \n Deepwm=Blue: {normal_deepwm_map_10mm_volume} ', fontsize=25) 
                
                
            #     image_template_2d[:,:,0]       =    flair_csb_clustered_array[:,:,slize]
            #     image_template_2d[:,:,1]       =    perivent_map_null_array[:,:,slize]
            #     image_template_2d[:,:,2]       =    deepwm_map_null_array[:,:,slize]
            #     axes[1][2].imshow(image_template_2d)
            #     axes[1][2].set_title(f'Nulled Image: \n Perivascular=Green: {null_perivent_map_10mm_volume}  \n  Deepwm=Blue: {null_deepwm_map_10mm_volume} ', fontsize=25) 
                
                
            #     no_nu = bianca_normal_cluster_array[:,:,slize]  - bianca_null_cluster_array[:,:,slize]
            #     nu_no = bianca_null_cluster_array[:,:,slize]  - bianca_normal_cluster_array[:,:,slize]
                
            #     both_vol =  int(np.sum(np.logical_and(bianca_normal_cluster_array[:,:,slize],bianca_null_cluster_array[:,:,slize])))
            #     no_nu_vol = int(np.sum( no_nu[no_nu > 0]))
            #     nu_no_vol = int( np.sum(nu_no[nu_no > 0]))
                
            
            #     axes[0][3].set_title(f'Green: removed {green_percent}% \nRed gained {red_percent}% \nYellow no change: {yellow_percent}%', fontsize=25)    
                
            #     image_template_2d[:,:,0] = bianca_null_cluster_array[:,:,slize]
            #     image_template_2d[:,:,1] = bianca_normal_cluster_array[:,:,slize]
            #     image_template_2d[:,:,2] = np.zeros(bianca_normal_cluster_array[:,:,slize].shape)
            
            #     axes[0][3].imshow(image_template_2d)
            #     axes[1][3].set_title(f'Bianca Null and mask: \n mask: blue  \n Union mask  with wm: white ', fontsize=25)
                
            #     image_template_2d[:,:,0] = bianca_null_cluster_array[:,:,slize]
            #     image_template_2d[:,:,1] = bianca_null_cluster_array[:,:,slize]
            #     image_template_2d[:,:,2] = mask_array[:,:,slize]
            
            #     axes[1][3].imshow(image_template_2d)
                

            #     jpg_file = test_image_dir /f"{slize}.jpg"
                
            #     print(jpg_file)
                
            #     plt.savefig(jpg_file, 
            #               dpi=50,
            #               bbox_inches ="tight",
            #               pad_inches=1)
                
                
            #     #plt.show()
            #     plt.close(fig)
            #     #fig.clear()
            #     plt.clf() 
            #     plt.close('all')
            #     MatplotlibClearMemory()

            pp_subject_result  = pd.concat([pp_subject_result,pp_result])
            
        
    return pp_subject_result


#debug
subject_list = subject_list [:]


num_cpus = mp.cpu_count()
print("Number of CPUs: ", num_cpus)

num_processes = num_cpus

# Calculate the size of each sub-list
sub_list_size = len(subject_list) // num_processes

# Split the original list into sub-lists
sub_lists = [subject_list[i:i+num_processes] for i in range(0, len(subject_list), num_processes)]


print(f"sub_lists len {len(sub_lists)}")


postprocessing_results = pd.DataFrame()

thresholds_list = [90,95,97]
#thresholds_list = [95]
minimum_clustersizes = [100, 150]
#minimum_clustersizes = [100]


print(list(bianca_result_table_df.columns))


image_controll_bool = 0

# if image_controll_bool:
#     run_multi_process = 0
# else:
#     run_multi_process = 1
run_multi_process = 1

data_dict = {"pp_result_table_path": analyse_result_set,
             "thresholds_list": thresholds_list,
             "minimum_clustersizes":minimum_clustersizes,
             "standard_ventrikel" : standard_ventrikel,
             "image_controll_bool": image_controll_bool
             }


# Print out each sub-list
for i, sub_list in enumerate(sub_lists):

    #create pool for multiprocess
    pool = mp.Pool(processes=num_processes)
    
    # create an empty list to store the concatenated dataframes
    concatenated_arg = []
    
    
    # loop through the values and concatenate the dataframes for each value
    for sub in sub_list:
        concatenated_arg.append((sub,bianca_result_table_df,data_dict))
        
        
    # if not run_multi_process:
    
    #     result  = []
    #     for arg in concatenated_arg:
    #         print(arg)
    #         part_result = run_preprocessing(arg)
            
    #         result += [part_result]
        
    # else:
    result = pool.map(run_preprocessing, concatenated_arg)
    
    print("result")
    print(result)
    print("")

    # if  result.empty:
    #     print("dataframe empty continue")
        
    #     continue
    # if not result: continue
    
    # if  result.shape[0] == 1:
    #     postprocessing_results = pd.concat([postprocessing_results, result], ignore_index=True)
    # elif result.shape[0] > 1:
    postprocessing_results = pd.concat([postprocessing_results, pd.concat(result)], ignore_index=True)
    #else: continue
    
    print(postprocessing_results.info())
    print(postprocessing_results.head())
    
    # if image_controll_bool:continue
    

postprocessing_results.to_excel(pp_result_table_path)
