
import os
import subprocess
import nibabel as nib
import numpy as np
#from path_utils import create_dir

import matplotlib

def normalize_volume(image):
    """normalize 3D Volume image"""
    
    
    dim     = image.shape
    volume = image.reshape(-1, 1)
    
    rvolume = np.divide(volume - volume.min(axis=0), (volume.max(axis=0) - volume.min(axis=0)))*1
    rvolume = rvolume.reshape(dim)
    
    return rvolume


def MatplotlibClearMemory():
    #usedbackend = matplotlib.get_backend()
    #matplotlib.use('Cairo')
    allfignums = matplotlib.pyplot.get_fignums()
    for i in allfignums:
        fig = matplotlib.pyplot.figure(i)
        fig.clear()
        matplotlib.pyplot.close( fig )

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

def get_volume(path):
    
    if not os.path.isfile(path): return 0
    
    output_all_matter = subprocess.run(["fslstats",
                    path,
                    "-V"], 
                    capture_output=True)
            
    result_str = str(output_all_matter.stdout)
    all_matter_result = result_str.split(" ")

    return float(all_matter_result[1])


def make_threshholde(output_mask_path , thresholds_path,th):
    

    mask_d                                  = nib.load(output_mask_path)
    output_mask_array                       = mask_d.get_fdata()
    
    #np.unique(output_mask)

    output_mask_copy = np.copy(output_mask_array)
    threshhold_float = th / 100

    output_mask_copy[output_mask_copy >=threshhold_float] = 1
    output_mask_copy[output_mask_copy <threshhold_float] = 0

    # #th_str = str(th).split(".")[1]
    # output_mask = np.copy(output_mask_original)
    # output_mask[output_mask >=th] = 1
    # output_mask[output_mask <th] = 0
    
    th_str          =  str(th)
    th_name         =  f"bianca_output_thr.nii.gz"

    ni_img          =  nib.Nifti1Image(output_mask_copy, mask_d.affine)
    th_path         =  os.path.join(thresholds_path,th_name)
    
    nib.save(ni_img, th_path)
    

    return output_mask_copy,th_path


def create_threshold(output_mask_path,th):

    #create one threshhold
    mask_d                                  = nib.load(output_mask_path)
    output_mask_array                       = mask_d.get_fdata()
    
    #np.unique(output_mask)
    output_mask_bin_segmentation_array      = make_threshholde(output_mask_array ,th)
    
    #np.unique(output_mask_bin_segmentation)
    
    th_str          = str(th)
    th_name         =  f"bianca_output_thr_{th_str}.nii.gz"

    ni_img          =  nib.Nifti1Image(output_mask_bin_segmentation_array, mask_d.affine)
    th_path         =  os.path.join(os.path.dirname(output_mask_path),th_name)
    
    nib.save(ni_img, th_path)
    
    return th_path



def create_wm_distances(bianca_normal_cluster_path, csb_fluid_new_name_path ,threshold,minimum_clustersize,distance_to_ventrickle   = 10.0001,cohort_name=""):
    
    
    normal_image_dir = os.path.dirname(bianca_normal_cluster_path)
    
    
    bianca_out_bin_distvent_normal_cluster_path = f"{normal_image_dir}/bianca_out_bin_distvent_{minimum_clustersize}_th_{threshold}.nii.gz"
    
    if cohort_name:
        bianca_out_bin_distvent_normal_cluster_path = f"{normal_image_dir}/bianca_out_bin_distvent_{minimum_clustersize}_th_{threshold}_{cohort_name}.nii.gz"
    
    
    if os.path.isfile(bianca_out_bin_distvent_normal_cluster_path):
        os.system(f"distancemap -i {csb_fluid_new_name_path} -m {bianca_normal_cluster_path} -o {bianca_out_bin_distvent_normal_cluster_path}")
    
    
    deepwm_map_normal_path   = f"{normal_image_dir}/deepwm_map_{round(distance_to_ventrickle)}mm_th_{threshold}.nii.gz"
    
    perivent_map_normal_path = f"{normal_image_dir}/perivent_map_{round(distance_to_ventrickle)}mm_th_{threshold}.nii.gz" 
    
    
    if os.path.isfile(deepwm_map_normal_path) and os.path.isfile(deepwm_map_normal_path):
        
        if cohort_name:
            deepwm_map_normal_path   = f"{normal_image_dir}/deepwm_map_{round(distance_to_ventrickle)}mm_th_{threshold}_{cohort_name}.nii.gz"
            
            perivent_map_normal_path = f"{normal_image_dir}/perivent_map_{round(distance_to_ventrickle)}mm_th_{threshold}_{cohort_name}.nii.gz" 
    
        os.system(f"fslmaths {bianca_out_bin_distvent_normal_cluster_path} -thr {distance_to_ventrickle} -bin {deepwm_map_normal_path}")
        os.system(f"fslmaths {bianca_normal_cluster_path} -sub {deepwm_map_normal_path} {perivent_map_normal_path}")
        

    return deepwm_map_normal_path,perivent_map_normal_path


def create_analyse_bianca_results(bianca_normal_prob,
                                  flair_csb_new_fluidname_path,
                                  th = 99,
                                  distance_to_ventrickle = 10,
                                  cohort_name=""):

    thresh_normal_path          =  create_threshold(bianca_normal_prob,th)
    
    bianca_normal_cluster_path  =  perform_cluster_analysis(thresh_normal_path,cohort_name=cohort_name)
    
    deepwm_map_normal_path,perivent_map_normal_path  =    create_wm_distances(bianca_normal_cluster_path,
                                                          flair_csb_new_fluidname_path ,th,
                                                          distance_to_ventrickle   = distance_to_ventrickle)
    
    
    
    
    return deepwm_map_normal_path,perivent_map_normal_path,bianca_normal_cluster_path 

                
def perform_cluster_analysis(th_path, minimum_clustersize = 150,th=99, cohort_name = ""):
    
    normal_image_dir = os.path.dirname(th_path)
    
    
    if not  os.path.isfile(th_path):
        
        print(f"file {th_path} doesnt exists! ")
        return 0
    

    cluster_path = os.path.join(normal_image_dir,"cluster")
    
    if not os.path.isdir(cluster_path): os.mkdir(cluster_path)
    
    
    # Generate output path for bianca output
    bianca_normal_cluster_path = f"{cluster_path}/bianca_out_th_{th}_mincluster_{minimum_clustersize}.nii.gz"
    
    if cohort_name:
        bianca_normal_cluster_path = f"{cluster_path}/bianca_out_th_{th}_mincluster_{minimum_clustersize}_{cohort_name}.nii.gz"
    
    # Perform cluster analysis on thresholded image
    os.system(f"cluster --in={th_path} --thresh=0.05 \
    --oindex={bianca_normal_cluster_path} --connectivity=6 --no_table --minextent={minimum_clustersize}")
    
    # Convert output to binary mask
    os.system(f"fslmaths {bianca_normal_cluster_path} -bin {bianca_normal_cluster_path}")
    
    return bianca_normal_cluster_path


def create_confusion_matrix(bianca_normal_cluster_path,mask_path,confusion_processed):
    
    confusion_matrix = {}

    th = os.path.basename(bianca_normal_cluster_path).split(".")[0].split("_")[-3]
    #print(th)
    
    minclustersize = os.path.basename(bianca_normal_cluster_path).split(".")[0].split("_")[-1]
    #print(minclustersize)
    
    confusion_image = os.path.join(confusion_processed,f"confusion_image")
    
    th_path = os.path.join(confusion_image,f"th_{th}")

    minclustersize_path = os.path.join(confusion_image,f"minclustersize_{minclustersize}")
    
    if not os.path.isdir(minclustersize_path): os.mkdir(minclustersize_path)

    
    bianca_out_nib                      = nib.load(bianca_normal_cluster_path)
    bianca_out_ms                       = bianca_out_nib.get_fdata()  

    ground_truth                        = nib.load(mask_path).get_fdata()  
    
    
    # bianca_out_ms.shape
    
    # ground_truth.shape
    
   
    tp = ((bianca_out_ms == 1) & (ground_truth == 1)).astype(np.float64) 
    tn = ((bianca_out_ms == 0) & (ground_truth == 0)).astype(np.float64) 
    fp = ((bianca_out_ms == 1) & (ground_truth == 0)).astype(np.float64) 
    fn = ((bianca_out_ms == 0) & (ground_truth == 1)).astype(np.float64) 


    tp_path = os.path.join(minclustersize_path,f"tp_segmentation.nii")
    fp_path = os.path.join(minclustersize_path,f"fp_segmentation.nii")
    fn_path = os.path.join(minclustersize_path,f"fn_segmentation.nii")
    
         
    tp_nib = nib.Nifti1Image(tp, bianca_out_nib.affine)
    nib.save(tp_nib, tp_path)

    fp_nib = nib.Nifti1Image(fp, bianca_out_nib.affine)
    nib.save(fp_nib, fp_path)

    fn_nib = nib.Nifti1Image(fn, bianca_out_nib.affine)
    nib.save(fn_nib, fn_path)

    tp_value = np.sum(tp)
    fp_value = np.sum(fp)
    fn_value = np.sum(fn)
    
    dice_score_cl     = round((2*(tp_value)) / (2*(tp_value)+(fp_value)+(fn_value)) ,2)
    
    #print("")
    #print(f"min clustersize: {minclustersize}")
    
    #print(f"dice_score_cl {dice_score_cl}")

    sensitivity_cl    = round( (tp_value) / ((tp_value) + (fn_value)),2)
    
    #print(f"dice_score_cl {dice_score_cl}")
    #print(f"sensitivity_cl {sensitivity_cl}")
    
    confusion_matrix["tp"] = {}
    confusion_matrix["tp"]["value"] = tp_value
    confusion_matrix["tp"]["path"] = tp_path
    
    confusion_matrix["fp"] = {}
    confusion_matrix["fp"]["value"] = fp_value
    confusion_matrix["fp"]["path"]  = fp_path

                    
    confusion_matrix["fn"] = {}
    confusion_matrix["fn"]["value"] = fn_value
    confusion_matrix["fn"]["path"]  = fn_path
    
    return dice_score_cl,confusion_matrix
