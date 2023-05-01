#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:43:51 2023

@author: temuuleu
"""


import tkinter as tk
from tkinter import ttk
from PIL import  Image, ImageTk
import os
os.chdir("/home/temuuleu/CSB_NeuroRad/temuuleu/Projekts/BIANCA/")
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from bianca_gui.config.config import Config
#from bianca_shell.bianca_shell import run_bianca

from bianca_gui.create_master_file import create_masterfile_for_train


# Function to handle file selection
def select_mat(Input_Mat,bianca_input_param_dict):

        cfg = Config()
        initial_directory = cfg.data_test_dir_path   
        image_file = filedialog.askopenfile(initialdir=initial_directory)
        
        image_file_path = image_file.name
        print(f"image_file {image_file_path}")
        
        # Clear existing items in the Listbox
        Input_Mat.delete(0, tk.END)
        Input_Mat.insert(tk.END, os.path.basename(image_file_path))
        
        bianca_input_param_dict['input_matrix'] = image_file_path
        
        
     
# Function to handle file selection
def select_image(Input_Image,bianca_input_param_dict):
     # Get the selected indices
     cfg = Config()
     initial_directory = cfg.data_test_dir_path   
     image_file = filedialog.askopenfile(initialdir=initial_directory)
     
     image_file_path = image_file.name
     print(f"image_file {image_file_path}")
     
     # Clear existing items in the Listbox
     Input_Image.delete(0, tk.END)
     Input_Image.insert(tk.END, os.path.basename(image_file_path))
     
     bianca_input_param_dict['input_FLAIR_image'] = image_file_path
     
     
     
     
# Function to handle file selection
def select_Masterfile(Input_MasterFile,bianca_input_param_dict):
     # Get the selected indices
     
     cfg = Config()
     initial_directory = cfg.masterfile_dir_path   
     image_file = filedialog.askopenfile(initialdir=initial_directory)
     

     
     image_file_path = image_file.name
     print(f"masterfile_file {image_file_path}")
     
     # Clear existing items in the Listbox
     Input_MasterFile.delete(0, tk.END)
     Input_MasterFile.insert(tk.END, os.path.basename(image_file_path))
     
     bianca_input_param_dict['input_masterfile'] = image_file_path
     
     
     
  
# Function to handle file selection
def select_Out_dir(Output_dir,bianca_input_param_dict):
     # Get the selected indices
     cfg = Config()
     initial_directory = cfg.data_out_test_dir_path   
     image_dir = filedialog.askopenfile(initialdir=initial_directory)
     
     image_file_path = image_dir
     print(f"oud_dir {image_dir}")
     
     # Clear existing items in the Listbox
     Output_dir.delete(0, tk.END)
     Output_dir.insert(tk.END, os.path.basename(image_dir))
     
     bianca_input_param_dict['out_put'] = image_dir
     
     
     
     

     
def run_bianca_shell(bianca_input_param_dict):
    
    input_FLAIR_image    = bianca_input_param_dict['input_FLAIR_image']
    input_T1_image       = bianca_input_param_dict['input_T1_image']
    
    input_matrix         = bianca_input_param_dict['input_matrix']
    input_masterfile     = bianca_input_param_dict['input_masterfile']
    
    out_put              = bianca_input_param_dict['out_put']
    
    out_put_file_name    = "bianca_out.nii.gz"
    bianca_output        = os.path.join(out_put,out_put_file_name)
    
    
    print(f"input image {input_FLAIR_image}")
    print(f"input matrix {input_matrix}")
    print(f"input masterfile {input_masterfile}")

    print("run bianca")
    
    
    
    run_bianca_cmd = f"run_bianca -image={input_FLAIR_image} -mni={input_matrix} -masterfile={input_masterfile} -output={bianca_output}"
    
    if input_T1_image:
        run_bianca_cmd = f"run_bianca -image={input_FLAIR_image} -mni={input_matrix} -masterfile={input_masterfile} -output={bianca_output}"
        
    
    print(run_bianca_cmd)
    
    os.system(f"{run_bianca_cmd}")
    
    if os.path.isfile(bianca_output):
        print("bianca success")
    
    print("end")
    
    
    
   
    
cfg = Config()
master_file_dir   = cfg.masterfile_dir_path

train_bids_dir    = cfg.train_bids_dir
        

train_dir_list =[ os.path.join(train_bids_dir, d ) for d in os.listdir(train_bids_dir)]


for bids_dir in train_dir_list:
    create_masterfile_for_train(bids_dir, master_file_dir)
    create_masterfile_for_train(bids_dir, master_file_dir,1)


    # def update_listboxes():
    #     # Clear the listboxes
    #     list_master_files_t1.delete(0, tk.END)
    #     list_master_files_not_t1.delete(0, tk.END)
        
    #     # Add file names to the Listboxes based on the value of t1_bool
    #     if t1_bool.get():
    #         # Only show files containing "T1"
    #         for subject in [d for d in os.listdir(cfg.masterfile_dir_path) if "T1" in d]:
    #             list_master_files_t1.insert(tk.END, os.path.basename(os.path.join(cfg.masterfile_dir_path, subject)))
    #         # Pack the listbox containing files with "T1" in their name
    #         list_master_files_t1.pack(side="left", padx=(10, 0), pady=5)
            
    #         # Hide the listbox containing files not containing "T1" in their name
    #         list_master_files_not_t1.pack_forget()
    #     else:
    #         # Only show files not containing "T1"
    #         for subject in [d for d in os.listdir(cfg.masterfile_dir_path) if "T1" not in d]:
    #             list_master_files_not_t1.insert(tk.END, os.path.basename(os.path.join(cfg.masterfile_dir_path, subject)))
    #         # Pack the listbox containing files not containing "T1" in their name
    #         list_master_files_not_t1.pack(side="left", padx=(10, 0), pady=5)
            
    #         # Hide the listbox containing files with "T1" in their name
    #         list_master_files_t1.pack_forget()

global bianca_input_param_dict
bianca_input_param_dict = {
    "input_FLAIR_image" :      "",
    "input_T1_image" :      "",
    "input_matrix" :     "",
    "input_masterfile":  "",
    "out_put":           f"{cfg.data_out_test_dir_path}"}

def main ():
    
    def get_selected_files(Input_MasterFile,bianca_input_param_dict):
        selection_indices = list_master_files_not_t1.curselection()
        selected_files = [list_master_files_not_t1.get(index) for index in selection_indices]
        
        print("Selected files:", selected_files)
        
        Input_MasterFile.delete(0, tk.END)
        Input_MasterFile.insert(tk.END, os.path.basename(selected_files[0]))
        
        bianca_input_param_dict['input_masterfile'] = selected_files[0]
        

        
    def get_selected_files_T1(Input_MasterFile,bianca_input_param_dict):
        
        selection_indices = list_master_files_t1.curselection()
        selected_files = [list_master_files_t1.get(index) for index in selection_indices]
        
        print("Selected files:", selected_files)
        Input_MasterFile.delete(0, tk.END)
        Input_MasterFile.insert(tk.END, os.path.basename(selected_files[0]))
        
        bianca_input_param_dict['input_masterfile'] = selected_files[0]
        
        
    
    logo_image_path  = "bianca_gui/gui_images/fsl_logo.png"
    width  = 400
    
    root = tk.Tk()
    root.title("BIANCA")
    #root.geometry("400x300")
    
    style = ttk.Style()
    style.theme_use("clam")
    
    frame = tk.Frame(root, bg="gray")

    # Create the top area
    top_area = tk.Frame(frame, bg="Aqua", height=150,width=width)

    # Create the left part of the top area
    logo_image = tk.PhotoImage(file=logo_image_path)
    logo_label = tk.Label(top_area, image=logo_image, bg="gray")
    
    # Create the top area
    master_1_area = tk.Frame(top_area, bg="Lime", height=150,width=width)
    
    master_1_area_label = ttk.Label(master_1_area, text="Masterfiles With T1 Added",padding=5)
    
    print("master_1_area_label:")
    for item in master_1_area_label.keys():
        print(item, ": ", master_1_area_label[item])
    
    
    # Create two listboxes
    list_master_files_t1 = tk.Listbox(master_1_area, selectmode=tk.SINGLE, height=7, width=80)

    for subject in [d for d in os.listdir(cfg.masterfile_dir_path) if "T1" in d]:
        list_master_files_t1.insert(tk.END, os.path.basename(os.path.join(cfg.masterfile_dir_path, subject)))
        
    select_masterfile_t1_button = tk.Button(master_1_area, text="Selected Master T1 File", command= lambda: get_selected_files_T1(Input_MasterFile,bianca_input_param_dict))
    
    # Create the top area
    master_2_area = tk.Frame(top_area, bg="Lime", height=150,width=width)
    master_2_area_label = ttk.Label(master_2_area, text="Masterfiles with T2 Only",padding=5)
    
    list_master_files_not_t1 = tk.Listbox(master_2_area, selectmode=tk.SINGLE, height=7, width=80)

    for subject in [d for d in os.listdir(cfg.masterfile_dir_path) if not "T1" in d]:
        list_master_files_not_t1.insert(tk.END, os.path.basename(os.path.join(cfg.masterfile_dir_path, subject)))
    
    select_masterfile_button = tk.Button(master_2_area, text="Selected Master File", command= lambda: get_selected_files(Input_MasterFile,bianca_input_param_dict))
    


    # Create a BooleanVar variable

    # Create the top area
    middle_area = tk.Frame(frame, bg="Blue", height=70,width=width)
    
    # Create the top area
    subject_1_area = tk.Frame(middle_area, bg="Lime", height=70,width=width)
    list_subject_files = tk.Listbox(subject_1_area, selectmode=tk.MULTIPLE, height=15)
    subject_2_area = tk.Frame(middle_area, bg="Gold", height=70,width=width)
    

    # Create the middle area
    middle_area_2 = tk.Frame(frame, bg="Brown", height=200,width=width)   
    input_1_area = tk.Frame(middle_area_2, bg="Magenta", height=200,width=width)
    input_2_area = tk.Frame(middle_area_2, bg="Navy Blue", height=200,width=width)
    input_3_area = tk.Frame(middle_area_2, bg="Blue", height=200,width=600)
    
    
    # Create a label in the new window
    Input_MasterFile    = tk.Entry(input_3_area,text="Input Masterfile:")
    Input_Flair_Image    = tk.Entry(input_3_area,text="Input Flair Image:", 
                              width=25) #Entry.grid(row=0, column=0, padx=1, pady=1) 
    
    Input_T1_Image    = tk.Entry(input_3_area,text="Input T1 Image:", 
                              width=25) #Entry.grid(row=0, column=0, padx=1, pady=1) 
    
    Input_Mat    = tk.Entry(input_3_area,text="Input Matrix:", 
                              width=25) #Entry.grid(row=0, column=0, padx=1, pady=1) 
    
    Output_dir    = tk.Entry(input_3_area,text="Output Dir:", 
                              width=25) #Entry.grid(row=0, column=0, padx=1, pady=1) 
    
    
    select_masterinput_button = tk.Button(input_3_area, text="Selected Master File", command= lambda: select_Masterfile(Input_MasterFile,bianca_input_param_dict))
    select_out_button = tk.Button(input_3_area, text="Select Output Directory", command= lambda: select_Out_dir(Output_dir,bianca_input_param_dict))
    select_Flair_image_button = tk.Button(input_3_area, text="Select Flair Input Image", command= lambda: select_image(Input_Flair_Image,bianca_input_param_dict))
    
    select_T1_image_button = tk.Button(input_3_area, text="Select T1 Input Image", command= lambda: select_image(Input_T1_Image,bianca_input_param_dict))
    select_affine_button = tk.Button(input_3_area, text="Select Input Affine Matrix", command= lambda: select_mat(Input_Mat,bianca_input_param_dict))
    

    
    run_bianca_button = tk.Button(input_3_area, text="Run Bianca", command= lambda: run_bianca_shell(bianca_input_param_dict))
    
    
    
    # Create the top area
    bottom_area = tk.Frame(frame, bg="Cyan", height=150,width=width)

    frame.pack(expand=True, fill="both")
    
    #
    top_area.grid(row=0, column=0, sticky="nsew")
    middle_area.grid(row=1, column=0, sticky="nsew")

    
    middle_area_2.grid(row=2, column=0, sticky="nsew")
    bottom_area.grid(row=3, column=0, sticky="nsew")
    
    #top_area
    logo_label.grid(row=0, column=0, sticky="nsew")
    
    master_1_area.grid(row=0, column=1, sticky="nsew")
    master_2_area.grid(row=0, column=2, sticky="nsew")
    
    #master_1_area
    master_1_area_label.grid(row=0, column=0, sticky="nsew")
    list_master_files_not_t1.grid(row=1, column=0, sticky="nsew")
    select_masterfile_button.grid(row=2, column=0, sticky="nsew")
    
    #master_2_area
    master_2_area_label.grid(row=0, column=0, sticky="nsew")
    list_master_files_t1.grid(row=1, column=0, sticky="nsew")
    select_masterfile_t1_button.grid(row=2, column=0, sticky="nsew")
    
    
    #middle areay
    subject_1_area.grid(row=0, column=0, sticky="nsew")
    
    input_1_area.grid(row=0, column=0, sticky="nsew")
    input_2_area.grid(row=0, column=2, sticky="nsew")
    
    
    input_3_area.grid(row=0, column=3, sticky="nsew")
    
    
    Input_MasterFile.grid(row=0, column=0, sticky="nsew")
    Input_Flair_Image.grid(row=1, column=0, sticky="nsew")
    Input_T1_Image.grid(row=2, column=0, sticky="nsew")   
    Input_Mat.grid(row=3, column=0, sticky="nsew")   
    Output_dir.grid(row=4, column=0, sticky="nsew")   
    
    select_masterinput_button.grid(row=0, column=1, sticky="nsew")
    select_Flair_image_button.grid(row=1, column=1, sticky="nsew")
    
    

    
    
    select_T1_image_button.grid(row=2, column=1, sticky="nsew")
    select_affine_button.grid(row=3, column=1, sticky="nsew")   
    select_out_button.grid(row=4, column=1, sticky="nsew")
    
    run_bianca_button.grid(row=5, column=1, sticky="nsew")
    
    subject_2_area.grid(row=0, column=1, sticky="nsew")
    
    
    #subject_1_area
    list_subject_files.grid(row=0, column=0, sticky="nsew")

    root.mainloop()
    
    
    
    
    
    
    
    
    
    
    
    # # Create the top area
    # top_area = tk.Frame(frame, bg="gray", height=50)
    # top_area.pack(side="top", fill="x")
    
    # # Create the left part of the top area
    # logo_image = tk.PhotoImage(file=logo_image_path)
    # logo_label = tk.Label(top_area, image=logo_image, bg="gray")
    # logo_label.pack(side="left", padx=(10, 0), pady=5)
    
    # # Create the right part of the top area
    # right_area = tk.Frame(top_area, bg="gray")
    # right_area.pack(side="right", padx=10, pady=5, fill="y")
    
    # # Create a checkbutton widget with style="Toolbutton" and add it to the right side of the top area
    # check_box = ttk.Checkbutton(right_area, text="Check me", style="Toolbutton")
    # check_box.pack(side="right", padx=(0, 20), pady=5)
    

    
    
# Aqua
# Black
# Blue
# Brown
# Cyan
# Dark Blue
# Dark Gray
# Dark Green
# Dark Red
# Gold
# Gray
# Green
# Light Blue
# Light Gray
# Light Green
# Lime
# Magenta
# Navy Blue
    
 
    
    
    # root = tk.Tk()
    # root.title("Bianca")
    # root.geometry("1200x800")
    # root.resizable(False,False)
    
    # style = ttk.Style()
    # style.theme_use("clam")
    
    # # Create the main frame
    # frame = tk.Frame(root, bg="gray")
    # frame.pack(expand=True, fill="both")
    
    # image_logo = Image.open(logo_image_path).resize((90,90))
    # image_logo_tk = ImageTk.PhotoImage(image_logo)
    
    # Area_1 = ttk.Label(frame, text="",background="grey")
    # Area_1.configure(width=10)
    # Area_1.pack(side="top", fill="both",expand=True)
    

    # Area_1_left = ttk.Label(Area_1 ,image=image_logo_tk)
    # Area_1_left.configure(text="Bianca Python")
    # Area_1_left.configure(font=("Courire",30))
    # Area_1_left.configure(background="white")

    # Area_1_left.pack(side="left")
    
    # # print("Area_1_left:")
    # # for item in Area_1_left.keys():
    # #     print(item, ": ", Area_1_left[item])
    
    
    # Area_1_right = ttk.Label(Area_1, text="", background="yellow")
    # Area_1_right.configure(width=50)
    # Area_1_right.pack(side="right", fill="both",expand=True)
    
    
    # # Create a checkbutton widget with style="Toolbutton"
    # # check_box = ttk.Checkbutton(Area_1_right, style="Toolbutton")
    # # check_box.pack(side="right")
    
    # # print("Area_1_left:")
    # # for key in check_box.keys():
    # #     print(key, ": ", check_box[key])
    
    
    # root.mainloop()
    
    
    # Area_2 = tk.Label(frame, text="", bg="purple")
    # Area_2.pack(side="top", fill="both",expand=True)
    
    
    
    # Area_2_left = ttk.Label(Area_2, text="", background="red")
    # Area_2_left.pack(side="left", fill="both",expand=True)
    
    
    # Area_2_right = ttk.Label(Area_2, text="", background="white")
    # Area_2_right.pack(side="right", fill="both",expand=True)
    
    
    
    # Area_3 = tk.Label(frame, text="", bg="black")
    # Area_3.pack(side="top", fill="both",expand=True)
    
    
    
    # Area_3_left = ttk.Label(Area_3, text="", background="black")
    # Area_3_left.pack(side="left", fill="both",expand=True)
    
    
    # Area_3_right = ttk.Label(Area_3, text="", background="green")
    # Area_3_right.pack(side="right", fill="both",expand=True)
    
    # Create the title bar
    #menu_bar = tk.Frame(frame, height=30, bg="grey")
    #menu_bar.pack(side="top", fill="x")
    
    
    #logo_label = ttk.Label(frame, image=image_tk)
    #logo_label.pack(side="top", fill="both",expand=True)
    
    
    # label_1_1 = tk.Label(label_1, text="", bg="grey")
    # label_1_1.pack(side="top", fill="x",expand=True)
    
    
    #label_2 = ttk.Label(frame, text="")
    #label_2.pack(side="bottom", fill="both",expand=True)
    
    # label_2_1 = tk.Label(label_2, text="", bg="red")
    # label_2_1.pack(side="bottom", fill="x",expand=True)
    

    


if __name__ == "__main__":

    main()
