import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from bianca_gui.config.config import Config
from bianca_shell.bianca_shell import run_bianca

import os
import pandas as pd

# Global variable to store the selected file path
selected_file_path = None

    
"""
first_column = [
    [tk.Label(text="Train directories")],
    [tk.Listbox(window, selectmode=tk.MULTIPLE, height=10)],
    [tk.Button(text="Select train set", command=lambda: show_selected_subject())],
    [tk.Label(text="Train Subjects")],
    [tk.Listbox(window, selectmode=tk.MULTIPLE, height=30)],
    [tk.Button(text="Show selected subject", command=lambda: show_selected_subject())]
]
"""

# Function to browse for a folder
def browse_folder_file(list_master_files):
    global train_folder
    cfg = Config()
              
    # Set the initial directory
    initial_directory = cfg.master_file_path
    
    #dir(cfg)
    
    print(f"masterfile_dir_path : {initial_directory}")
    #print(f"{os.path.isdir(initial_directory)}")
    
    train_folder = filedialog.askdirectory()
    
    #folder_path = filedialog.askdirectory()
    print("Selected folder:", train_folder)

    train_dir_list = [os.path.join(train_folder, f) for f in os.listdir(train_folder)  if ".txt" in f.lower()]  
    train_dir_names = [os.path.basename( f) for f in train_dir_list ] 

    print("train_dir_names :", train_dir_names)
    
    train_dir_dict = {}

    for index, (p,n) in enumerate(zip(train_dir_list,train_dir_names)):
        train_dir_dict[n] ={ }
        train_dir_dict[n]["index"]= index
        train_dir_dict[n]["path"] = p
        train_dir_dict[n]["name"] =n

    #print("train_dir_dict :", train_dir_dict)
    
    # Clear existing items in the Listbox
    list_master_files.delete(0, tk.END)
    
    # Add file names to the Listbox
    for name in train_dir_names:
        list_master_files.insert(tk.END, name)
        
   
        
        
# # Function to browse for a folder
# def browse_image_file(list_master_files):
#     global train_folder
#     cfg = Config()
    
#     # Set the initial directory
#     initial_directory = cfg.master_file_path

    
#     train_folder = filedialog.askdirectory(initialdir=initial_directory)
    
#     #folder_path = filedialog.askdirectory()
#     print("Selected folder:", train_folder)

#     train_dir_list = [os.path.join(train_folder, f) for f in os.listdir(train_folder)  if ".txt" in f.lower()]  
#     train_dir_names = [os.path.basename( f) for f in train_dir_list ] 

#     print("train_dir_names :", train_dir_names)
    
#     train_dir_dict = {}

#     for index, (p,n) in enumerate(zip(train_dir_list,train_dir_names)):
#         train_dir_dict[n] ={ }
#         train_dir_dict[n]["index"]= index
#         train_dir_dict[n]["path"] = p
#         train_dir_dict[n]["name"] =n

#     #print("train_dir_dict :", train_dir_dict)
    
#     # Clear existing items in the Listbox
#     list_master_files.delete(0, tk.END)
    
#     # Add file names to the Listbox
#     for name in train_dir_names:
#         list_master_files.insert(tk.END, name)
        
        

# Function to create a new window
def check_train_set():
    
    #global list_subject_files
    new_window = tk.Tk()
    new_window.title("Train Data")
    new_window.geometry("600x600")
    
    # Create a label in the new window
    label = tk.Label(new_window, text="Input Image:")
     #Entry  = tk.Entry(new_window, width=25) #Entry.grid(row=0, column=0, padx=1, pady=1) 
     
    list_master_files =  tk.Listbox(new_window, selectmode=tk.MULTIPLE, height=5)
    list_subject_files = tk.Listbox(new_window, selectmode=tk.MULTIPLE, height=15)
    
    
    #standard directory chosen before asted

    selected_master_folder_button = tk.Button(text="Select Master File Folder", command=lambda: browse_folder_file(list_master_files)) 

    # Function to handle file selection
    def select_files(list_subject_files):
        # Get the selected indices
        selected_indices = list_master_files.curselection()
    
        # Get the file names corresponding to the selected indices
        for index in selected_indices:
            file_name = list_master_files.get(index)
            selected_file = os.path.join(train_folder,file_name)
            #selected_file =file_name

        if selected_file:
            # Print selected file names (for debug)
            with open(selected_file) as f:
                lines = f.readlines()
                
            data = [line.strip().split() for line in lines]
            train_master_df = pd.DataFrame(data, columns=['flair', 'mni_matrix', 'mask'])
            train_master_df['subj'] = train_master_df['flair'].apply(lambda x: x.split('/')[-2])
 
            train_dict_data = {}
            for i, row in train_master_df.iterrows():
                subj = row['subj']
                train_dict_data[subj] = {'flair': row['flair'], 'mask': row['mask']}
                if i >=10: break
              
            subject_list = list(train_dict_data.keys())
            subject_list.sort()
            
            # Add file names to the Listbox
            for subject in subject_list:
                list_subject_files.insert(tk.END, subject)
            
        
    # Function to close the new window
    def close_new_window():
        new_window.destroy()
         
    
    # Create a button to trigger file selection
    select_masterfile_button = tk.Button(new_window, text="Selected Master File", command= lambda: select_files(list_subject_files))
    # Create a button to trigger file selection
    
        
    select_subject_button = tk.Button(new_window, text="Selected Subject File", command= lambda: select_files(list_subject_files))
     
    # Create an exit button for the new window
    exit_button = tk.Button(new_window, text="Exit", command=close_new_window)
    
    # Create Listbox 2
    label.grid(row=0, column=0, padx=0, pady=0) 
    selected_master_folder_button.grid(row=0, column=0, padx=0, pady=10) 
    select_masterfile_button.grid(row=0, column=1, padx=0, pady=0) 
    select_subject_button.grid(row=2, column=1, padx=0, pady=0) 
    
    list_master_files.grid(row=1, column=0, padx=0, pady=0) 
    list_subject_files.grid(row=1, column=1, padx=0, pady=0)  
    exit_button.grid(row=2, column=3, padx=1, pady=1) 
    
    #debug
    new_window.mainloop()
    
    
    
    
            
    # cfg = Config()
    # initial_directory = cfg.masterfile_dir_path   
    # initialdir=initial_directory
    
    
    
    # cfg.masterfile_dir_path      
    # cfg.masterfile_file_path      
    # cfg.standard_space_flair     
    # cfg.standard_space_ventrikel  
    # cfg.standard_mask             
    # cfg.data_test_dir_path     

    
    
# Function to handle file selection
def select_mat(Input_Mat,bianca_input_param_dict):
    # Get the selected indices
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
    
    bianca_input_param_dict['input_image'] = image_file_path
    
    
    
    
# Function to handle file selection
def select_Masterfile(Input_MasterFile,bianca_input_param_dict):
    # Get the selected indices
    image_file = filedialog.askopenfile()
    
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
    print(f"oud_dir {image_file_path}")
    
    # Clear existing items in the Listbox
    Output_dir.delete(0, tk.END)
    Output_dir.insert(tk.END, os.path.basename(image_file_path))
    
    bianca_input_param_dict['out_put'] = image_file_path



def run_bianca_shell(bianca_input_param_dict):
    
    input_image  = bianca_input_param_dict['input_image']
    input_matrix  = bianca_input_param_dict['input_matrix']
    input_masterfile  = bianca_input_param_dict['input_masterfile']
    out_put  = bianca_input_param_dict['out_put']
    
    
    print(f"input image {bianca_input_param_dict['input_image']}")
    print(f"input matrix {bianca_input_param_dict['input_matrix']}")
    print(f"input masterfile {bianca_input_param_dict['input_masterfile']}")
    
    print("run bianca")
    run_bianca_cmd = f"run_bianca -image={input_image} -mni={input_matrix} -masterfile={input_masterfile} -output={out_put}"
    os.system(f"{run_bianca_cmd}")
    
    print("end")
    
    
    
def run_bianca_help(bianca_input_param_dict):
    
    
    print(f"input image {bianca_input_param_dict['input_image']}")
    print(f"input matrix {bianca_input_param_dict['input_matrix']}")
    print(f"input masterfile {bianca_input_param_dict['input_masterfile']}")
    
    print("run bianca")
    run_bianca_cmd = f"run_bianca -h"
    os.system(f"{run_bianca_cmd}")
    
    print("end") 

    
    
# Function to create a new window
def run_bianca_gui():
    
    #global list_subject_files
    
    new_window = tk.Tk()
    new_window.title("Run Bianca")
    new_window.geometry("600x600")
    
    cfg = Config()

    global bianca_input_param_dict
    bianca_input_param_dict = {
        "input_image" : "",
        "input_matrix" : "",
        "input_masterfile": "",
        "out_put":"cfg.data_out_test_dir_path "}
    
    
    # Create a label in the new window
    Input_MasterFile    = tk.Entry(new_window,text="Input Masterfile:")
    Input_Image    = tk.Entry(new_window,text="Input Image:", 
                              width=25) #Entry.grid(row=0, column=0, padx=1, pady=1) 
    
    Input_Mat    = tk.Entry(new_window,text="Input Matrix:", 
                              width=25) #Entry.grid(row=0, column=0, padx=1, pady=1) 
    
    Output_dir    = tk.Entry(new_window,text="Output Dir:", 
                              width=25) #Entry.grid(row=0, column=0, padx=1, pady=1) 

    # Create a button to trigger file selection
    select_masterfile_button = tk.Button(new_window, text="Selected Master File", command= lambda: select_Masterfile(Input_MasterFile,bianca_input_param_dict))
    select_image_button = tk.Button(new_window, text="Select Input Image", command= lambda: select_image(Input_Image,bianca_input_param_dict))
    select_affine_button = tk.Button(new_window, text="Select Input Affine Matrix", command= lambda: select_mat(Input_Mat,bianca_input_param_dict))
    
    select_out_button = tk.Button(new_window, text="Select Output Directory", command= lambda: select_Out_dir(Output_dir,bianca_input_param_dict))
    

    
    run_bianca_button = tk.Button(new_window, text="Run Bianca", command= lambda: run_bianca_shell(bianca_input_param_dict))
    help_button = tk.Button(new_window, text="Help Bianca", command= lambda: run_bianca_shell(bianca_input_param_dict))

    #pack Masterfile
    Input_MasterFile.grid(row=1, column=1, padx=0, pady=0)  
    select_masterfile_button.grid(row=1, column=2, padx=0, pady=0) 

    

    # Pack the text box widget to the window
    Input_Image.grid(row=2, column=1, padx=0, pady=0)  
    select_image_button.grid(row=2, column=2, padx=0, pady=0) 
    
    
    #select matrix
    Input_Mat.grid(row=3, column=1, padx=0, pady=0)  
    select_affine_button.grid(row=3, column=2, padx=0, pady=0) 
    
    
    Output_dir.grid(row=4, column=1, padx=0, pady=0)  
    select_out_button.grid(row=4, column=2, padx=0, pady=0) 

    run_bianca_button.grid(row=5, column=2, padx=0, pady=0) 
    help_button.grid(row=6, column=2, padx=0, pady=0) 
    
        
    # Function to close the new window
    def close_new_window():
         new_window.destroy()
    # Create an exit button for the new window
    exit_button = tk.Button(new_window, text="Exit", command=close_new_window)
    
    #list_subject_files.grid(row=1, column=1, padx=0, pady=0)  
    exit_button.grid(row=3, column=3, padx=1, pady=1) 
    
    #debug
    new_window.mainloop()
    

# Function to exit the main window
def exit_main_window():
    result = messagebox.askquestion("Exit", "Are you sure you want to exit?")
    if result == 'yes':
        root.destroy()
        
        
# Function to create the main window
def create_bianca_gui():
    global root
    root = tk.Tk()
    root.title("Main Window")
    root.geometry("1000x800")  # Set the window size to 1000x800 pixels

    # Create a menu
    menu = tk.Menu(root)
    root.config(menu=menu)

    # Create a File menu
    file_menu = tk.Menu(menu)
    menu.add_cascade(label="File", menu=file_menu)
    #file_menu.add_command(label="Options", command=create_new_window)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=exit_main_window)

    # Create a button to open new window
    button = tk.Button(root, text="Inspect Train Data", command=check_train_set)
    button.pack()

    # Create an exit button for the main window
    exit_button = tk.Button(root, text="Exit", command=exit_main_window)
    exit_button.pack()

    # Start the main event loop
    root.mainloop()
    
    
    
# Call the create_bianca_gui() function to start the GUI
run_bianca_gui()












# # Function to create a new window with 600x600 pixels
# def create_new_window():
#     new_window = tk.Toplevel(root)
#     new_window.title("Train Data")
#     new_window.geometry("600x600")
    
#     #new_window.Label(text="Input Image:")

#     # Function to close the new window
#     def close_new_window():
#         new_window.destroy()

#     # Create an exit button for the new window
#     exit_button = tk.Button(new_window, text="Exit", command=close_new_window)
#     exit_button.pack()
    
    


# # Function to exit the main window
# def exit_main_window():
#     result = messagebox.askquestion("Exit", "Are you sure you want to exit?")
#     if result == 'yes':
#         root.destroy()

# # Create the main window


# def create_bianca_gui():

#     root = tk.Tk()
#     root.title("Main Window")
#     root.geometry("1000x800")  # Set the window size to 600x600 pixels
    
#     # Create a menu
#     menu = tk.Menu(root)
#     root.config(menu=menu)
    
#     # Create a File menu
#     file_menu = tk.Menu(menu)
#     menu.add_cascade(label="File", menu=file_menu)
#     file_menu.add_command(label="Inspect Train Data", command=create_new_window)
#     file_menu.add_command(label="New Window 2", command=create_new_window)
#     file_menu.add_command(label="New Window 3", command=create_new_window)
#     file_menu.add_command(label="New Window 4", command=create_new_window)
#     file_menu.add_separator()
#     file_menu.add_command(label="Exit", command=exit_main_window)
    
#     # Create 4 buttons to open new windows
#     for i in range(4):
#         button = tk.Button(root, text=f"Open Window {i+1}", command=create_new_window)
#         button.pack()
    
#     # Create an exit button for the main window
#     exit_button = tk.Button(root, text="Exit", command=exit_main_window)
#     exit_button.pack()
    
#     # Start the main event loop
#     root.mainloop()
    
   
"""


window = tk.Tk()
window.title("Bianca GUI")
window.geometry("1000x800")  # Set the window size to 600x600 pixels

# Function to close the window
def close_window():
    window.destroy()

# Create an exit button
exit_button = tk.Button(window, text="Exit", command=close_window)
exit_button.pack()  # Add the button to the window

# Start the main event loop
window.mainloop()

# Create main window


# Define menu layout
menu_layout = [
    ['File', ['Open', 'Save', '---', 'Exit']],
    ['Tools', ['Word Count']],
    ['Behaviour', ['Directories', ['Choose Output Directory', 'Choose Training Directory']]]
]

# Define first column layout
first_column = [
    [tk.Label(text="Train directories")],
    [tk.Listbox(window, selectmode=tk.MULTIPLE, height=10)],
    [tk.Button(text="Select train set", command=lambda: show_selected_subject())],
    [tk.Label(text="Train Subjects")],
    [tk.Listbox(window, selectmode=tk.MULTIPLE, height=30)],
    [tk.Button(text="Show selected subject", command=lambda: show_selected_subject())]
]

# Define second column layout
second_column = [
    [tk.Label(text="Train directories")],
    [tk.Listbox(window, selectmode=tk.MULTIPLE, height=10)],
    [tk.Label(text="Result directories")],
    [tk.Listbox(window, selectmode=tk.MULTIPLE, height=30)],
    [tk.Button(text="Delete selected result", command=lambda: delete_selected_result())],
    [tk.Button(text="Open Folder", command=lambda: open_folder())]
]

# Define thresholds and minimum cluster sizes
thresholds = [t for t in range(90, 100)]
minimum_clustersizes = [t for t in range(50, 500, 50)]

# Define third column layout
third_column = [
    [tk.Label(text="Input Image:")],
    [tk.Entry(width=25)],
    [tk.Button(text="Input Image", command=lambda: browse_folder("-INPUT_IMAGE-"))],
    [tk.Label(text="Threshold:")],
    [ttk.Spinbox(values=thresholds, width=23)],
    [tk.Label(text="Minimum Cluster Size:")],
    [ttk.Spinbox(values=minimum_clustersizes, width=23)],
    [tk.Label(text="Image Folder:")],
    [tk.Entry(width=25)],
    [tk.Button(text="Browse", command=lambda: browse_folder("-IMAGE_FOLDER-"))],
    [tk.Entry(width=25)],
    [tk.Button(text="Browse", command=lambda: browse_folder("-FOLDER-"))],
    [tk.Button(text="RUN BIANCA", command=lambda: run_bianca())],
    [ttk.Frame()],
    [tk.Label(text="Images from selected result directories")],
    [tk.Listbox(window, selectmode=tk.MULTIPLE, height=30)],
    [tk.Label(text="Test")],
    [tk.Button(text="Exit")]
]

# Define fourth column layout
fourth_column = [
    [tk.Label(text="Image", width=80, height=40, bg="white")]
]

# Create layout using grid
for i, column in enumerate([first_column, second_column, third_column, fourth_column]):
    for j, row in enumerate(column):
        for k, widget in enumerate(row):
            widget.grid(row=j, column=k, sticky="w", padx=5, pady=5)

# Add menu bar
menu_bar = tk.Menu(window)

for item in menu_layout:
    
if item == '---':
    item.add_separator()
else:
    item.add_command(label=item)
    item.add_cascade(label=item[0], menu=item)
    
    window.config(menu=menu_bar)
    window.mainloop()gg
    
        
        
"""
    

