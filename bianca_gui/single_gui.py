import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from bianca_gui.config.config import Config


# Function to browse for a folder
def browse_folder_file():
    cfg = Config()
    
    # Set the initial directory
    initial_directory = cfg.master_file_path
    folder_path = filedialog.askopenfile(initialdir=initial_directory)
    
    #folder_path = filedialog.askdirectory()
    print("Selected folder:", folder_path)


# Function to create a new window
def check_train_set():

    #new_window = tk.Toplevel(root)
    #debug
    
    #print(cfg.master_file_path)
    
    new_window = tk.Tk()
    new_window.title("Train Data")
    new_window.geometry("600x600")
    
    # Create a label in the new window
    label = tk.Label(new_window, text="Input Image:")
    label.pack()
    
    Entry  = tk.Entry(new_window, width=25)
    Entry.pack()
    
    INPUT_IMAGE = tk.Button(text="Input Image", command=lambda: browse_folder_file())
    INPUT_IMAGE.pack()

    # Function to close the new window
    def close_new_window():
        new_window.destroy()
         
    # Create an exit button for the new window
    exit_button = tk.Button(new_window, text="Exit", command=close_new_window)
    exit_button.pack()

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
check_train_set()












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
    window.mainloop()
    
        
        
"""
    

