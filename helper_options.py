#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:58:17 2023

@author: temuuleu
"""

tasks = []

def start_single_gui():
    task = input("Enter task to add: ")
    tasks.append(task)
    print("Task added successfully!")

def start_multi_gui():
    task = input("Enter task to add: ")
    tasks.append(task)
    print("Task added successfully!")
    
def configuration():
    task = input("Enter task to add: ")
    tasks.append(task)
    print("Task added successfully!")


def create_option_table():

    print("Welcome to Bianca Python Pipeline")
    
    while True:
        print("Please select the following options")
        print("1. Start Single Patient GUI")
        print("2. Start Multiple Patien GUI")
        print("3. Configuration")
        print("4. Quit")
    
        choice = input("Enter your choice (1/2/3/4): ")
        if choice == "1":
            start_single_gui()
        elif choice == "2":
            start_multi_gui()
        elif choice == "3":
            configuration()
        elif choice == "4":
            print("Thank you for using Todo List App!")
            break
        else:
            print("Invalid choice. Please try again.")