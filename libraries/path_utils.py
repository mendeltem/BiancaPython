#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:42:27 2022

@author: temuuleu
"""

import os
from os import system, name
from pathlib import Path
import configparser
import sys

import collections
import csv
import errno
import getpass
import itertools
import json
import locale
import os
import platform
import threading
import time
import shlex
import socket
import sys
import readline
import tempfile
import re
import fileinput
import tempfile

# tempfile.tempdir = "/data01/tmp"

from optparse import OptionParser, OptionGroup, SUPPRESS_HELP
from re import compile, escape, sub
from subprocess import Popen, call, PIPE, STDOUT


try:
    from subprocess import DEVNULL  # py3
except ImportError:
    DEVNULL = open(os.devnull, 'wb')



class Spinner(object):
    spinner = itertools.cycle(('-', '\\', '|', '/', ))
    busy = False
    delay = 0.2

    def __init__(self, delay=None, quiet=False):
        if delay:
            try:
                self.delay = float(delay)
            except ValueError:
                pass
        self.quiet = quiet

    def spin_it(self):
        while self.busy:
            sys.stdout.write(next(self.spinner))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def start(self):
        if not self.quiet:
            self.busy = True
            threading.Thread(target=self.spin_it).start()

    def stop(self):
        self.busy = False
        time.sleep(self.delay)


def memoize(f):
    cache = f.cache = {}

    def g(*args, **kwargs):
        key = (f, tuple(args), frozenset(list(kwargs.items())))
        if key not in cache:
            cache[key] = f(*args, **kwargs)
        return cache[key]
    return g


def check_sudo(sudo_pwd):
    command_line = ['sudo', '-S', 'true']
    MsgUser.debug("Checking sudo password")
    cmd = Popen(
        command_line,
        stdin=PIPE,
        stdout=DEVNULL,
        stderr=DEVNULL,
        universal_newlines=True
    )
    cmd.stdin.write(sudo_pwd + '\n')
    cmd.stdin.flush()
    cmd.communicate()

    if cmd.returncode != 0:
        return False
    else:
        return True
    
    
class SudoPasswordError(Exception):
    pass


class RunCommandError(Exception):
    pass


@memoize
def get_sudo_pwd():
    '''Get the sudo password from the user'''
    MsgUser.message("We require your password to continue...")
    attempts = 0
    valid = False

    while attempts < 3 and not valid:
        sudo_pwd = getpass.getpass('password: ')
        valid = check_sudo(sudo_pwd)
        if not valid:
            MsgUser.failed("Incorrect password")
        attempts += 1
    if not valid:
        raise SudoPasswordError()
    return sudo_pwd


def run_cmd(command, as_root=False):
    '''Run the command and return result.'''
    command_line = shlex.split(command)

    if as_root and os.getuid() != 0:
        try:
            sudo_pwd = get_sudo_pwd()
        except SudoPasswordError:
            raise RunCommandError(
                "Unable to get valid administrator's password")
        command_line.insert(0, '-S')
        command_line.insert(0, 'sudo')
    else:
        sudo_pwd = ''
    MsgUser.debug("Will call %s" % (command_line))
    try:
        my_spinner = Spinner(quiet=MsgUser.isquiet())
        my_spinner.start()
        cmd = Popen(command_line, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                    universal_newlines=True)
        if sudo_pwd:
            cmd.stdin.write(sudo_pwd + '\n')
            cmd.stdin.flush()
        (output, error) = cmd.communicate()
    except Exception:
        raise
    finally:
        my_spinner.stop()
    if cmd.returncode:
        MsgUser.debug("An error occured (%s, %s)" % (cmd.returncode, error))
        raise RunCommandError(error)
    MsgUser.debug("Command completed successfully (%s)" % (output))
    return output

def get_feature_paths(start_dir, extensions = ['nii','gz']):
    """Returns all image paths with the given extensions in the directory.
    Arguments:
        start_dir: directory the search starts from.
        extensions: extensions of image file to be recognized.
    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths   


def create_result_dir(result_path,name="bids"):
    """
    

    Parameters
    ----------
    result_path : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is "bids".

    Returns
    -------
    try_result_path : TYPE
        DESCRIPTION.

    """
    
    create_dir(result_path)
    
    trypaths_indizes = [int(p.split("_")[-1]) for p in os.listdir(result_path) if name in p]
    
    if trypaths_indizes:
        trypaths_indizes.sort()                  
        try_result_index = trypaths_indizes[-1] +1
        
    else:
        try_result_index = 1
         
    try_result_path = result_path +name+"_"+str(try_result_index)+"/"

    if not os.path.exists(try_result_path):
        os.makedirs(try_result_path)
        
        
    return try_result_path



def create_dir(output_path):
    """creates a directory of the given path"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def copy_dir(file_path, to_copy_file_path):
    

    cmd = "cp -r "+file_path+" "+to_copy_file_path 
        
    #print(cmd)
        
    system(cmd)
    
    
class shell_colours(object):
    default = '\033[0m'
    rfg_kbg = '\033[91m'
    gfg_kbg = '\033[92m'
    yfg_kbg = '\033[93m'
    mfg_kbg = '\033[95m'
    yfg_bbg = '\033[104;93m'
    bfg_kbg = '\033[34m'
    bold = '\033[1m'  
    
    
class MsgUser(object):
    __debug = False
    __quiet = False

    @classmethod
    def debugOn(cls):
        cls.__debug = True

    @classmethod
    def debugOff(cls):
        cls.__debug = False

    @classmethod
    def quietOn(cls):
        cls.__quiet = True

    @classmethod
    def quietOff(cls):
        cls.__quiet = False

    @classmethod
    def isquiet(cls):
        return cls.__quiet

    @classmethod
    def isdebug(cls):
        return cls.__debug

    @classmethod
    def debug(cls, message, newline=True):
        if cls.__debug:
            mess = str(message)
            if newline:
                mess += "\n"
            sys.stderr.write(mess)

    @classmethod
    def message(cls, msg):
        if cls.__quiet:
            return
        print(msg)

    @classmethod
    def question(cls, msg):
        print(msg, end=' ')

    @classmethod
    def skipped(cls, msg):
        if cls.__quiet:
            return
        print("".join(
            (shell_colours.mfg_kbg, "[Skipped] ", shell_colours.default, msg)))

    @classmethod
    def ok(cls, msg):
        if cls.__quiet:
            return
        print("".join(
            (shell_colours.gfg_kbg, "[OK] ", shell_colours.default, msg)))

    @classmethod
    def failed(cls, msg):
        print("".join(
            (shell_colours.rfg_kbg, "[FAILED] ", shell_colours.default, msg)))

    @classmethod
    def warning(cls, msg):
        if cls.__quiet:
            return
        print("".join(
            (shell_colours.bfg_kbg,
             shell_colours.bold,
             "[Warning]",
             shell_colours.default, " ", msg)))



class IsDirectoryError(Exception):
    pass


class CopyFileError(Exception):
    pass


def copy_file(fname, destination, as_root):
    '''Copy a file using sudo if necessary'''
    MsgUser.debug("Copying %s to %s (as root? %s)" % (
        fname, destination, as_root))
    if os.path.isdir(fname):
        raise IsDirectoryError('Source (%s) is a directory!' % (fname))

    if os.path.isdir(destination):
        # Ensure that copying into a folder we have a terminating slash
        destination = destination.rstrip('/') + "/"
    copy_opts    = '-p'
    fname        = '"%s"' % fname
    destination  = '"%s"' % destination
    command_line = " ".join(('cp', copy_opts, fname, destination))
    try:
        result = run_cmd(command_line, as_root)
    except RunCommandError as e:
        raise CopyFileError(str(e))
    return result


# def copy_file(file_path, to_copy_file_path):
    

#     cmd = "cp "+file_path+" "+to_copy_file_path

#     print(cmd)
        
#     system(cmd)
    
def get_img_paths(start_dir, extensions = ['png','jpeg','jpg','pneg','peng']):
    """Returns all image paths with the given extensions in the directory.
    Arguments:
        start_dir: directory the search starts from.
        extensions: extensions of image file to be recognized.
    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths


def get_dat_paths(start_dir, extensions = ['nii', 'gz','mat']):
    """Returns all image paths with the given extensions in the directory.
    Arguments:
        start_dir: directory the search starts from.
        extensions: extensions of image file to be recognized.
    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths



def get_main_paths(config_path = 'param.ini'):
    
    
    config = configparser.ConfigParser()
    config.read('param.ini')   
     
    return config
    


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



def plot_tree(bids_path):

    paths = DisplayablePath.make_tree(Path(bids_path))
    for path in paths:
        print(path.displayable())

def clear():
  
    # for windows
    if name == 'nt':
        _ = system('cls')
  
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')
        


class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))