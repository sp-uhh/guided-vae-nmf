import os
import platform
import subprocess
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def open_file(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

# function to return key for any value in a dict
def get_key(my_dict, val): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key
