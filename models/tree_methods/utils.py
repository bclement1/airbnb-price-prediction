# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:29:18 2022

@author: trisr
"""

def writing_description(description, filename):
    with open(filename, "w") as f:
        for key, value in description.items():
            f.write("%s : %s \n" % (key, value))