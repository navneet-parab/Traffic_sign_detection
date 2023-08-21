#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:51:26 2022

@author: Navneet
"""

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="Ls7Syw4bNDA6JI3vTYrs")
project = rf.workspace("navneet-parab").project("traffic-sign-detection-and-classification-pj8a2")
dataset = project.version(1).download("darknet")


from roboflow import Roboflow

rf = Roboflow(api_key="Ls7Syw4bNDA6JI3vTYrs")

#%%

workspace = rf.workspace()

# name
workspace.name

# URL
workspace.url

# Projects
workspace.projects()

#%%
project = workspace.project("traffic-sign-detection-and-classification-pj8a2")

#%%
project.upload("/Users/Navneet/Downloads/roboflow-customer-demo/IMG_0193 Large.jpeg")

#%%
version = project.version(1)
#%%
model = version.model
prediction = model.predict("/Users/Navneet/Downloads/roboflow-customer-demo/IMG_0193 Large.jpeg")
#%%