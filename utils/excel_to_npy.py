#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : YPC
# @time    : 2025/4/29 下午10:20
# @function: the script is used to do something.
# @version : V1
import os
import pandas as pd
import numpy as np

df = pd.read_excel("assets/results.xlsx", sheet_name="Sheet1")
data = df['Max100'].values
np.save('assets/object.npy', data)
