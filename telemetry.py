

# %%

import pyedflib
import numpy as np
import matplotlib.pyplot as plt

# %%

# Replace with your file path
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\nss-edf\convert\2508171\2508171 - 1.edf'

# Open the EDF file for reading
f = pyedflib.EdfReader(file_path)

type(f)
    # Out[4]: pyedflib.edfreader.EdfReader

# %%

# --- Get file and channel information ---
n_channels = f.signals_in_file
channel_labels = f.getSignalLabels()
sfreq = f.getSampleFrequency(0) # Get sfreq of the first channel

# %%

n_channels
    # Out[13]: 10

channel_labels
    # Out[14]: 
    # ['STE20a1.SN_92053',
    #  'STE20a1.SN_92053',
    #  'STE20a1.SN_92053',
    #  'STE20a1.SN_92053',
    #  'STE20a1.SN_92053',
    #  'STE20a1.SN_92133',
    #  'STE20a1.SN_92133',
    #  'STE20a1.SN_92133',
    #  'STE20a1.SN_92133',
    #  'STE20a1.SN_92133']

sfreq
    # Out[15]: 1.0

# %%

for i in range(10) :
    sfreq = f.getSampleFrequency(i)
    print(sfreq)

    # 1.0
    # 500.0
    # 500.0
    # 250.0
    # 1.0
    # 1.0
    # 500.0
    # 500.0
    # 250.0
    # 1.0

# %%

# 1 is the index of this channel, according to :
    # the screenshot of the nss-to-edf converter
    # the sampling rates of each channel
    
ecg_1 = f.readSignal(1)

type(ecg_1)
    # Out[18]: numpy.ndarray

ecg_1.shape
    # Out[19]: (42542000,)

# %%

