

# %%'

import pyedflib
import mne

import numpy as np
import matplotlib.pyplot as plt

# %% read

# Replace with your file path
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\nss-edf\convert\2508171\2508171 - 1.edf'

# Open the EDF file for reading
f = pyedflib.EdfReader(file_path)

type(f)
    # Out[4]: pyedflib.edfreader.EdfReader

# %% start-time

f.getStartdatetime()
    # Out[5]: datetime.datetime(2025, 8, 17, 9, 30, 41)

# %%'

# if you want to format it :

start_time = f.getStartdatetime()
formatted_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")

formatted_start_time
    # Out[7]: '2025-08-17 09:30:41'

# %%'

# --- Get file and channel information ---
n_channels = f.signals_in_file

n_channels
    # Out[13]: 10

# %%'

channel_labels = f.getSignalLabels()

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

# %%'
in
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

# %%'

# 1 is the index of this channel, according to :
    # the screenshot of the nss-to-edf converter
    # the sampling rates of each channel
    
ecg_1 = f.readSignal(1)
ecg_2 = f.readSignal(2)


type(ecg_1)
    # Out[18]: numpy.ndarray


# %%'

ecg_1.shape
    # Out[19]: (42542000,)

# how long does this file correspond to ?

42542000 / 500
    # Out[3]: 85084.0

85084 / 3600
    # Out[4]: 23.634444444444444

# 23 hours !

# %%'

# blood pressure
bp = f.readSignal(3)

# the length is half of ECG channel, as the  sampling rate is half of it !

bp.shape
    # Out[4]: (21271000,)

21271000 * 2
    # Out[6]: 42542000

# %%'

temperature = f.readSignal(4)

temperature.shape
    # 85084

85084 * 250
    # Out[9]: 21271000

# %% slice

ecg_1_slice = ecg_1[:5000]

ecg_2_slice = ecg_2[:5000]

# the part with less signal drop-outs.
ecg_2_slice_2 = ecg_2[3000:5000]

bp_slice = bp[ 1500:2500 ]

# %%'

# frequency in Hz
sfreq_Hz = 500

# time in seconds.
time_s = np.arange( len( ecg_2_slice_2 )) / sfreq_Hz

# %%'

sfreq_Hz_bp = 250

# time in seconds.
time_s_bp = np.arange( len( bp_slice )) / sfreq_Hz_bp


# %% plot

plt.plot( time_s , ecg_2_slice_2  )

plt.plot( time_s_bp , bp_slice  )

# %%'

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\2508171_1_ecg_2_0.pdf' )
plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\bp.pdf' )

# %% log

# reading the log file !

onsets, durations, descriptions = f.readAnnotations()

type(onsets)
    # Out[11]: numpy.ndarray
type(durations)
    # Out[13]: numpy.ndarray
type(descriptions)
    # Out[14]: numpy.ndarray

onsets.shape
    # Out[12]: (3425,)

durations.shape
    # Out[15]: (3425,)

descriptions.shape
    # Out[16]: (3425,)

# %%'

onsets[:10]
    # Out[17]: 
    # array([  1.8899,   2.2029,   3.1399,  25.4369,  62.7339,  73.6249,
    #        165.1719, 317.2339, 548.0149, 746.9839])

durations[:10]
    # Out[18]: array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])

descriptions[:10]
    # Out[19]: 
    # array(['AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST_AND_RESYNC',
    #        'AP 3710100 / AN SN_921336130 -> DATA_RECEIVED = DATA_SAMPLES_LOST_AND_RESYNC',
    #        'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST',
    #        'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST',
    #        'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST',
    #        'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST',
    #        'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST',
    #        'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST',
    #        'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST',
    #        'AP 3710100 / AN SN_921336130 -> DATA_RECEIVED = DATA_SAMPLES_LOST'],
    #       dtype='<U76')

# %% log _ pyedflib

# this threw an error.

# Check if any annotations were returned
if len(onsets) > 0:
    print(f"✅ Found {len(onsets)} annotations in the file.\n")
    print("--- First 10 Annotations ---")
    
    # We use zip to iterate through the three arrays simultaneously
    for i, (onset, duration, description) in enumerate(zip(onsets, durations, descriptions)):
        if i >= 10:
            break
        # The description is returned as bytes, so we decode it to a string
        print(f"Time: {onset:.2f}s,  Duration: {duration:.2f}s,  Description: '{description.decode('utf-8')}'")
        
else:
    print("❌ No annotations were found in this file.")

# %%'

    # ---------------------------------------------------------------------------
    # AttributeError                            Traceback (most recent call last)
    # Cell In[20], line 5
    #       3     break
    #       4 # The description is returned as bytes, so we decode it to a string
    # ----> 5 print(f"Time: {onset:.2f}s,  Duration: {duration:.2f}s,  Description: '{description.decode('utf-8')}'")
    
    # AttributeError: 'numpy.str_' object has no attribute 'decode'

# %%'
# %% mne

raw_mne = mne.io.read_raw_edf(file_path, preload=True)
    # Extracting EDF parameters from F:\OneDrive - Uniklinik RWTH Aachen\home_cage\nss-edf\convert\2508171\2508171 - 1.edf...
    # EDF file detected
    # Setting channel info structure...
    # Creating raw.info structure...
    # C:\Users\azare\AppData\Local\Temp\ipykernel_11952\197008347.py:1: RuntimeWarning: Channel names are not unique, found duplicates for: {'STE20a1.SN_92133', 'STE20a1.SN_92053'}. Applying running numbers for duplicates.
    #   raw_mne = mne.io.read_raw_edf(file_path, preload=True)
    # Reading 0 ... 42541999  =      0.000 ... 85083.998 secs...

# %%'

# time of start of capturing !
raw_mne.info
    # <Info | 7 non-empty values
    #  bads: []
    #  ch_names: STE20a1.SN_92053-0, STE20a1.SN_92053-1, STE20a1.SN_92053-2, ...
    #  chs: 10 EEG
    #  custom_ref_applied: False
    #  highpass: 0.0 Hz
    #  lowpass: 250.0 Hz
    #  meas_date: 2025-08-17 09:30:41 UTC
    #  nchan: 10
    #  projs: []
    #  sfreq: 500.0 Hz
    # >

raw_mne.ch_names
    # Out[21]: 
    # ['STE20a1.SN_92053-0',
    #  'STE20a1.SN_92053-1',
    #  'STE20a1.SN_92053-2',
    #  'STE20a1.SN_92053-3',
    #  'STE20a1.SN_92053-4',
    #  'STE20a1.SN_92133-0',
    #  'STE20a1.SN_92133-1',
    #  'STE20a1.SN_92133-2',
    #  'STE20a1.SN_92133-3',
    #  'STE20a1.SN_92133-4']

# ! this outputs 1 frequency !
raw_mne.info['sfreq']
    # Out[22]: 500.0

# %% log _ mne

annotations_mne = raw_mne.annotations

# %%'

# Check if there are any annotations in the file
if len(annotations_mne) > 0:
    print(f"✅ Found {len(annotations_mne)} annotations in the file.\n")
    print("--- First 10 Annotations ---")
    
    # Loop through and print the first 10 annotations to see what they contain
    # Each annotation has an onset (start time in seconds), duration, and a description (the text)
    for i, ann in enumerate(annotations_mne):
        if i >= 10:
            break
        onset = ann['onset']
        duration = ann['duration']
        description = ann['description']
        print(f"Time: {onset:.2f}s,  Duration: {duration:.2f}s,  Description: '{description}'")

else:
    print("❌ No annotations were found in this file.")

# %%'

    # ✅ Found 3425 annotations in the file.
    
    # --- First 10 Annotations ---
    # Time: 1.89s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST_AND_RESYNC'
    # Time: 2.20s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_921336130 -> DATA_RECEIVED = DATA_SAMPLES_LOST_AND_RESYNC'
    # Time: 3.14s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST'
    # Time: 25.44s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST'
    # Time: 62.73s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST'
    # Time: 73.62s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST'
    # Time: 165.17s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST'
    # Time: 317.23s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST'
    # Time: 548.01s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_920536131 -> DATA_RECEIVED = DATA_SAMPLES_LOST'
    # Time: 746.98s,  Duration: 0.00s,  Description: 'AP 3710100 / AN SN_921336130 -> DATA_RECEIVED = DATA_SAMPLES_LOST'

# %%'



