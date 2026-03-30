

# %%'

import pyedflib
# import mne

import numpy as np
import matplotlib.pyplot as plt

# %% pyedflib

# read

# %%% 24-h recrodings


# Replace with your file path
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\nss-edf\convert\2508171\2508171 - 1.edf'


file_path_new = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\STE20a1__SN_920336130_-ECG_1__2511201 - 1.edf'
# this is the same date as above , another animal.
file_path_2= r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\STE20a1__SN_920336131_-ECG_1__2511201 - 1.edf'


file_path_3= r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\STE20a1__SN_920336131__ECG_1__2511211 - 1.edf'

file_path_4= r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\STE20a1__SN_920336131__ECG_1__2511221 - 1.edf'

# %%%  terminal experiment.

# recordings during the terminal experiment ( sacrifice ).

file_path_bach_4_sacrifice_cage_2 = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512054__SN_920336131__.edf'

# %%%'

# Open the EDF file for reading
f = pyedflib.EdfReader(file_path)
f_new = pyedflib.EdfReader(file_path_new)

f_2 = pyedflib.EdfReader(file_path_2)

f_3 = pyedflib.EdfReader(file_path_3)
f_4 = pyedflib.EdfReader(file_path_4)

f_5 = pyedflib.EdfReader( file_path_bach_4_sacrifice_cage_2 )

type(f)
    # Out[4]: pyedflib.edfreader.EdfReader

# %%% start-date-time

f.getStartdatetime()
    # Out[5]: datetime.datetime(2025, 8, 17, 9, 30, 41)

# 1st day of the recording, after surgery, in the late evening.
f_new.getStartdatetime()
    # Out[46]: datetime.datetime(2025, 11, 20, 21, 5, 12)
f_2.getStartdatetime()
    # Out[57]: datetime.datetime(2025, 11, 20, 21, 5, 12)


f_3.getStartdatetime()
    # Out[58]: datetime.datetime(2025, 11, 21, 10, 21, 17)

f_4.getStartdatetime()
    # Out[59]: datetime.datetime(2025, 11, 22, 12, 45, 31)

f_5 = pyedflib.EdfReader( file_path_bach_4_sacrifice_cage_2 )


# %%%%'

# if you want to format it :

start_time = f.getStartdatetime()
formatted_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")

formatted_start_time
    # Out[7]: '2025-08-17 09:30:41'

# %%%'

# --- Get file and channel information ---
n_channels = f.signals_in_file

n_channels
    # Out[13]: 10

f_5.getStartdatetime()
    # Out[32]: datetime.datetime(2025, 12, 5, 10, 37, 41)
f_5.signals_in_file
    # Out[9]: 3

# %%%'

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

f_5.getSignalLabels()
    # Out[10]: ['STE20a1.SN_92033', 'STE20a1.SN_92033', 'STE20a1.SN_92033']

# %%% sampling rate

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

f_3.getSampleFrequency(0)
    # Out[63]: 500.0

# %%

for i in range(3) :
    sfreq = f_5.getSampleFrequency(i)
    print(sfreq)

    # 1.0
    # 500.0
    # 1.0

# %%%'

# 1 is the index of this channel, according to :
    # the screenshot of the nss-to-edf converter
    # the sampling rates of each channel


ecg_1 = f.readSignal(1)     # noise
ecg_2 = f.readSignal(2)

ecg_3 = f_3.readSignal(0)


type(ecg_1)
    # Out[18]: numpy.ndarray


# %%%'

ecg_1.shape
    # Out[19]: (42542000,)

# how long does this file correspond to ?

42542000 / 500
    # Out[3]: 85084.0

85084 / 3600
    # Out[4]: 23.634444444444444

# 23 hours !

# %%%'

# blood pressure
bp = f.readSignal(3)

# the length is half of ECG channel, as the  sampling rate is half of it !

bp.shape
    # Out[4]: (21271000,)

21271000 * 2
    # Out[6]: 42542000

# %%%'

temperature = f.readSignal(4)

temperature.shape
    # 85084

85084 * 250
    # Out[9]: 21271000

# %%%  sample trace

# plotting a slice of the trace.

# %%%% slice

ecg_1_slice = ecg_1[:5000]

ecg_2_slice = ecg_2[:5000]

# end of the trace.
ecg_2_slice_end = ecg_2[-5000:]


# the part with less signal drop-outs.
ecg_2_slice_2 = ecg_2[3000:5000]

bp_slice = bp[ 1500:2500 ]

# %%%%'

# sampling rate
# frequency in Hz
sfreq_Hz = 500

# x-axis
# time in seconds.
time_s = np.arange( len( ecg_2_slice_2 )) / sfreq_Hz

time_s_ecg_2_slice = np.arange( len( ecg_2_slice )) / sfreq_Hz


time_s_ecg_2_slice_end = np.arange( len( ecg_2_slice_end )) / sfreq_Hz


# %%%%'

sfreq_Hz_bp = 250

# x-axis
# time in seconds.
time_s_bp = np.arange( len( bp_slice )) / sfreq_Hz_bp


# %%%% plot

plt.plot( time_s_ecg_2_slice , ecg_2_slice  )

plt.plot( time_s , ecg_2_slice_2  )

plt.plot( time_s_ecg_2_slice_end , ecg_2_slice_end  )

plt.plot( time_s_bp , bp_slice  )

plt.title( 'end of the trace' )

# %%%%'

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\2508171_1_ecg_2_end.pdf' )
plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\magnified_voltage.pdf' )

# %%%% noise channel

# ecg-1 channel : noise.

# x-axis
# time in seconds.
time_s_ecg_1_slice = np.arange( len( ecg_1_slice  )) / sfreq_Hz

plt.plot( time_s_ecg_1_slice , ecg_1_slice  )

# this was saved after magnifications.
# possibly 50-Hz power-grid noise.
plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\ecg_1.pdf' )


# %%% log

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

# %%%%'

# these are 's' ( seconds from the onset of recording ) , according to the mne outputs ( below ).
    # this is from pyedflib.
onsets[:10]
    # Out[17]: 
    # array([  1.8899,   2.2029,   3.1399,  25.4369,  62.7339,  73.6249,
    #        165.1719, 317.2339, 548.0149, 746.9839])

durations[:10]
    # Out[18]: array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])

# these are annotations.
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

# %%%% log _ pyedflib

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

# %%%%'

    # ---------------------------------------------------------------------------
    # AttributeError                            Traceback (most recent call last)
    # Cell In[20], line 5
    #       3     break
    #       4 # The description is returned as bytes, so we decode it to a string
    # ---> 5 print(f"Time: {onset:.2f}s,  Duration: {duration:.2f}s,  Description: '{description.decode('utf-8')}'")
    
    # AttributeError: 'numpy.str_' object has no attribute 'decode'

# %%% neurokit

import neurokit2 as nk

signals , info = nk.ecg_process( ecg_2_slice , sampling_rate=500 )

type(signals)
    # Out[17]: pandas.DataFrame

signals.shape
    # Out[18]: (5000, 19)

# as the initial phase of the trace is only drop-outs, the end of the trace is selected.
signals[-10:]
    # Out[22]: 
    #        ECG_Raw  ECG_Clean   ECG_Rate  ECG_Quality  ECG_R_Peaks  ECG_P_Peaks  \
    # 4990  0.324022   0.165731 146.341463     0.000000            0            0   
    # 4991  0.060023   0.155274 146.341463     0.000000            0            0   
    # 4992 -0.089976   0.140902 146.341463     0.000000            0            0   
    # 4993 -0.241975   0.124233 146.341463     0.000000            0            0   
    # 4994 -0.115975   0.109566 146.341463     0.000000            0            0   
    # 4995 -0.201975   0.093818 146.341463     0.000000            0            0   
    # 4996 -0.083976   0.079687 146.341463     0.000000            0            0   
    # 4997 -0.207975   0.063549 146.341463     0.000000            0            0   
    # 4998 -0.077976   0.046361 146.341463     0.000000            0            0   
    # 4999 -0.185975   0.024537 146.341463     0.000000            0            0   
    
    #       ECG_P_Onsets  ECG_P_Offsets  ECG_Q_Peaks  ECG_R_Onsets  ECG_R_Offsets  \
    # 4990             0              0            0             0              0   
    # 4991             0              0            0             0              0   
    # 4992             0              0            0             0              0   
    # 4993             0              0            0             0              0   
    # 4994             0              0            0             0              0   
    # 4995             0              0            0             0              0   
    # 4996             0              0            0             0              0   
    # 4997             0              0            0             0              0   
    # 4998             0              0            0             0              0   
    # 4999             0              0            0             0              0   
    
    #       ECG_S_Peaks  ECG_T_Peaks  ECG_T_Onsets  ECG_T_Offsets  ECG_Phase_Atrial  \
    # 4990            0            0             0              0               NaN   
    # 4991            0            0             0              0               NaN   
    # 4992            0            0             0              0               NaN   
    # 4993            0            0             0              0               NaN   
    # 4994            0            0             0              0               NaN   
    # 4995            0            0             0              0               NaN   
    # 4996            0            0             0              0               NaN   
    # 4997            0            0             0              0               NaN   
    # 4998            0            0             0              0               NaN   
    # 4999            0            0             0              0               NaN   
    
    #       ECG_Phase_Completion_Atrial  ECG_Phase_Ventricular  \
    # 4990                     0.000000                    NaN   
    # 4991                     0.000000                    NaN   
    # 4992                     0.000000                    NaN   
    # 4993                     0.000000                    NaN   
    # 4994                     0.000000                    NaN   
    # 4995                     0.000000                    NaN   
    # 4996                     0.000000                    NaN   
    # 4997                     0.000000                    NaN   
    # 4998                     0.000000                    NaN   
    # 4999                     0.000000                    NaN   
    
    #       ECG_Phase_Completion_Ventricular  
    # 4990                          0.000000  
    # 4991                          0.000000  
    # 4992                          0.000000  
    # 4993                          0.000000  
    # 4994                          0.000000  
    # 4995                          0.000000  
    # 4996                          0.000000  
    # 4997                          0.000000  
    # 4998                          0.000000  
    # 4999                          0.000000  

type(info)
    # Out[19]: dict

info
    # Out[20]: 
    # {'method_peaks': 'neurokit',
    #  'method_fixpeaks': 'None',
    #  'ECG_R_Peaks': array([ 868, 1143, 1381, 1539, 2610, 2806, 3011]),
    #  'ECG_R_Peaks_Uncorrected': array([ 868, 1143, 1381, 1539, 2610, 2806, 3011]),
    #  'ECG_fixpeaks_ectopic': [],
    #  'ECG_fixpeaks_missed': [],
    #  'ECG_fixpeaks_extra': [],
    #  'ECG_fixpeaks_longshort': [],
    #  'ECG_fixpeaks_method': 'kubios',
    #  'ECG_fixpeaks_rr': array([0.71433333, 0.55      , 0.476     , 0.316     , 2.142     ,
    #         0.392     , 0.41      ]),
    #  'ECG_fixpeaks_drrs': array([-0.021802  , -0.0706356 , -0.03180751, -0.068773  ,  0.78487185,
    #         -0.75220468,  0.00773696]),
    #  'ECG_fixpeaks_mrrs': array([ 0.40613233,  0.1492738 ,  0.        , -0.64550834,  3.36067778,
    #         -0.33889188, -0.13313609]),
    #  'ECG_fixpeaks_s12': array([-0.0706356 , -0.03180751, -0.0706356 , -0.03180751, -0.068773  ,
    #          0.00773696, -0.75220468]),
    #  'ECG_fixpeaks_s22': array([-0.03180751, -0.03180751,  0.78487185,  0.78487185, -0.75220468,
    #          0.00773696, -0.75220468]),
    #  'ECG_fixpeaks_c1': 0.13,
    #  'ECG_fixpeaks_c2': 0.17,
    #  'sampling_rate': 500,
    #  'ECG_P_Peaks': [nan, 1105, nan, nan, 2573, nan, 2931],
    #  'ECG_P_Onsets': [nan, 1094, nan, nan, 2562, nan, 2923],
    #  'ECG_P_Offsets': [nan, 1110, nan, nan, 2577, nan, 2942],
    #  'ECG_Q_Peaks': [np.int64(850),
    #   np.int64(1138),
    #   nan,
    #   np.int64(1521),
    #   nan,
    #   nan,
    #   nan],
    #  'ECG_R_Onsets': [nan, 1127, nan, nan, nan, nan, nan],
    #  'ECG_R_Offsets': [884, nan, nan, 1559, nan, nan, 3036],
    #  'ECG_S_Peaks': [np.int64(902),
    #   nan,
    #   np.int64(1521),
    #   np.int64(1553),
    #   np.int64(2630),
    #   np.int64(2862),
    #   nan],
    #  'ECG_T_Peaks': [909, nan, nan, 1627, nan, nan, 3059],
    #  'ECG_T_Onsets': [902, nan, nan, 1619, nan, nan, 3052],
    #  'ECG_T_Offsets': [918, nan, nan, 1638, nan, nan, 3070]}

# %% mne

raw_mne = mne.io.read_raw_edf(file_path, preload=True)
    # Extracting EDF parameters from F:\OneDrive - Uniklinik RWTH Aachen\home_cage\nss-edf\convert\2508171\2508171 - 1.edf...
    # EDF file detected
    # Setting channel info structure...
    # Creating raw.info structure...
    # C:\Users\azare\AppData\Local\Temp\ipykernel_11952\197008347.py:1: RuntimeWarning: Channel names are not unique, found duplicates for: {'STE20a1.SN_92133', 'STE20a1.SN_92053'}. Applying running numbers for duplicates.
    #   raw_mne = mne.io.read_raw_edf(file_path, preload=True)
    # Reading 0 ... 42541999  =      0.000 ... 85083.998 secs...

# %%% info _ mne

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

# %%% log _ mne

annotations_mne = raw_mne.annotations

# %%%'

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

# %%%'

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

# %% drop-out


# Let's find the most common value in the signal (the 'mode')
# If there are many dropouts, the dropout value will be very frequent.

np.unique( ecg_2 , return_counts=True)
    # Out[22]: 
    # (array([-4.16795051, -4.16595052, -4.16395053, ...,  2.76200677,
    #          2.89000598,  3.0000053 ], shape=(2890,)),
    #  array([4604311,      18,      19, ...,       1,       1,       1],
    #        shape=(2890,)))

vals , counts = np.unique( ecg_2 , return_counts=True)

vals.shape
    # Out[25]: (2890,)

counts.shape

dropout_value = vals[np.argmax(counts)]

# The most frequent value (likely the dropout floor)
dropout_value
    # Out[23]: np.float64(-4.167950506165304)

# %%% partition-sort

first_10 = np.partition( ecg_2 , 10)[:10]
first_10 = np.sort(first_10)

first_10
    # Out[29]: 
    # array([-4.16795051, -4.16795051, -4.16795051, -4.16795051, -4.16795051,
    #        -4.16795051, -4.16795051, -4.16795051, -4.16795051, -4.16795051])


# %%% hist

# note : ! : if you set te number of bins to '1000' : the plot will be blank !
plt.hist(ecg_2, bins=100)

plt.xlabel(' y-axis values in ECG trace ( voltage )' , loc='right' )
plt.ylabel('number of samples' , loc='top' )

plt.title( 'Distribution of voltage values of ECG trace.' 
          '\n Finding-out signal drop-outs.'
          '\n STE20a1__SN_920336131__ECG_1__2511211 '
          )

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\hist_2.pdf' )

# %%% quantitation

# quantification of signal drop-out times.
# sample : each 1 sample acquired in time ( x-axis ).
drop_samples = ecg_2 < -3

# tns : total number of samples.
# sum of boolean values.
drop_samples_tns = np.sum( drop_samples )

drop_samples_tns
    # Out[53]: np.int64(4614220)

# pt : percent time
drop_samples_pt  =  ( drop_samples_tns / ecg_2.size ) * 100


drop_samples_pt
    # Out[55]: np.float64(10.84626956889662)
#  =>  signal drop-out occured in about 11 % of the data ( total recording time ).

# %%%%'

# checking the percentage signal drop-out in a newer recording.

drop_samples = ecg_3 < -3
drop_samples_tns = np.sum( drop_samples )
drop_samples_pt  =  ( drop_samples_tns / ecg_3.size ) * 100
drop_samples_pt
    # Out[71]: np.float64(2.1118464551178118)

# %%% duration of each drop-out segment.

# duration of each drop-out segment.

changes = np.diff( drop_samples.astype(int) )

changes[:10]
    # Out[62]: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# sample indices at the start of the drop-outs.
    # ( whenever the signal drops ).
# [0] : extracts the array from the tuple.
start = np.where(changes == 1)[0] + 1

start.shape
    # Out[64]: (18273,)

start[:10]
    # Out[65]: 
    # array([ 1153,  1404,  2620,  2829,  2955, 12387, 12512, 12638, 15572,
    #        31249])

# sample indices at the end of the drop-outs.
    # ( whenever the signal returns back to normal ).
end = np.where(changes == -1)[0] + 1

end.shape
    # Out[67]: (18273,)

# note : the 1st value ( 859 ).
    # this signal already starts with a drop-out !
    # that's why the index of 1st end-point (859) is earlier that the index of the first start-point ( above : 1153 ).
    # you may take a look at the plot ( below ) to have a better understanding.
    # a way to fix this : is to add 1 sample at the beginning of the trace equaling perhaps '0'.
        # this fixes the occasion where signal-drop-out is present at the beginnin gof a trace !
end[:10]
    # Out[68]: 
    # array([  859,  1362,  1530,  2787,  2871,  2997, 12469, 12595, 12721,
    #        15697])

# Then duration:
durations = ( end - start ) / fs

# %%% explore drop-outs

drop_out_slice = ecg_2[ : 2000]

plt.plot( drop_out_slice )

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\drop_out_slice.pdf' )


# %% analysis


import pyedflib
import numpy as np
import pandas as pd
from scipy import signal
from datetime import timedelta

# %%%'

# --- 1. Load Data ---
f = pyedflib.EdfReader("your_data.edf")
start_time = f.getStartdatetime()
sfreq = f.getSampleFrequency(0)
ecg_raw = f.readSignal(0)
f.close()

# --- 2. Baseline Correction (High-pass filter) ---
# Removes low-frequency oscillations below 1.0 Hz
sos = signal.butter(4, 1.0, 'hp', fs=sfreq, output='sos')
ecg_filtered = signal.sosfiltfilt(sos, ecg_raw)

# --- 3. Binning Logic ---
bin_size_sec = 30
samples_per_bin = int(bin_size_sec * sfreq)
total_bins = len(ecg_raw) // samples_per_bin

results = []

for i in range(total_bins):
    start_idx = i * samples_per_bin
    end_idx = start_idx + samples_per_bin
    
    # Get segments from both raw (for dropouts) and filtered (for peaks)
    raw_segment = ecg_raw[start_idx:end_idx]
    filtered_segment = ecg_filtered[start_idx:end_idx]
    
    # Calculate Timestamp for this bin
    timestamp = start_time + timedelta(seconds=i * bin_size_sec)
    
    # Check for any dropout (value <= -4) in this segment
    if np.any(raw_segment <= -4.0):
        avg_hr = np.nan
    else:
        # Peak Detection Parameters:
        # height: adjusts to your signal amplitude
        # distance: ensures peaks aren't too close (e.g., 0.1s distance for max 600 BPM).
        peaks, _ = signal.find_peaks(filtered_segment, 
                                     height=0.15, 
                                     distance=int(sfreq * 0.1)
                                     )
        
        # Calculate BPM for this 30s segment
        avg_hr = (len(peaks) / bin_size_sec) * 60

    results.append([timestamp, avg_hr])

# --- 4. Create Final DataFrame ---
df = pd.DataFrame(results, columns=['Timestamp_Start', 'Average_Heart_Rate'])

print(df.head(20))


# %% signal processing

from scipy import signal

file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\nss-edf\convert\2508171\2508171 - 1.edf'

# Open the EDF file for reading
f = pyedflib.EdfReader(file_path)

raw_trace = f.readSignal(2)

# Sampling frequency ( Hz )
sfreq = 500

# %%%'

raw_trace = ecg_2

# %%% Notch Filter

# unnecessary !

# # Design the Notch Filter ---
#     # w0: Center frequency to remove (50Hz for Germany)
#     # Q: Quality factor (higher = narrower notch)
#     # fs: Sampling frequency of your Stellar data
# b_notch , a_notch = signal.iirnotch(
#                                     w0=50.0, 
#                                     Q=15.0, 
#                                     fs=sfreq
# )

# # Apply the Notch Filter ---
#     # x: The raw input signal
#     # b, a: The coefficients calculated above
# ecg_notched_Q_15 = signal.filtfilt(
#                                 b=b_notch, 
#                                 a=a_notch, 
#                                 x=raw_trace
# )

# %%% band-pass

# the fuzzy noise dissapeared with higher-badn=100.
    # so this was not a 50 Hz power-grid noise !

# Design the High-Pass Filter (Baseline Removal) ---
    # N: Order of the filter (4 is usually sufficient)
    # Wn: Critical frequency (1.0 Hz to remove baseline wander)
    # btype: Type of filter ('high' or 'hp')
    # fs: Sampling frequency
    # output: 'sos' (Second-Order Sections) is more numerically stable than 'ba'
sos_bp = signal.butter(
                        N=4, 
                        Wn=[1,80] ,
                        btype='bandpass',    # 'highpass'
                        fs=sfreq, 
                        output='sos'
)

# Apply the High-Pass Filter ---
    # sos: The filter design from above
    # x: The notched signal from step 2
ecg_final = signal.sosfiltfilt(
                                sos=sos_bp, 
                                x= raw_trace         # ecg_notched
)

# %%%'

ecg_final.shape
    # Out[23]: (42542000,)

# %%% plot


def plot_ecg_segment(signal, sfreq=500, start_s=0, duration_s=10):
    start_idx = int(start_s * sfreq)
    end_idx = int((start_s + duration_s) * sfreq)

    segment = signal[start_idx:end_idx]
    time_s = np.arange(len(segment)) / sfreq + start_s

    plt.figure( figsize=(12, 4) )
    plt.plot(time_s, segment)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"ECG segment ({start_s}s – {start_s + duration_s}s)")
    plt.tight_layout()

# %%%%'

plot_ecg_segment(
                    signal=ecg_final ,     # raw_trace, 
                    sfreq=500, 
                    start_s=10007, 
                    duration_s=1
)

# %%%%'

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\high-band-80_3_.pdf' )

# %%%%'

# this is from before writing the function.


# sfreq = 500

# ecg_final_slice = ecg_final[:5000]

# time_s_ecg_final_slice = np.arange( len( ecg_final_slice )) / sfreq

# plt.plot( time_s_ecg_final_slice , ecg_final_slice )

# plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\ecg_2_notch_bp_filtered__.pdf' )


# %% sanity check

# random trace : this randomly checks trace slices throughout the whole recording period, to detect various types of abnormal signals.

import os
import random
import matplotlib.pyplot as plt

# %%%'

# Define your path (using raw string for Windows paths)
save_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\random_trace'

if not os.path.exists(save_path):
    os.makedirs(save_path)

file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\nss-edf\convert\2508171\2508171 - 1.edf'

# Open the EDF file for reading
f = pyedflib.EdfReader(file_path)


# or = f.readSignal(2)    ( the raw signal )
source_trace = ecg_final  # ecg_final

sfreq = 500

# %%%'

duration_sec = 4
samples_to_plot = int(duration_sec * sfreq)
total_samples = len(source_trace)

# Generate 20 random start indices, ensuring we don't go past the end of the file
# varialbe '_' : it's equivalent to putting 'i' :
    # since this variable ( or a potential 'i' ) is not used, & is only a counter, the convention is to put '_'.
    # note : it's not a special keyword, but an ordinary variable as a convention.
random_starts = [random.randint(0, total_samples - samples_to_plot) for _ in range(20)]

for i, start_idx in enumerate(random_starts):
    end_idx = start_idx + samples_to_plot
    slice_data = source_trace[start_idx:end_idx]
    
    # Calculate time in seconds for the x-axis relative to start of file
    time_axis = np.arange(start_idx, end_idx) / sfreq
    
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, slice_data, linewidth=0.8)
    plt.title(f"Random Slice {i+1} | Start: {time_axis[0]:.2f}s")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    # plt.ylim(-4.5, 2.0) # Keeps the scale consistent to see dropouts
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_path, f"random_slice_{i+1:02d}.pdf"))
    plt.close() # Close to save memory

print(f"Successfully saved 20 random plots to: {save_path}")

# %% audit : peak detection

# quality control of the algorithm.
    # this finds peaks on random segments of a trace.
    # plots the detected peaks over the trace.

import os
import random
from pathlib import Path

from pypdf import PdfWriter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats

# %%% variables


# this prints the file-name on top of each plot.
source_file = 'STE20a1__SN_920336131__ECG_1__2511221'
# '2508171 - 1'   # the old file.

ecg_raw = f.readSignal(2)
ecg_raw = f_3.readSignal(0)
ecg_raw = f_4.readSignal(0)

# sampling frequency ( Hz )
sfreq = 500

save_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\audit _ peak detection'

# %%% process

#---- Configuration & Paths
if not os.path.exists(save_path):
    os.makedirs(save_path)

#=========================================================

#---- Pre-processing 
# applied to the whole trace.

# Interpolate dropouts to prevent filter ringing [cite: 59, 71]
# For this audit, we'll use a simple version; in the final, we'll use the robust one.
ecg_working = ecg_raw.copy()
dropout_mask = ecg_working <= -3.0
ecg_working[dropout_mask] = 0 # Temporary flat-fill for filtering : to reduce ringing artifacts.

# Band-pass Filter (1Hz to 100Hz) to remove drift and 'fuzz'.
sos = signal.butter(N=4, Wn=[1.0, 100.0], btype='bandpass', fs=sfreq, output='sos')
ecg_filtered = signal.sosfiltfilt(sos, ecg_working)

#=========================================================

#---- Random Audit Loop
# number of example slices of trace.
num_samples = 20
# each segment has a duration of 4 s.
    # this is only for plotting & visualization.
    # the practical segmentaion should be either /30s or at the start time-stamps of video recordings.
duration_sec = 4  
samples_per_plot = int(duration_sec * sfreq)

random_starts = [random.randint(0, len(ecg_raw) - samples_per_plot) for _ in range(num_samples)]

#=========================================================
# start_idx : in samples ( not s )
for i, start_idx in enumerate(random_starts):
    end_idx = start_idx + samples_per_plot
    
    # Get segment data
    raw_seg = ecg_raw[start_idx:end_idx]
    filt_seg = ecg_filtered[start_idx:end_idx]
    
    # a. Check for Dropouts in the raw trace [cite: 63, 76]
    has_dropout = np.any(raw_seg <= -3.0)
    
    # b. Polarity Correction (Flip if inverted) using Skewness [cite: 134, 139]
    if stats.skew(filt_seg) < 0:
        filt_seg = filt_seg * -1   # flip the signal.
        plot_label = "Flipped & Filtered"
    else:
        plot_label = "Filtered"

    # c. Local Dynamic Peak Detection
    seg_sd = np.std(filt_seg)
    
    # prominence of the peak.
        # this is calculated per-segment to dynamically adjust for signal quality.
        # hence it adjusts to low or high signal/noise , unlike defining a fixed peak height.
    dynamic_prom = 3 * seg_sd
    
    peaks, props = signal.find_peaks(
        filt_seg, 
        prominence=dynamic_prom, 
        distance=int(sfreq * 0.1), # 100ms refractory period
        height=-np.inf, # this parameter is only present to retrieve actual peak heights in the returned properties dictionary.
        width=0  #  # this parameter is only present to retrieve actual peak widths in the returned properties dictionary.
    )

    #=================================================================
    #---- Plotting
    time_axis = np.arange(0, duration_sec, 1/sfreq)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Plot: Raw Data (to see dropouts/artifacts) [cite: 59, 71]
    ax1.plot(time_axis, raw_seg, color='gray', alpha=0.6, label='Raw Signal')
    ax1.set_title(
                    f'source file : {source_file}'
                    f"\n Validation Slice {i+1} | Start Time Sample-Index : {start_idx}"
                  )
    ax1.set_ylabel("Voltage (V)")
    ax1.legend(loc='lower right')
    if has_dropout:
        ax1.text(0.5, 0.5, 'DROPOUT DETECTED', transform=ax1.transAxes, 
                 color='red', fontsize=20, ha='center', fontweight='bold')

    # Bottom Plot: Processed Data + Detected Peaks
    ax2.plot(time_axis, filt_seg, color='tab:blue', label=plot_label)
    ax2.plot(time_axis[peaks], filt_seg[peaks], "x", color='red', label='Detected R-Peaks')
    
    # Label each peak with its Heart Rate (BPM) based on instantaneous interval
    if len(peaks) > 1:
        intervals = np.diff(peaks) / sfreq
        instant_bpm = 60 / intervals
        for p_idx, bpm in zip(peaks[1:], instant_bpm):
            ax2.annotate(f"{int(bpm)}", (time_axis[p_idx], filt_seg[p_idx]), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    ax2.set_ylabel("Relative Voltage")
    ax2.set_xlabel("Time in Segment (s)")
    ax2.legend(loc='lower right', frameon=True, framealpha=0.8, fontsize='small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"validation_{i+1:02d}.pdf"))
    plt.close()

print(f"Audit complete. Please check the 20 plots in: {save_path}")

#=================================================================
#---- merge pdfs

folder = Path( save_path )
output = folder / "merged_output.pdf"

writer = PdfWriter()

# folder.glob("*.pdf") :
    # it merges only the PDFs in the immediate folder, not PDFs inside subfolders.
for pdf in sorted(folder.glob("*.pdf")):
    writer.append(str(pdf))

with open(output, "wb") as f:
    writer.write(f)

print("PDFs merged successfully.")

# %% -  ( combined ( HR ))

# don't use this :
    # this combines signal processing & extraction of peaks + binning & calculating the average heart-rate.
    # if a different bin-size is desired, the whole signal-processing ... should be repeated.

# Creates 30 s bins  =>  calculates average heart rate in that bin.
# Contains dataframes to save 
    # average heart rates / bin
    # peak locations & other properties.

import numpy as np
import pandas as pd
from scipy import signal, stats
from datetime import timedelta
import pyedflib

# %%% variables

# Load Data

file_path = "your_data_file.edf"
f = pyedflib.EdfReader(file_path)
start_time = f.getStartdatetime()
sfreq = f.getSampleFrequency(0)
ecg_raw = f.readSignal(0)
f.close()

# %%%'

#---- Pre-processing (Whole Signal)
# Pad dropouts with 0 to prevent filter ringing
ecg_clean_for_filt = np.where(ecg_raw <= -4.0, 0, ecg_raw)

# Band-pass filter: 1Hz (remove drift) to 100Hz (remove grid/EMG fuzz)
sos = signal.butter(N=4, Wn=[1.0, 100.0], btype='bandpass', fs=sfreq, output='sos')
ecg_filtered = signal.sosfiltfilt(sos, ecg_clean_for_filt)

#=================================================================
#---- Binning & Peak Detection

bin_size_sec = 30
samples_per_bin = int(bin_size_sec * sfreq)
total_bins = len(ecg_raw) // samples_per_bin

hr_averages = []
master_peak_log = []

#=================================================================

for b_idx in range(total_bins):
    start_s = b_idx * samples_per_bin   # s ( start_s ) : segment : start of the segment ( in samples ).
    end_s = start_s + samples_per_bin
    
    raw_seg = ecg_raw[start_s:end_s]
    filt_seg = ecg_filtered[start_s:end_s]
    bin_timestamp = start_time + timedelta(seconds=b_idx * bin_size_sec)
    
    # Check for Dropout or Extreme Artifacts
    if np.any(raw_seg <= -4.0) or np.any(np.abs(raw_seg) > 2.0):
        hr_averages.append([bin_timestamp, np.nan])
        continue
    
    # Universal Polarity Check
    if stats.skew(filt_seg) < 0:
        filt_seg = filt_seg * -1
        
    # Dynamic Peak Detection (Prominence = 4 * SD)
    seg_sd = np.std(filt_seg)
    peaks, props = signal.find_peaks(
        filt_seg, 
        prominence=4 * seg_sd, 
        distance=int(sfreq * 0.1),
        height=-np.inf, # Unlock height for logging
        width=0         # Unlock width for logging
    )
    
    # Calculate Average HR for this bin
    avg_hr = (len(peaks) / bin_size_sec) * 60
    hr_averages.append([bin_timestamp, avg_hr])
    
    # Log Individual Peak Parameters for Research Reserves
    for i, p_idx in enumerate(peaks):
        master_peak_log.append({
            'timestamp': bin_timestamp + timedelta(seconds=p_idx/sfreq),
            'amplitude': props['peak_heights'][i],
            'prominence': props['prominences'][i],
            'width_ms': (props['widths'][i] / sfreq) * 1000,
            'bin_index': b_idx
        })

#=================================================================
#---- Export Results

df_hr = pd.DataFrame(hr_averages, columns=['Timestamp_Start', 'Average_Heart_Rate'])
df_peaks = pd.DataFrame(master_peak_log)

df_hr.to_csv("Heart_Rate_30s_Averages.csv", index=False)
df_peaks.to_csv("Master_Peak_Log.csv.gz", index=False, compression='gzip')

print("Analysis Complete. 30s averages and Master Peak Log have been saved.")

# %% program-1

import numpy as np
import pandas as pd
import pyedflib
from scipy import signal, stats, ndimage
from datetime import timedelta
from pathlib import Path

# %%%' path

file_path = file_path_bach_4_sacrifice_cage_2
#---- save path
output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe')
# output_dir.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist

# %%%'

#---- Load Raw Telemetry Data

# note : the file should not have been oepned before , otherwise :
    # OSError: F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512054__SN_920336131__.edf: 
        # file has already been opened
    # you should close the file if previously have been opened :
        # f_5.close()
f = pyedflib.EdfReader(file_path)
start_time = f.getStartdatetime() 
sfreq = f.getSampleFrequency(1)
ecg_raw = f.readSignal(1)
f.close()

#---- Create the Dropout Map (Dataframe-3)
# Identify zones where voltage is strictly < -3.5V [cite: 188, 207]
is_dropout_global = ecg_raw <= -3.5
diff = np.diff(is_dropout_global.astype(int))
starts = np.where(diff == 1)[0]
ends = np.where(diff == -1)[0]

# Handle edge cases (if recording starts or ends with a dropout)
if is_dropout_global[0]: starts = np.insert(starts, 0, 0)
if is_dropout_global[-1]: ends = np.append(ends, len(ecg_raw)-1)

# Apply 100ms safety buffer to the map for Program-2's future use.
    # this is to avoid finding peaks over a potential ringing artifact.
safety_buffer_sec = 0.1 
buffer_samples = int(safety_buffer_sec * sfreq) # e.g., 50 samples at 500Hz
total_samples = len(ecg_raw)

dropout_zones = []

for s, e in zip(starts, ends):
    # Calculate buffered Sample Stamps (clamped to signal boundaries)
    # Using max/min ensures we don't get negative indices or go past the end
    s_buffered = max(0, s - buffer_samples)
    e_buffered = min(total_samples - 1, e + buffer_samples)
    
    # Calculate buffered Timestamps based on these exact indices
    t_start = start_time + timedelta(seconds=s_buffered / sfreq)
    t_end = start_time + timedelta(seconds=e_buffered / sfreq)
    
    dropout_zones.append({
        'start_sample': int(s_buffered), # Original integer index
        'end_sample': int(e_buffered),   # Original integer index
        'start_time': t_start,
        'end_time': t_end
    })

# Save the map with both precision formats 
df_dropouts = pd.DataFrame(dropout_zones)
df_dropouts.to_csv( output_dir / "Dropout_Map_2.csv", index=False)

#---- Pre-processing & Filtering (Whole Signal)
# Fill dropouts with 0s to prevent 'ringing' artifacts in the filter [cite: 180, 181, 182]
ecg_filled = np.where(is_dropout_global, 0, ecg_raw)

# 1-80 Hz Band-pass filter 
# 'sos' (Second-Order Sections) is used for maximum numerical stability
sos_bp = signal.butter(N=4, Wn=[1.0, 80.0], btype='bandpass', fs=sfreq, output='sos')
ecg_filtered = signal.sosfiltfilt(sos=sos_bp, x=ecg_filled)



#---- Granular Peak Extraction
bin_size_sec = 30 # Internal processing window for dynamic SD/Skewness [cite: 183, 186]
samples_per_bin = int(bin_size_sec * sfreq)
total_bins = len(ecg_raw) // samples_per_bin
buffer_samples = int(safety_buffer_sec * sfreq) # 100ms in samples

master_peak_log = []

for b_idx in range(total_bins):
    start_idx = b_idx * samples_per_bin   # sample_index of the start of the current ( iterated ) bin.
    end_idx = start_idx + samples_per_bin
    
    raw_seg = ecg_raw[start_idx:end_idx]
    filt_seg = ecg_filtered[start_idx:end_idx]
    # time-delta : how much the bin is far from the start time of aquisition.
    bin_timestamp = start_time + timedelta(seconds=b_idx * bin_size_sec)
    
    # A. Create a buffered 'Forbidden Mask' for this bin
    # We dilate the dropout zones by 100ms to hide edge transients
    is_dropout_seg = raw_seg <= -3.5
    forbidden_mask = ndimage.binary_dilation(
        is_dropout_seg, 
        structure=np.ones(buffer_samples * 2 + 1)
    )
    clean_mask = ~forbidden_mask
    
    

    # B. Skip bin if it contains no clean data (less than 1s)
    if np.sum(clean_mask) < (sfreq * 1):
        continue

    # C. Measure Skewness & SD ONLY on buffered clean data 
    # This prevents artifacts from inflating your thresholds
    seg_skew = stats.skew(filt_seg[clean_mask])
    seg_sd = np.std(filt_seg[clean_mask])

    # D. Polarity Check: Flip signal if skewness is negative [cite: 192]
    if seg_skew < 0:
        filt_seg = filt_seg * -1
        
    # E. Find Peaks & Properties using dynamic prominence (4 * SD) [cite: 195, 196]
    # We unlock height and width to save them for future research
    peaks, props = signal.find_peaks(
        filt_seg, 
        prominence=4 * seg_sd, 
        distance=int(sfreq * 0.1), # 100ms refractory period
        height=-np.inf, 
        width=0
    )
    
    # F. 
    for i, p_idx in enumerate(peaks):
        # Filter Peaks: Only save peaks that are in the clean zone
            # this may not be absolutely needed as program-2 also avoids +/- 100ms around the drop-out zones.
        if clean_mask[p_idx]:
            
            # sample_stamp : the sample_index from the start of the file for each peak.
            # Calculate the absolute sample index (The 'Sample Stamp')
            # This is the most original, accurate ground-truth location.
            absolute_sample = start_idx + p_idx
                # = ( b_idx * bin_size_sec * sfreq ) + p_idx
            
            master_peak_log.append({
                'sample_stamp' : absolute_sample ,
                # time-delta : how much this specific peak is distant from the start of that bin.
                'timestamp': bin_timestamp + timedelta(seconds=p_idx/sfreq),
                'amplitude_v': props['peak_heights'][i],
                'prominence_v': props['prominences'][i],
                'width_ms': (props['widths'][i] / sfreq) * 1000
            })

#---- Save the Master Peak Log (Dataframe-1)
df_peaks = pd.DataFrame(master_peak_log)
# Ensure sample_stamp is saved as an integer to prevent scientific notation.
df_peaks['sample_stamp'] = df_peaks['sample_stamp'].astype(int)
df_peaks.to_csv( output_dir / "Master_Peak_Log.csv.gz", index=False, compression='gzip')
# Dropout_Map.csv : saved higer in th program.

print("Program 1 Complete.")
print(f"Master Peak Log: {len(df_peaks)} beats identified.")
print("Dropout Map saved for Program 2 analysis.")

# %%% out

# Program 1 Complete.
# Master Peak Log: 1105 beats identified.
# Dropout Map saved for Program 2 analysis.


df_dropouts
        # Out[37]: 
        #    start_sample  end_sample              start_time                end_time
        # 0             0        1041 2025-12-05 10:37:41.000 2025-12-05 10:37:43.082
        # 1        226449      226999 2025-12-05 10:45:13.898 2025-12-05 10:45:14.998
    # note : if yo uopen it in excel, each cell may only show the minute-second !
        # you should click on the cell  =>  the function-bar on top will show a more complete version.
        # however, the sub-seconds are not visible there.

df_peaks.shape
    # Out[29]: (1105, 5)

df_peaks[:10]
    # Out[30]: 
    #    sample_stamp               timestamp  amplitude_v  prominence_v  width_ms
    # 0          1101 2025-12-05 10:37:43.202     0.484368      0.721350  9.013177
    # 1          1188 2025-12-05 10:37:43.376     0.564310      0.817075  8.776676
    # 2          1274 2025-12-05 10:37:43.548     0.568072      0.829985  8.832137
    # 3          1361 2025-12-05 10:37:43.722     0.484357      0.738015  8.896035
    # 4          1448 2025-12-05 10:37:43.896     0.465744      0.738688  9.087089
    # 5          1533 2025-12-05 10:37:44.066     0.493679      0.773321  9.070266
    # 6          1621 2025-12-05 10:37:44.242     0.593571      0.864803  8.746738
    # 7          1707 2025-12-05 10:37:44.414     0.573133      0.834913  8.818648
    # 8          1795 2025-12-05 10:37:44.590     0.596577      0.883905  9.050516
    # 9          1883 2025-12-05 10:37:44.766     0.475522      0.715336  8.912181

# %% program-2

import pandas as pd
from datetime import timedelta

# %%% load _ path

#---- Load the Data Created by Program 1
source_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe')
df_peaks = pd.read_csv( source_dir / "Master_Peak_Log.csv.gz", parse_dates=['timestamp'])
df_drops = pd.read_csv( source_dir / "Dropout_Map.csv", parse_dates=['start', 'end'])

#---- save path
output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe')
# output_dir.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist

# %%%'

#---- Set Flexible Timing Parameters
# these 2 are the same, with different 'types'.
    # you should keep them the same : change them simultansously.
bin_size_str = "10S" 
bin_duration_sec = 10

# Define your custom start time (e.g., 14:00:00 on the day of recording)
# If you want to use the original start time, just set this to None.
custom_start_time = pd.Timestamp("2026-03-30 14:00:00") 

#---- Filter and Align (time) the Data
if custom_start_time is not None:
    # Discard any peaks that occurred before our custom start time
    df_peaks = df_peaks[df_peaks['timestamp'] >= custom_start_time]
    
    # Anchor the binning grid to the custom start time
    # 'origin' ensures the first bin is [custom_start_time, custom_start_time + 20s)
    bin_groups = df_peaks.set_index('timestamp').resample(bin_size_str, origin=custom_start_time)
else:  # this will just start binning from the start-time of recording.
    bin_groups = df_peaks.set_index('timestamp').resample(bin_size_str)

#---- Perform Binning & Overlap Checking
results = []

for bin_start, peak_data in bin_groups:
    bin_end = bin_start + timedelta(seconds=bin_duration_sec)
    
    # Temporal intersection check: Does this bin hit any dropout zone? 
    overlaps = df_drops[(bin_start < df_drops['end_time']) & (bin_end > df_drops['start_time'])]
    
    if not overlaps.empty:
        avg_hr = None # Data is corrupt in this window [cite: 177, 210]
    else:
        # Calculate Heart Rate: (Peaks / Seconds) * 60
        peak_count = len(peak_data) # [cite: 204]
        avg_hr = (peak_count / bin_duration_sec) * 60

    results.append({
        'Bin_Start': bin_start,
        'Bin_End': bin_end,
        'Average_Heart_Rate_BPM': avg_hr
    })

#---- Export Results
df_final = pd.DataFrame(results)
output_filename = f"HR_Analysis_{bin_size_str}_from_{custom_start_time.strftime('%H%M') if custom_start_time else 'start'}.csv"
df_final.to_csv(output_filename, index=False)

print(f"Analysis complete. \n Bins anchored to {custom_start_time if custom_start_time else 'recording start'}.")


# %% shift

# this tests the shift in time relative to Windows.

from datetime import datetime, timedelta

# %%% add

# notocord : bottom of the window.
    # start_time + elapsed_time

def add_time(t1, t2):
    base = datetime.strptime(t1, "%H:%M:%S")
    h, m, s = map(int, t2.split(":"))
    return (base + timedelta(hours=h, minutes=m, seconds=s)).time()

# F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\screenshot\2025-12-05
print(add_time("10:37:41", "00:05:57"))
print(add_time("11:10:06", "00:08:02"))
print(add_time("11:45:18", "00:08:00"))

    # 10:43:38
    # 11:18:08
    # 11:53:18

# %%% subtract

# current time differences : 
        # notocord ( A )  -  Windows ( Aw )

def time_diff(t1, t2):
    t1_dt = datetime.strptime(t1, "%H:%M:%S")
    t2_dt = datetime.strptime(t2, "%H:%M:%S")
    return t1_dt - t2_dt

# Given times
# F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\screenshot\2025-12-05
A  = "10:43:38"
Aw = "09:46:51"

B  = "11:18:08"
Bw = "10:21:25"

C  = "11:53:18"
Cw = "10:56:33"

print(time_diff(A, Aw))
print(time_diff(B, Bw))
print(time_diff(C, Cw))

    # 0:56:47
    # 0:56:43
    # 0:56:45

# %%'




