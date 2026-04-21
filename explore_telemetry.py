
# get info from a file.
# plot it.

# %%'

import numpy as np
import pandas as pd
from scipy import signal
import pyedflib
import matplotlib.pyplot as plt
%matplotlib qt

from pathlib import Path
from datetime import timedelta

# %% path

file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\nss-edf\convert\2508171\2508171 - 1.edf'

# batch-4
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512054__SN_920336131__.edf'
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512055__SN_920536131__.edf'
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512058__SN_921536130__.edf'


# batch-3
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3\conversion\sacrifice\2509262__SN_921336130__.edf'
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3\conversion\sacrifice\2509264__SN_920336131__.edf'
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3\conversion\sacrifice\2509265__SN_920536131__.edf'




# %% explore_file

# prerequisite : shift.py : get this info :
    # the quantity of time-shift ( h:m:s )
    # whether if Windows- or software-time is ahead.
    
# windows_is_ahead : windows time is higher than software time.
def explore_file(file_path, h=0, m=0, s=0, windows_is_ahead=True):
    print(file_path)
    print('='*30)
    
    # Open the EDF file for reading
    f = pyedflib.EdfReader(file_path)
    uncorrected_start_time = f.getStartdatetime()
    print(f'Uncorrected acquisition start-time (Software): {uncorrected_start_time}')
    
    # 1. Create the timedelta object for the mismatch
    time_mismatch = timedelta(hours=h, minutes=m, seconds=s)
    
    # 2. Logic to add or subtract based on which clock is higher
    if windows_is_ahead:
        # If Windows time (tw) > Software time (ts), we ADD the mismatch to ts
        corrected_start_time = uncorrected_start_time + time_mismatch
        higher_clock = "Windows Time"
    else:
        # If Windows time (tw) < Software time (ts), we SUBTRACT the mismatch from ts
        corrected_start_time = uncorrected_start_time - time_mismatch
        higher_clock = "Software Time"
        
    # 3. Print the new entities for your records
    print(f'Time mismatch for records: {h:02d}:{m:02d}:{s:02d}')
    print(f'Reference status: {higher_clock} is higher than the other.')
    print(f'Corrected start-time (Windows Reference): {corrected_start_time}')
    
    print('='*30)
    
    n_channels = f.signals_in_file
    print(f'Number of channels: {n_channels}')
    print('Sampling frequencies:')
    for i in range(n_channels):
        sfreq = f.getSampleFrequency(i)
        print(f'    Channel {i}: {sfreq} Hz')
    
    f.close() # Always good practice to close the file handle
    

# without closing the file ( end of the function ) :
    # OSError: F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512058__SN_921536130__.edf: 
        # file has already been opened

# %%% execute

explore_file(file_path, h=0, m=0, s=0, windows_is_ahead=True)

# outputs were saved in the correponding tables :  
    # F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4  \  table_batch_4  .docx
    # ...

# %% process the signal
# %%% band-pass filter

f = pyedflib.EdfReader(file_path)
raw_trace = f.readSignal(1)


# Sampling frequency ( Hz )
sfreq = 500

# the fuzzy noise dissapeared with higher-badn=100.
    # so this was not a 50 Hz power-grid noise !

# Design the High-Pass Filter (Baseline Removal) ---
    # N: Order of the filter (4 is usually sufficient)
    # Wn: Critical frequency (1.0 Hz to remove baseline wander)
    # btype: Type of filter ('high' or 'hp')
    # fs: Sampling frequency
    # output: 'sos' (Second-Order Sections) is more numerically stable than 'ba'
    # bp ; bandpass
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


# %% plot
# %%% old plot

# this plots from the start of acquisition.

# def plot_ecg_segment(signal, sfreq=500, start_s=0, duration_s=10):
#     start_idx = int(start_s * sfreq)
#     end_idx = int((start_s + duration_s) * sfreq)

#     segment = signal[start_idx:end_idx]
#     time_s = np.arange(len(segment)) / sfreq + start_s

#     plt.figure( figsize=(12, 4) )
#     plt.plot(time_s, segment)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.title(f"ECG segment ({start_s}s – {start_s + duration_s}s)")
#     plt.tight_layout()

# %%%%'

# plot_ecg_segment(
#                     signal= ecg_final ,  # ecg_filtered ,  #ecg_final ,     # raw_trace, 
#                     sfreq=500, 
#                     start_s=360, 
#                     duration_s=120
# )

# %%% offset plot

# this plots from the start of gassing.

def plot_ecg_segment_2(signal, 
                     sfreq=500, 
                     # the time-difference between the start of acquisition & gassing ( in seconds ).
                     # get this from your tables :  F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4
                     acq_gass_offset_s = 54 ,
                     start_s=0,  # start time from the start of gassing.
                     duration_s=10
                     ):
    
    acq_gass_offset_sample = int(acq_gass_offset_s * sfreq)
    
    start_idx = int(start_s * sfreq) + acq_gass_offset_sample
    end_idx =  start_idx + int(duration_s * sfreq)

    segment = signal[start_idx:end_idx]
    time_s = np.arange(len(segment)) / sfreq + start_s

    # extract the file_name.
    file_path_2 = Path(file_path)
    file_name = file_path_2.stem

    plt.figure( figsize=(12, 4) )
    plt.plot(time_s, segment)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"ECG segment ({start_s}s – {start_s + duration_s}s)  \n from start of gassing \n {file_name} ")
    plt.tight_layout()


# %%%% execute

# do not forgot to first do the signal-processing on the raw signal.
    # a few cells above.

plot_ecg_segment_2(
                    signal= ecg_final ,  # ecg_filtered ,  #ecg_final ,     # raw_trace, 
                    acq_gass_offset_s = 120 ,
                    sfreq=500, 
                    start_s=480, 
                    duration_s=120
)


# %%%% save the figure

# plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512058__SN_921536130\fibrilation_.pdf' )

# plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_3\terminal\2509262__SN_921336130\fibrilation_.pdf' )
# plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_3\terminal\2509264__SN_920336131\ecg_9_.pdf' )
plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\nss-edf\error__conversion\conversion_test\resolved__2511201_.pdf' )


# %% convert : csv => pickle

# beginning of program_2.
df_peaks = pd.read_csv( source_dir / "Master_Peak_Log.csv.gz", parse_dates=['timestamp'])
df_dropouts = pd.read_csv( source_dir / "Dropout_Map.csv", parse_dates=['start', 'end'])

# %%% problem--csv

# problem arising from saving it to csv :
    # Timestamp column gets from index to a new column.
    # it gets an artifical name ( 'Unnamed: 0' ).
    # it is turned to a string : it looks like the timestamp, but is not a Datetime object !
df_pulse_binned_1s = pd.read_csv( output_dir / output_name )
df_pulse_binned_1s.set_index('Unnamed: 0', inplace=True)
df_pulse_binned_1s.index.name = None
df_pulse_binned_1s.index = pd.to_datetime(df_pulse_binned_1s.index)

#==============================================

# Save the gassing time inside the dataframe object.
df_pulse_binned_1s.attrs['rec_start_windows'] = rec_start_windows
df_pulse_binned_1s.attrs['gassing_start_windows'] = gassing_start_windows

#==============================================

output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
output_name = "df_pulse_binned_1s.pkl"
df_pulse_binned_1s.to_pickle( output_dir / output_name )

# %%% attributes


# %%'

