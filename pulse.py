

# %% explore

# random trace : this randomly checks trace slices throughout the whole recording period, to detect various types of abnormal signals.

import os
import random
import matplotlib.pyplot as plt

# %%% file

# batch-4  3rd
# file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512058__SN_921536130__.edf'

# batch-4  2nd
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512055__SN_920536131__.edf'


# batch-3  , 3rd
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3\conversion\sacrifice\2509265__SN_920536131__.edf'


# %%% read

f = pyedflib.EdfReader(file_path)
# blood-pressure cannel
raw_trace = f.readSignal(2)

# f.close()

# %%%'

def plot_pulse_segment(signal, 
                     sfreq=250,
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
    plt.title(f"Pulse segment ({start_s}s – {start_s + duration_s}s)  \n from start of gassing \n {file_name} ")
    plt.tight_layout()



# %%%'


plot_pulse_segment(
                    signal= raw_trace ,  # ecg_filtered ,  #ecg_final ,     # raw_trace, 
                    acq_gass_offset_s = 0 ,
                    sfreq=250, 
                    start_s=60, 
                    duration_s=60
)

# %%%'

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\pulse\batch_3\2509265\explore\3.pdf' )

# %% merrge pdf

folder_path =  r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\pulse\2512058_pulse'

from pypdf import PdfWriter, PdfReader
import os

writer = PdfWriter()

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".pdf"):
        reader = PdfReader(os.path.join(folder_path, filename))
        for page in reader.pages:
            writer.add_page(page)

with open(os.path.join(folder_path, "merged_output.pdf"), "wb") as f:
    writer.write(f)

# %% pulse-processing

import numpy as np
import pandas as pd
import pyedflib
from scipy import ndimage
from datetime import timedelta
from pathlib import Path

# %%% variables

#====================================================
# batch_4  , 3rd
# --- 1. Manual User Inputs ---
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512058__SN_921536130__.edf'
output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')


# Windows ( corrected ) Start Time from your exploration function/screenshots
rec_start_windows = pd.Timestamp("2025-12-05 10:48:33") 
# Windows Gassing Start Time
gassing_start_windows = pd.Timestamp("2025-12-05 10:49:27") 

#====================================================
# batch_4  , 2nd 
# --- 1. Manual User Inputs ---
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512055__SN_920536131__.edf'
output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512055__SN_920536131')


# Windows ( corrected ) Start Time from your exploration function/screenshots
rec_start_windows = pd.Timestamp("2025-12-05 10:13:23") 
# Windows Gassing Start Time
gassing_start_windows = pd.Timestamp("2025-12-05 10:14:24") 

#====================================================
# batch_3  , 3rd
# --- 1. Manual User Inputs ---
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3\conversion\sacrifice\2509265__SN_920536131__.edf'
output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_3\terminal\2509265__SN_920536131')


# Windows ( corrected ) Start Time from your exploration function/screenshots
rec_start_windows = pd.Timestamp("2025-09-26 10:58:53") 
# Windows Gassing Start Time
gassing_start_windows = pd.Timestamp("2025-09-26 11:00:30") 

#====================================================
# batch_3  , 1st
# --- 1. Manual User Inputs ---
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3\conversion\sacrifice\2509262__SN_921336130__.edf'
output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_3\terminal\2509262__SN_921336130')


# Windows ( corrected ) Start Time from your exploration function/screenshots
rec_start_windows = pd.Timestamp("2025-09-26 09:13:54") 
# Windows Gassing Start Time
gassing_start_windows = pd.Timestamp("2025-09-26 9:45:00") 


#====================================================

output_name = "df_pulse_binned_1s.pkl"
# Pulse_Mean_Pressure_1s_Bins : this was probably later renamed in windows as : df_pulse_binned_1s .

# this will be the original dataframe
    # you can later resample for larger bins.
bin_size_str = '1s'

# choosing the appropriate cannel  =>  below.

# %%% process

# function of it is created in the below cell.

# --- 2. Load Signal ---
f = pyedflib.EdfReader(file_path)
sfreq = f.getSampleFrequency(2) # Pulse is Channel 2.    # check the appropriate channel accoding to below.
pulse_raw = f.readSignal(2)
f.close()

# --- 3. Remove "Valleys" (Signal Drop-outs) ---
# Create a mask for any value < 0 (the transmitter floor)
is_dropout = pulse_raw < 0
buffer_samples = int(0.1 * sfreq) # 100ms safety buffer
# Dilate the mask to ensure we don't include artifacts on the edges of dropouts
# adds 1s (True) to the 2 ends of each dropout area.
forbidden_mask = ndimage.binary_dilation(is_dropout, structure=np.ones(buffer_samples * 2 + 1))

# Replace forbidden data with NaN so they are ignored in the average
pulse_cleaned = np.where(~forbidden_mask, pulse_raw, np.nan)

# --- 4. Time Alignment & 2s Binning ---
# converting samples to time-stamps.
# Create a Series with Windows timestamps
# We use the corrected start time you provided
times = [rec_start_windows + timedelta(seconds=i/sfreq) for i in range(len(pulse_cleaned))]
df_pulse = pd.DataFrame({'pressure': pulse_cleaned}, index=times)

# Tight binning (1-second) for a continuous-looking profile
# Resample automatically ignores NaN values in its mean calculation
df_binned = df_pulse.resample(bin_size_str).mean()

# --- 5. Final Formatting (Gassing Start = 0) ---
# Calculate the relative time from gassing in decimal minutes.
# to be put at the x-axis of the plot later.
df_binned['Minutes_from_Gassing'] = (df_binned.index - gassing_start_windows).total_seconds() / 60.0

# --- 6. Export ---
df_binned.to_pickle( output_dir / output_name )

print(f"Analysis complete. Mean pressure saved to {output_name}")


# %%% function

def process_pulse_mean_pressure(file_path, 
                                rec_start_windows, 
                                gassing_start_windows, 
                                output_dir, 
                                output_name, 
                                bin_size_str='1S', 
                                channel=2):
    """
    Reads a telemetry EDF file, cleans pulse dropouts, bins the data, 
    and exports a Pickled dataframe with relative gassing times.
    """
    # 1. Path Management
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output folder exists
    
    print(f"Processing: {file_path.name}...")

    # --- 2. Load Signal ---
    f = pyedflib.EdfReader(str(file_path)) # pyedflib usually prefers string paths
    sfreq = f.getSampleFrequency(channel) # Pulse is usually Channel 2
    pulse_raw = f.readSignal(channel)
    f.close()

    # --- 3. Remove "Valleys" (Signal Drop-outs) ---
    is_dropout = pulse_raw < 0
    buffer_samples = int(0.1 * sfreq) # 100ms safety buffer
    
    # Dilate the mask to ensure we don't include artifacts on the edges of dropouts
    forbidden_mask = ndimage.binary_dilation(is_dropout, structure=np.ones(buffer_samples * 2 + 1))

    # Replace forbidden data with NaN so they are ignored in the average
    pulse_cleaned = np.where(~forbidden_mask, pulse_raw, np.nan)

    # --- 4. Time Alignment & Binning ---
    # Convert samples to time-stamps using the Windows start time
    times = [pd.to_datetime(rec_start_windows) + timedelta(seconds=i/sfreq) for i in range(len(pulse_cleaned))]
    df_pulse = pd.DataFrame({'pressure': pulse_cleaned}, index=times)

    # Tight binning for a continuous-looking profile
    df_binned = df_pulse.resample(bin_size_str).mean()

    # --- 5. Final Formatting (Gassing Start = 0) ---
    gassing_start_ts = pd.to_datetime(gassing_start_windows)
    df_binned['Minutes_from_Gassing'] = (df_binned.index - gassing_start_ts).total_seconds() / 60.0

    # --- 6. Attach Metadata ---
    df_binned.attrs['file_source'] = file_path.name
    df_binned.attrs['rec_start_windows'] = rec_start_windows
    df_binned.attrs['gassing_start_windows'] = gassing_start_windows

    # --- 7. Export ---
    out_file = output_dir / output_name
    df_binned.to_pickle(out_file)

    print(f"  -> Analysis complete. Saved to: {out_file.name}\n")
    return df_binned

# %%% execute

# Process a single file
df_binned = process_pulse_mean_pressure(
                                        file_path = file_path,
                                        rec_start_windows = rec_start_windows,
                                        gassing_start_windows = gassing_start_windows,
                                        output_dir = output_dir,
                                        output_name = output_name,
                                        bin_size_str = bin_size_str,
                                        channel = 2 # check the appropriate channel accoding to below.
)

# choosing the appropriate channel :
    # F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4  \  table_batch_4__.docx
        #  <=    C:\code\telemetry\explore_telemetry.py  |  explore_file__.py

# %%% out

df_binned.shape
    # Out[49]: (248, 2)

# 1s bins.
df_binned[:5]
    # Out[58]: 
    #                        pressure  Minutes_from_Gassing
    # 2025-12-05 10:48:33         NaN             -0.900000
    # 2025-12-05 10:48:34  138.133109             -0.883333
    # 2025-12-05 10:48:35  129.885842             -0.866667
    # 2025-12-05 10:48:36  126.532331             -0.850000
    # 2025-12-05 10:48:37  133.489083             -0.833333

# 2s bins
df_binned[:5]
    # Out[52]: 
    #                        pressure  Minutes_from_Gassing
    # 2025-12-05 10:48:32         NaN             -0.916667
    # 2025-12-05 10:48:34  130.847538             -0.883333
    # 2025-12-05 10:48:36  130.010707             -0.850000
    # 2025-12-05 10:48:38  134.598271             -0.816667
    # 2025-12-05 10:48:40  136.343135             -0.783333

# %%% simple plot

df_binned['pressure'].plot()

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\pulse\batch_3\2509265\average_1s_.pdf' )
plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\pulse\batch_3\2509262\average_1s_.pdf' )

# average_2s_.pdf

# %%% calibration

# =>  file_dataframe.py : to load the data.

#================================
#---- batch_3 , 1st

df_pulse_binned_1s.head()
    # Out[15]: 
    #                      pressure  Minutes_from_Gassing
    # 2025-09-26 09:13:54       NaN            -31.100000
    # 2025-09-26 09:13:55 93.979854            -31.083333
    # 2025-09-26 09:13:56 86.792964            -31.066667
    # 2025-09-26 09:13:57 89.955967            -31.050000
    # 2025-09-26 09:13:58 92.824171            -31.033333

# 30 : calibration value :
    # <=  F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3  |  the table
df_pulse_binned_1s['pressure_calibrated'] = df_pulse_binned_1s['pressure'] + 30 
df_pulse_binned_1s.head()
    # Out[17]: 
    #                      pressure  Minutes_from_Gassing  pressure_calibrated
    # 2025-09-26 09:13:54       NaN            -31.100000                  NaN
    # 2025-09-26 09:13:55 93.979854            -31.083333           123.979854
    # 2025-09-26 09:13:56 86.792964            -31.066667           116.792964
    # 2025-09-26 09:13:57 89.955967            -31.050000           119.955967
    # 2025-09-26 09:13:58 92.824171            -31.033333           122.824171

source_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_3\terminal\2509262__SN_921336130')
df_pulse_binned_1s.to_pickle( source_dir / 'df_pulse_binned_1s.pkl' )

#================================
#---- batch_3 , 3rd

df_pulse_binned_1s.head()
    # Out[22]: 
    #                       pressure  Minutes_from_Gassing
    # 2025-09-26 10:58:53        NaN             -1.616667
    # 2025-09-26 10:58:54 172.233086             -1.600000
    # 2025-09-26 10:58:55 164.223210             -1.583333
    # 2025-09-26 10:58:56 165.755849             -1.566667
    # 2025-09-26 10:58:57 163.896786             -1.550000

df_pulse_binned_1s['pressure_calibrated'] = df_pulse_binned_1s['pressure'] - 33
df_pulse_binned_1s.head()
    # Out[24]: 
    #                       pressure  Minutes_from_Gassing  pressure_calibrated
    # 2025-09-26 10:58:53        NaN             -1.616667                  NaN
    # 2025-09-26 10:58:54 172.233086             -1.600000           139.233086
    # 2025-09-26 10:58:55 164.223210             -1.583333           131.223210
    # 2025-09-26 10:58:56 165.755849             -1.566667           132.755849
    # 2025-09-26 10:58:57 163.896786             -1.550000           130.896786

source_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_3\terminal\2509265__SN_920536131')
df_pulse_binned_1s.to_pickle( source_dir / 'df_pulse_binned_1s.pkl' )

# %% rebin
# %%%  test

# note : a function is written for this in the next cell .
# here, it is for testing.
# if later decided to lower the temporal resolution of the pulse amplitude averages.

# output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')

output_name = "Pulse_Mean_Pressure_5s_Bins.pkl"


# --- Re-bin to 5 Seconds ---
# We take the mean of the 1s averages
df_5s = df_binned[['pressure']].resample('5s').mean()

df_5s[:5]
    # Out[60]: 
    #                        pressure
    # 2025-12-05 10:48:30  138.133109
    # 2025-12-05 10:48:35  131.820760
    # 2025-12-05 10:48:40  133.655785
    # 2025-12-05 10:48:45  136.755675
    # 2025-12-05 10:48:50  139.746592

# Calculate percentage from baseline based on this 5s resolution
baseline_5s = df_5s['pressure'].iloc[0]
df_5s['pressure_pct_of_baseline'] = (df_5s['pressure'] / baseline_5s) * 100

df_5s[:5]
    # Out[63]: 
    #                        pressure  pressure_pct_of_baseline
    # 2025-12-05 10:48:30  138.133109                100.000000
    # 2025-12-05 10:48:35  131.820760                 95.430242
    # 2025-12-05 10:48:40  133.655785                 96.758689
    # 2025-12-05 10:48:45  136.755675                 99.002821
    # 2025-12-05 10:48:50  139.746592                101.168064

# Re-calculate the relative time from gassing in decimal minutes.
# to be put at the x-axis of the plot later.
# (Assuming gassing_start_windows is still defined)
df_5s['Minutes_from_Gassing'] = (df_5s.index - gassing_start_windows).total_seconds() / 60.0


df_5s[:5]
    # Out[65]: 
    #                        pressure  pressure_pct_of_baseline  Minutes_from_Gassing
    # 2025-12-05 10:48:30  138.133109                100.000000             -0.950000
    # 2025-12-05 10:48:35  131.820760                 95.430242             -0.866667
    # 2025-12-05 10:48:40  133.655785                 96.758689             -0.783333
    # 2025-12-05 10:48:45  136.755675                 99.002821             -0.700000
    # 2025-12-05 10:48:50  139.746592                101.168064             -0.616667

df_5s.to_pickle( output_dir / output_name )


# %%% function 

def rebin_pulse_data(df_input, 
                     bpps=5, 
                     bin_size='5s', 
                     gassing_time = None, 
                     output_dir = None , 
                     output_name = None ):
    """
    Downsamples a 1s pulse dataframe to a lower resolution.
    
    Parameters:
    df_input: The 1s master dataframe (with datetime index)
    bpps : baseline pressure period in seconds.
        a prerequisite to calculate the percentage pressure from basline.
        this defines the duration of the baseline.
        example : 5 : average pressure for the first 5s of the recrding to be set as baseline pressure.
    bin_size: String for pandas resample (e.g., '5s', '10s', '1min')
    gassing_time: pd.Timestamp for relative time calculation
        Windows start-time of gassing.
    """
    # 1. Resample and average
    df_rebinned = df_input[['pressure_calibrated']].resample(bin_size).mean()
    
    # 2. Update Baseline Percentage
    # We re-calculate this so 100% is based on the new, larger bin
    
    # calculate the baseline pressure.
    # baseline is calculated based on the 1s binned data.
    baseline = df_input['pressure_calibrated'][:bpps].mean()
    # new pressure column / baseline-pressure.
    df_rebinned['pressure_pct_of_baseline'] = (df_rebinned['pressure_calibrated'] / baseline) * 100
    
    # 3. Update Relative Time
    if gassing_time is not None:
        # new ( rebinned ) index ( time-stamp )  -  gassing_start.
        diff = df_rebinned.index - gassing_time
        df_rebinned['Minutes_from_Gassing'] = diff.total_seconds() / 60.0
    
    # adding the attributes is not needed as the file retains its attributes from the previous step !

    out_file = output_dir / output_name
    df_rebinned.to_pickle(out_file)
    
    return df_rebinned

# --- Usage Example ---
# df_5s = rebin_pulse_data(df_binned, bin_size='5s', gassing_time=gassing_start_windows)


# %%% execute

df_pulse_binned_5s = rebin_pulse_data(
                                    df_input = df_pulse_binned_1s ,    # df_pulse_binned_1s
                                    bpps=5,
                                    bin_size='5s', 
                                    gassing_time= df_pulse_binned_1s.attrs['gassing_start_windows'],
                                    output_dir = source_dir,
                                    output_name = "df_pulse_binned_5s.pkl"
                                    )
#====


df_pulse_binned_5s[:4]
    # batch_3 , 1st
        # Out[44]: 
        #                      pressure_calibrated  pressure_pct_of_baseline  Minutes_from_Gassing
        # 2025-09-26 09:13:50                  NaN                       NaN            -31.166667
        # 2025-09-26 09:13:55           120.807912                 99.933552            -31.083333
        # 2025-09-26 09:14:00           122.263710                101.137803            -31.000000
        # 2025-09-26 09:14:05           119.905331                 99.186928            -30.916667
    
    # batch_3 , 3rd  
        # Out[40]: 
        #                      pressure_calibrated  pressure_pct_of_baseline  Minutes_from_Gassing
        # 2025-09-26 10:58:50           139.233086                104.273176             -1.666667
        # 2025-09-26 10:58:55           129.816166                 97.220742             -1.583333
        # 2025-09-26 10:59:00           133.110787                 99.688120             -1.500000
        # 2025-09-26 10:59:05           136.541748                102.257604             -1.416667
    
    # Out[31]: 
        #                        pressure  pressure_pct_of_baseline  Minutes_from_Gassing
        # 2025-12-05 10:48:30  138.133109                104.638295             -0.950000
        # 2025-12-05 10:48:35  131.820760                 99.856578             -0.866667
        # 2025-12-05 10:48:40  133.655785                101.246643             -0.783333
        # 2025-12-05 10:48:45  136.755675                103.594864             -0.700000

#===================
# batch-4 , 3rd : pressure-offset = 0  ( no need for calibration ).
# but just for compatibility, a duplicate column with the name 'pressure_calibrated' was added to them.

source_dir
    # Out[54]: WindowsPath('F:/OneDrive - Uniklinik RWTH Aachen/home_cage/Stellar_notocord_tse/analysis__telemetry/dataframe/batch_4/terminal/2512058__SN_921536130')

df_pulse_binned_5s['pressure_calibrated'] = df_pulse_binned_5s['pressure'] 
df_pulse_binned_1s['pressure_calibrated'] = df_pulse_binned_1s['pressure'] 

df_pulse_binned_5s.to_pickle( source_dir / "df_pulse_binned_5s.pkl" )
df_pulse_binned_1s.to_pickle( source_dir / "df_pulse_binned_1s.pkl" )

#====

df_pulse_binned_5s.head()
    # Out[51]: 
    #                       pressure  pressure_pct_of_baseline  Minutes_from_Gassing  pressure_calibrated
    # 2025-12-05 10:48:30 138.133109                104.638295             -0.950000           138.133109
    # 2025-12-05 10:48:35 131.820760                 99.856578             -0.866667           131.820760
    # 2025-12-05 10:48:40 133.655785                101.246643             -0.783333           133.655785
    # 2025-12-05 10:48:45 136.755675                103.594864             -0.700000           136.755675
    # 2025-12-05 10:48:50 139.746592                105.860537             -0.616667           139.746592


df_pulse_binned_1s.head()
    # Out[55]: 
    #                       pressure  Minutes_from_Gassing  pressure_calibrated
    # 2025-12-05 10:48:33        NaN             -0.900000                  NaN
    # 2025-12-05 10:48:34 138.133109             -0.883333           138.133109
    # 2025-12-05 10:48:35 129.885842             -0.866667           129.885842
    # 2025-12-05 10:48:36 126.532331             -0.850000           126.532331
    # 2025-12-05 10:48:37 133.489083             -0.833333           133.489083

#=================================

df_pulse_binned_5s['pressure'].plot()
plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\pulse\batch_3\2509262\average_5s_.pdf' )


# # Save the gassing time inside the dataframe object
# df_pulse_binned_5s.attrs['rec_start_windows'] = rec_start_windows
# df_pulse_binned_5s.attrs['gassing_start_windows'] = gassing_start_windows


# output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')

# run this, as earlier there was another output name given.
# output_name = "df_pulse_binned_5s.pkl"
# df_pulse_binned_5s.to_pickle( output_dir / output_name  )

# %% NaN

# batch_3 , 1st
df_pulse_binned_5s.tail
    # Out[149]: 
    # <bound method NDFrame.tail of                       pressure  pressure_pct_of_baseline  Minutes_from_Gassing
    # 2025-09-26 09:13:50        NaN                       NaN            -31.166667
    # 2025-09-26 09:13:55  90.807912                 99.911620            -31.083333
    # 2025-09-26 09:14:00  92.263710                101.513365            -31.000000
    # 2025-09-26 09:14:05  89.905331                 98.918553            -30.916667
    # 2025-09-26 09:14:10  98.588327                108.472040            -30.833333
    #                        ...                       ...                   ...
    # 2025-09-26 09:52:15        NaN                       NaN              7.250000
    # 2025-09-26 09:52:20        NaN                       NaN              7.333333
    # 2025-09-26 09:52:25        NaN                       NaN              7.416667
    # 2025-09-26 09:52:30        NaN                       NaN              7.500000
    # 2025-09-26 09:52:35        NaN                       NaN              7.583333
    
    # [466 rows x 3 columns]>

# %% read attributes

df_pulse_binned_5s.attrs
    # Out[53]: 
    # {'file_source': 'F:\\OneDrive - Uniklinik RWTH Aachen\\home_cage\\Stellar_notocord_tse\\save_notocord\\batch_4\\conversion\\sacrifice\\2512055__SN_920536131__.edf',
    #  'rec_start_windows': Timestamp('2025-12-05 10:13:23'),
    #  'gassing_start_windows': Timestamp('2025-12-05 10:14:24')}

df_pulse_binned_5s.attrs['rec_start_windows']
    # Out[50]: Timestamp('2025-12-05 10:48:33')
    
type(df_pulse_binned_5s.attrs['rec_start_windows'])
    # Out[19]: pandas.Timestamp

df_pulse_binned_5s.attrs['gassing_start_windows']
    # Out[51]: Timestamp('2025-12-05 10:49:27')


# %%

df_pulse_binned_5s['pressure_pct_of_baseline'].plot()

output_dir = Path( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\pulse\2512058_pulse' )
output_name = 'average_pulse_5s.pdf'
plt.savefig( output_dir / output_name  )


# %%

