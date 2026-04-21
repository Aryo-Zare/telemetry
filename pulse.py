

# %% explore

# random trace : this randomly checks trace slices throughout the whole recording period, to detect various types of abnormal signals.

import os
import random
import matplotlib.pyplot as plt

# %%%'

file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512058__SN_921536130__.edf'

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
                    start_s=480, 
                    duration_s=120
)

# %%%'

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\pulse\2512058_pulse\pulse_2512058_end_2_.pdf' )

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


# --- 1. Manual User Inputs ---
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512058__SN_921536130__.edf'

# Windows ( corrected ) Start Time from your exploration function/screenshots
rec_start_windows = pd.Timestamp("2025-12-05 10:48:33") 
# Windows Gassing Start Time
gassing_start_windows = pd.Timestamp("2025-12-05 10:49:27") 

# this will be the original dataframe
    # you can later resample for larger bins.
bin_size_str = '1s'

output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
output_name = "Pulse_Mean_Pressure_1s_Bins.pkl"

# %%% process

# --- 2. Load Signal ---
f = pyedflib.EdfReader(file_path)
sfreq = f.getSampleFrequency(2) # Pulse is Channel 2
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

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\pulse\2512058_pulse\average_2s_.pdf' )


# %% rebin
# %%%  test

# note : a function is writen for this in the next cell .
# here, it is for testing.
# if later decided to lower the temporal resolution of the pulse amplitude averages.

output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
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


# %%%' 

def rebin_pulse_data(df_input, bpps=5, bin_size='5s', gassing_time=None):
    """
    Downsamples a 1s pulse dataframe to a lower resolution.
    
    Parameters:
    df_input: The 1s master dataframe (with datetime index)
    bpps : baseline pressure period in seconds.
        a prerequisite tocalculate the percentage pressure from basline.
        this defines the duration of the baseline.
        example : 5 : average pressure for the first 5s of the recrding to be set as baseline pressure.
    bin_size: String for pandas resample (e.g., '5s', '10s', '1min')
    gassing_time: pd.Timestamp for relative time calculation
        Windows start-time of gassing.
    """
    # 1. Resample and average
    df_rebinned = df_input[['pressure']].resample(bin_size).mean()
    
    # 2. Update Baseline Percentage
    # We re-calculate this so 100% is based on the new, larger bin
    
    # calculate the baseline pressure.
    # baseline is calculated based on the 1s binned data.
    baseline = df_input['pressure'][:bpps].mean()
    # new pressure column / baseline-pressure.
    df_rebinned['pressure_pct_of_baseline'] = (df_rebinned['pressure'] / baseline) * 100
    
    # 3. Update Relative Time
    if gassing_time is not None:
        # new ( rebinned ) index ( time-stamp )  -  gassing_start.
        diff = df_rebinned.index - gassing_time
        df_rebinned['Minutes_from_Gassing'] = diff.total_seconds() / 60.0
        
    return df_rebinned

# --- Usage Example ---
# df_5s = rebin_pulse_data(df_binned, bin_size='5s', gassing_time=gassing_start_windows)


# %%% execute

df_5s = rebin_pulse_data(df_pulse_binned_1s, 
                         bpps=5,
                         bin_size='5s', 
                         gassing_time=gassing_start_windows)


df_5s[:4]
    # Out[31]: 
    #                        pressure  pressure_pct_of_baseline  Minutes_from_Gassing
    # 2025-12-05 10:48:30  138.133109                104.638295             -0.950000
    # 2025-12-05 10:48:35  131.820760                 99.856578             -0.866667
    # 2025-12-05 10:48:40  133.655785                101.246643             -0.783333
    # 2025-12-05 10:48:45  136.755675                103.594864             -0.700000

df_pulse_binned_5s = df_5s.copy()

# Save the gassing time inside the dataframe object
df_pulse_binned_5s.attrs['rec_start_windows'] = rec_start_windows
df_pulse_binned_5s.attrs['gassing_start_windows'] = gassing_start_windows


output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
output_name = "df_pulse_binned_5s.pkl"
df_pulse_binned_5s.to_pickle( output_dir / output_name  )

# %% read attributes

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

