
# first run : expoore_telemetry.py : to
    # get info from the file.
    # plot it.
# this analyzes the terminal experiment ( during the sacrifice ).

# %% program-1

# extracts the information related to 
    # ECG peaks.
            # location along the x-axis.
            # properties ( height , width , ... ).
        # saves them in a dataframe.
    # signal drop-outs.
        # saves it in another dataframe.

import numpy as np
import pandas as pd
import pyedflib
from scipy import signal, stats, ndimage
from datetime import timedelta
from pathlib import Path

# %%%' path

# batch_4
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512054__SN_920336131__.edf'
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512055__SN_920536131__.edf'
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512058__SN_921536130__.edf'


# batch-3
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3\conversion\sacrifice\2509262__SN_921336130__.edf'
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3\conversion\sacrifice\2509264__SN_920336131__.edf'
file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_3\conversion\sacrifice\2509265__SN_920536131__.edf'


# save path
# output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512055__SN_920536131')
# output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
# output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_3\terminal\2509262__SN_921336130')
# output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_3\terminal\2509264__SN_920336131')
output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_3\terminal\2509265__SN_920536131')



# output_dir.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist

# ths is the corrected ( Windows ) acquisition start-time.
    # taken from : explore_telemetry.py
start_time = pd.Timestamp("2025-09-26 10:58:53")

# %%%'

#---- Load Raw Telemetry Data

# note : the file should not have been oepned before , otherwise :
    # OSError: F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512054__SN_920336131__.edf: 
        # file has already been opened
    # you should close the file if previously have been opened :
        # f_5.close()
f = pyedflib.EdfReader(file_path)
sfreq = f.getSampleFrequency(1)
ecg_raw = f.readSignal(1)
f.close()

#===========================================================
#---- Dropout Map (Dataframe-3)
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
df_dropouts.to_csv( output_dir / "Dropout_Map.csv", index=False)

#===========================================================
#---- Pre-processing & Filtering (Whole Signal)
# Fill dropouts with 0s to prevent 'ringing' artifacts in the filter [cite: 180, 181, 182]
ecg_filled = np.where(is_dropout_global, 0, ecg_raw)

# 1-80 Hz Band-pass filter 
# 'sos' (Second-Order Sections) is used for maximum numerical stability
sos_bp = signal.butter(N=4, Wn=[1.0, 80.0], btype='bandpass', fs=sfreq, output='sos')
ecg_filtered = signal.sosfiltfilt(sos=sos_bp, x=ecg_filled)


#===========================================================
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

#===========================================================
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

# Original EDF Start: 2025-12-05 10:37:41
# Corrected Windows Start: 2025-12-05 09:40:54
# Program 1 Complete.
# Master Peak Log: 946 ( me : initially : 1105 ) beats identified.
# Dropout Map saved for Program 2 analysis.


df_dropouts
        # Out[21]: 
        #    start_sample  end_sample              start_time                end_time
        # 0             0        1041 2025-12-05 09:40:54.000 2025-12-05 09:40:56.082
        # 1        226449      226999 2025-12-05 09:48:26.898 2025-12-05 09:48:27.998
    # note : if yo uopen it in excel, each cell may only show the minute-second !
        # you should click on the cell  =>  the function-bar on top will show a more complete version.
        # however, the sub-seconds are not visible there.

df_peaks.shape
    # Out[22]: (946, 5)

df_peaks[:10]
    # Out[23]: 
    #      sample_stamp               timestamp  amplitude_v  prominence_v  width_ms
    # 159         15080 2025-12-05 09:41:24.160     0.654954      0.764685  7.720026
    # 160         15166 2025-12-05 09:41:24.332     0.630342      0.910330  8.948962
    # 161         15253 2025-12-05 09:41:24.506     0.573779      0.809840  8.732825
    # 162         15340 2025-12-05 09:41:24.680     0.600332      0.845800  8.667174
    # 163         15427 2025-12-05 09:41:24.854     0.600466      0.850464  8.701622
    # 164         15513 2025-12-05 09:41:25.026     0.585453      0.856120  8.863439
    # 165         15598 2025-12-05 09:41:25.196     0.593502      0.870499  8.744148
    # 166         15685 2025-12-05 09:41:25.370     0.583278      0.851466  8.697559
    # 167         15772 2025-12-05 09:41:25.544     0.578515      0.842532  8.830270
    # 168         15858 2025-12-05 09:41:25.716     0.574684      0.832503  8.938483


# 2512055__SN_920536131
    # Original EDF Start: 2025-12-05 11:10:06
    # Corrected Windows Start: 2025-12-05 10:13:23
    # Program 1 Complete.
    # Master Peak Log: 1502 beats identified.
    # Dropout Map saved for Program 2 analysis.


# %% program-2

# bins the R-peaks in a custom time-period.
# calculates the heart-rate at each bin.
# saves it in a dataframe.

import pandas as pd
from datetime import timedelta

# %%% variables

#---- Load the Data Created by Program 1
source_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe')
df_peaks = pd.read_csv( source_dir / "Master_Peak_Log.csv.gz", parse_dates=['timestamp'])
df_dropouts = pd.read_csv( source_dir / "Dropout_Map.csv", parse_dates=['start', 'end'])

#---- save path
# this is the same as that of program_1.
    # so you do not need to change it.
output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512055__SN_920536131')
# output_dir.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist


#---- custom start_time
# Define your custom start time (e.g., 14:00:00 on the day of recording)
    # = start of gassing.
    # get it from your own table ( matches Windows-start-time ( unlike what the experimenter may have recorded )) : 
# If you want to use the original start time, just set this to None.
custom_start_time = pd.Timestamp("2025-09-26 11:00:30") 


# %%%'

#---- Set Flexible Timing Parameters
# these 2 are the same, with different 'types'.
    # you should keep them the same : change them simultansously.
# 's' should not be capital, otherwise :
    # ValueError: Invalid frequency: 10S. Failed to parse with error message: ValueError("Invalid frequency: S. 
        # Failed to parse with error message: KeyError('S'). 
        # Did you mean s?")
bin_size_str = "10s" 
bin_duration_sec = 10


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
    # this probably checks (loops) for every row in df_dropout !
    overlaps = df_dropouts[(bin_start < df_dropouts['end_time']) & (bin_end > df_dropouts['start_time'])]
    
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
df_final.to_csv( output_dir / output_filename, index=False)

print(f"Analysis complete. \nBins anchored to {custom_start_time if custom_start_time else 'recording start'}.")

# %%% out

# Analysis complete. 
#  Bins anchored to 2025-12-05 09:41:24.

df_final.shape
    # Out[29]: (42, 3)

df_final[:10]
    # Out[30]: 
    #             Bin_Start             Bin_End  Average_Heart_Rate_BPM
    # 0 2025-12-05 09:41:24 2025-12-05 09:41:34                   342.0
    # 1 2025-12-05 09:41:34 2025-12-05 09:41:44                   348.0
    # 2 2025-12-05 09:41:44 2025-12-05 09:41:54                   306.0
    # 3 2025-12-05 09:41:54 2025-12-05 09:42:04                   240.0
    # 4 2025-12-05 09:42:04 2025-12-05 09:42:14                   192.0
    # 5 2025-12-05 09:42:14 2025-12-05 09:42:24                   174.0
    # 6 2025-12-05 09:42:24 2025-12-05 09:42:34                   156.0
    # 7 2025-12-05 09:42:34 2025-12-05 09:42:44                   138.0
    # 8 2025-12-05 09:42:44 2025-12-05 09:42:54                   150.0
    # 9 2025-12-05 09:42:54 2025-12-05 09:43:04                   186.0

# 2512055__SN_920536131
    # Analysis complete. 
    # Bins anchored to 2025-12-05 10:14:24.

# %% attributes

# this should later be automated as part of Progrmas_1_2.

source_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
file_name='HR_Analysis_10s_from_1049.csv'
df_HR_10s = pd.read_csv( source_dir / file_name )


# Windows ( corrected ) Start Time from your exploration function/screenshots
rec_start_windows = pd.Timestamp("2025-12-05 10:48:33") 
# Windows Gassing Start Time
gassing_start_windows = pd.Timestamp("2025-12-05 10:49:27") 


df_HR_10s.attrs['rec_start_windows'] = rec_start_windows
    # Out[50]: Timestamp('2025-12-05 10:48:33')

df_HR_10s.attrs['gassing_start_windows'] = rec_start_windows

df_HR_10s.attrs['bin_size'] = '10s'

output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
output_name = 'df_HR_10s.pkl'
df_HR_10s.to_pickle( output_dir / output_name )


# a dictionary.
df_HR_10s.attrs
    # Out[69]: 
    # {'rec_start_windows': Timestamp('2025-12-05 10:48:33'),
    #  'gassing_start_windows': Timestamp('2025-12-05 10:48:33'),
    #  'bin_size': '10s'}

# it shows the same as above.
# import pprint
# pprint.pprint( df_HR_10s.attrs )

# %% plot__HR

# plot for heart-rate.

import matplotlib.pyplot as plt
%matplotlib qt
import pandas as pd
import numpy as np

# %%% variables

# for this, you only need to load :df_final : from program-2.
    # no need to load df_dropout.

#=====================================================================

# output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512055__SN_920536131')
# output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512058__SN_921536130')
# output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512058__SN_921536130')

# output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_3\terminal\2509262__SN_921336130')
# output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_3\terminal\2509264__SN_920336131')
output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_3\terminal\2509265__SN_920536131')



file_name = 'batch-3 , 2509265__SN_920536131'


# note : this is actually not a varialbe to be changed !!
    # this should be the same bin-size in whih the dataframe was created !
    # the dataframe ( df_final ) was created, calculating average heart-rate for every 10s.
    # so the x-axis of the plot should account for this.
    # to check the binning of the dataframe, you can print it & visually check the time-diff.
bin_duration_sec = 10  # As specified for this run

#======================================================================

# Define clinical events in "Minutes:Seconds" converted to decimal minutes
    # Loss of Consciousness & apnea : these are extracted from the excel file provided by Thomas.
        # F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry  \  PO87_Auswertung_Thomas__.xlsx
        # Zeit Start zu Bewusstlosigkeit Stoppuhr
            # this is independent of mismatch between Windows-time & experimenter's time !
    # Ventricular Flutter :
        # telemetry.py | function : plot_ecg_segment_2 ( from the start of gassing )  =>   get the start-time.


#=======================================================

# batch-4

# 2512054__SN_920336131
    # events = {
    #     (1 + 45/60): "Loss of Consciousness",
    #     (4 + 35/60): "Apnea",
    #     ( 358 / 60): "Ventricular Flutter"   # visually defined on the ECG signa.
    # }

# 2512055__SN_920536131
events = {
    (3 + 10/60): "Loss of Consciousness",
    (4 + 50/60): "Apnea",
    ( 405 / 60): "Ventricular Flutter"   # visually defined on the ECG signal.
}

# 2512058__SN_921536130
events = {
    (2 + 57/60): "Loss of Consciousness",
    (4 + 0/60): "Apnea",
    # ( 405 / 60): "First Ventricular Flutter"   # visually defined on the ECG signal.
}

#=======================================================

# batch-3

events = {
    # (2 + 57/60): "Loss of Consciousness",
    (4 + 50/60): "Apnea",
    # ( 405 / 60): "First Ventricular Flutter"   # visually defined on the ECG signal.
}

events = {
    (2 + 15/60): "Loss of Consciousness",
    (4 + 23/60): "Apnea",
    # ( 405 / 60): "First Ventricular Flutter"   # visually defined on the ECG signal.
}

events = {
    (2 + 17/60): "Loss of Consciousness",
    (3 + 50/60): "Apnea",
    # ( 405 / 60): "First Ventricular Flutter"   # visually defined on the ECG signal.
    # (6 + 35/60): 'confirmation of death'
}

# %%% overview

# simple overview plot.

df_final['Average_Heart_Rate_BPM'].plot()
plt.title('batch_4 \n 2512055__SN_920536131 ')
plt.xlabel('bin' , loc='right')
plt.ylabel('Average_Heart_Rate_BPM' , loc='top')


plt.savefig( output_dir_plot / 'HR__2512055__SN_920536131.pdf' )

# %%%'


# Calculate minutes for the x-axis
# df_final.index represents the bin numbers [0, 1, 2, ...]
time_minutes = (df_final.index * bin_duration_sec) / 60


# --- 2. Create the Plot ---
fig, ax = plt.subplots(figsize=(14, 7))

# Plot the Heart Rate trend
ax.plot(time_minutes, df_final['Average_Heart_Rate_BPM'], 
        linewidth=2, color='#1f77b4', label='Rat Heart Rate')

# --- 3. Shading Dropout Zones (Data Integrity) ---
# This identifies where Program-2 assigned NaN due to the Dropout Map
nan_mask = df_final['Average_Heart_Rate_BPM'].isna()
# We shade these areas in light grey to show why data is missing
for i in range(len(nan_mask) - 1):
    if nan_mask[i]:
        ax.axvspan(time_minutes[i], time_minutes[i+1], 
                   color='grey', alpha=0.3, label='Signal Dropout' if i == 0 else "")

# --- 4. Add Clinical Annotations ---
for x_min, label in events.items():
    # Vertical dashed line
    ax.axvline(x=x_min, 
               color='darkred', 
               linestyle='--', 
               alpha=0.8
               )
    
    # Text label
    # We place text at the top of the plot (y=400 )
    ax.text(x_min, # x-position of the vertical line.
            400, # y-value of where the text ends printing ( end of the phrase ).
            f"  {label}", 
            rotation=90, 
            verticalalignment='top', 
            horizontalalignment='right' ,   # whether if the text is printed on the left or right side of the vertical line.
            color='darkred', 
            fontweight='bold', 
            fontsize=15
            )

# --- 5. Aesthetics & Formatting ---
ax.set_title("Terminal Heart Rate Profile: CO2 Euthanasia Induction"
             f'\n {file_name }', 
             fontsize=16, pad=20)
ax.set_xlabel("Time from Start of Induction (Minutes)", fontsize=12 , loc='right' )
ax.set_ylabel("Average Heart Rate (BPM)", fontsize=12 , loc='top' )
ax.set_ylim(-10, 420)
ax.set_xlim(0, time_minutes.max())

# Add a clean grid and legend
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='lower left', frameon=True, shadow=True)

plt.tight_layout()
plt.show()

# %%% save


# save path
# output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512054__SN_920336131')
# output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512055__SN_920536131')
# output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512058__SN_921536130')



# output_dir.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist

plt.savefig( output_dir_plot / f'HR_profile_{file_name}_.pdf' )


# plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\HR_3.pdf' )


# %%'




