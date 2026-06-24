
# =>  file_dataframe.py : to load the data.
# edited serial-numers are mentioned here.
# event data :
    # F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry  \  PO87_Auswertung_Thomas__.xlsx


# %% annotation
# %%% batch_3 , 1st

title_text = 'batch_3 , 2509262__SN_921336130'

events = {
    # (): "Loss of Consciousness",
    (4 + 50/60): "Apnea",
    (6 + 20/60): 'Determination of death'
}


# %%% batch_3 , 2nd

title_text = 'batch_3 , 2509264__SN_920336131'  

events = {
    (2 + 15/60): "Loss of Consciousness",
    (4 + 23/60): "Apnea" ,
    (6 + 35/60): 'Determination of death'
}


# %%% batch_3 , 3rd

title_text = 'batch_3 , 2509265__SN_920536131'  

events = {
    (2 + 17/60): "Loss of Consciousness",
    (3 + 50/60): "Apnea",
    (6 + 36/60): 'Determination of death'
}

# %%% batch_4 , 1st

title_text = 'batch_4 , 2512054__SN_920336131'

events = {
    (1 + 45/60): "Loss of Consciousness",
    (4 + 35/60): "Apnea" ,
    (6 + 41/60): 'Determination of death'
}


# %%% batch_4 , 2nd

title_text = 'batch_4 , 2512055__SN_920536131'

events = {
    (3 + 10/60): "Loss of Consciousness",
    (4 + 50/60): "Apnea" ,
    (6 + 50/60): 'Determination of death'
}


# %%% batch_4 , 3rd

title_text = 'batch_4 , 2512058__SN_921536130'

events = {
    (2 + 57/60): "Loss of Consciousness",
    (4 + 0/60): "Apnea",
    (7 + 5/60): 'Determination of death'
}

# %% joint-plot

# this plots HR & pulse-pressure : for files having both traces.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%% data

# the following dat are needed
        # check above to get where they are loaded from.
    # df_HR_10s
    # df_pulse_binned_5s


# %%% joint


# --- 1. Extract Metadata and Setup Time ---
# Get the shared gassing anchor from your pickled attributes
gassing_start = df_HR_10s.attrs['gassing_start_windows']

# Calculate X-axis (Minutes) for Heart Rate
# We align Bin_Start to the gassing anchor
hr_time_min = (df_HR_10s['Bin_Start'] - gassing_start).dt.total_seconds() / 60.0

# Pulse time is already calculated in your pulse dataframe
pulse_time_min = df_pulse_binned_5s['Minutes_from_Gassing']

# --- 2. Initialize Dual-Axis Plot ---
fig, ax1 = plt.subplots(figsize=(14, 7))

# --- 3. Plot Left Axis: Heart Rate (Blue) ---
color_hr = '#1f77b4' # Standard blue
line1, = ax1.plot(hr_time_min, df_HR_10s['Average_Heart_Rate_BPM'], 
                  color=color_hr, linewidth=2, label='Average Heart Rate (BPM)')
ax1.set_ylabel("Average Heart Rate (BPM)", fontsize=12, fontweight='bold', color=color_hr, loc='top')
ax1.tick_params(axis='y', labelcolor=color_hr)
ax1.set_ylim(-10, 420)

# --- 4. Plot Right Axis: Pulse Pressure (Red) ---
ax2 = ax1.twinx() # Create the secondary axis
color_pulse = 'tab:red'
line2, = ax2.plot(pulse_time_min, df_pulse_binned_5s['pressure_pct_of_baseline'], 
                  color=color_pulse, linewidth=1.5, alpha=0.7, label='Average Pulse-Pressure (% of Baseline)')
ax2.set_ylabel("Average Pulse Pressure (% of Baseline)", fontsize=12, fontweight='bold', color=color_pulse, loc='top')
ax2.tick_params(axis='y', labelcolor=color_pulse)
ax2.set_ylim(0, 130) # Percentage scale

# --- 5. Add Shading for HR Dropouts ---
# Using the logic from your plot.txt to mark missing data zones
nan_mask = df_HR_10s['Average_Heart_Rate_BPM'].isna()
for i in range(len(nan_mask) - 1):
    if nan_mask[i]:
        ax1.axvspan(hr_time_min[i], hr_time_min[i+1], 
                    color='grey', alpha=0.2, label='Signal Dropout' if i == 0 else "")

# --- 6. Add Clinical Annotations (from plot.txt) ---
for x_min, label in events.items():
    ax1.axvline(x=x_min, color='darkred', linestyle='--', alpha=0.8) # [cite: 205]
    ax1.text(x_min, 400, f"  {label}", # [cite: 206, 207]
             rotation=90, verticalalignment='top', horizontalalignment='right', # [cite: 207]
             color='darkred', fontweight='bold', fontsize=12) # [cite: 208]

# --- 7. Aesthetics & Legend ---
ax1.set_title(f'Terminal Circulatory Profile: CO2 Induction'
              f'\n {title_text} '
              '\n HR/10s , pulse_pressure/5s',
              fontsize=16)

ax1.set_xlabel("Time from Start of Induction (Minutes)", fontsize=12, loc='right')
# ax1.set_xlim(pulse_time_min.min(), pulse_time_min.max())
ax1.set_xlim( left=0, right=None )
ax1.grid(True, which='both', linestyle=':', alpha=0.5) # [cite: 209]

# Merge legends from both axes into one box
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', frameon=True, shadow=True)

plt.tight_layout()
plt.show()

# %%% lim

plt.xlim( right=6.5 )

# batch-3 , 1st , 2509262  :  right=6.5 

# %%% save 


#---- batch-3 , 1st
output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_3\terminal\2509262__SN_921336130')
file_name = 'joint_2509262_2.pdf'

#----


#---- batch-3 , 3rd
output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_3\terminal\2509265__SN_920536131')
file_name = 'joint_2509265.pdf'

#============================================================


#---- batch-4 , 3rd
output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512058__SN_921536130')
file_name = 'joint_2512058.pdf'


plt.savefig( output_dir_plot / file_name )


# %% single plot

# this only plots heart-rate : for those files with no pulse-pressure data.

# --- 1. Extract Metadata and Setup Time ---
# Get the shared gassing anchor from your pickled attributes
gassing_start = df_HR_10s.attrs['gassing_start_windows']

# Calculate X-axis (Minutes) for Heart Rate
# We align Bin_Start to the gassing anchor
hr_time_min = (df_HR_10s['Bin_Start'] - gassing_start).dt.total_seconds() / 60.0


# --- 2. Initialize Dual-Axis Plot ---
fig, ax = plt.subplots(figsize=(14, 7))

# --- 3. Plot Left Axis: Heart Rate (Blue) ---
color_hr = '#1f77b4' # Standard blue
ax.plot(hr_time_min, df_HR_10s['Average_Heart_Rate_BPM'], 
                  color=color_hr, linewidth=2, label='Average Heart Rate (BPM)')
ax.set_ylabel("Average Heart Rate (BPM)", fontsize=12, fontweight='bold', color=color_hr, loc='top')
ax.tick_params(axis='y', labelcolor=color_hr)
ax.set_ylim(-10, 420)


# --- 5. Add Shading for HR Dropouts ---
# Using the logic from your plot.txt to mark missing data zones
nan_mask = df_HR_10s['Average_Heart_Rate_BPM'].isna()
for i in range(len(nan_mask) - 1):
    if nan_mask[i]:
        ax.axvspan(hr_time_min[i], hr_time_min[i+1], 
                    color='grey', alpha=0.2, label='Signal Dropout' if i == 0 else "")

# --- 6. Add Clinical Annotations (from plot.txt) ---
for x_min, label in events.items():
    ax.axvline(x=x_min, color='darkred', linestyle='--', alpha=0.8) # [cite: 205]
    ax.text(x_min, 400, f"  {label}", # [cite: 206, 207]
             rotation=90, verticalalignment='top', horizontalalignment='right', # [cite: 207]
             color='darkred', fontweight='bold', fontsize=12) # [cite: 208]

# --- 7. Aesthetics & Legend ---
ax.set_title(f'Terminal Circulatory Profile: CO2 Induction'
              f'\n {title_text} '
              '\n HR/10s , pulse_pressure/5s',
              fontsize=16)

ax.set_xlabel("Time from Start of Induction (Minutes)", fontsize=12, loc='right')
ax.set_xlim( left=0, right=None )
ax.grid(True, which='both', linestyle=':', alpha=0.5) # [cite: 209]


plt.tight_layout()


# %%% save

#---- batch-3 , 2nd
output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_3\terminal\2509264__SN_920336131')
file_name = 'single_2509264.pdf'


#---- batch-4 , 1st
output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512054__SN_920336131')
file_name = 'single_2512054.pdf'

#---- batch-4 , 2nd
output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512055__SN_920536131')
file_name = 'single_2512055.pdf'


#=================================================
plt.savefig( output_dir_plot / file_name )


# %%'

