

# %%

source_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
file_name='df_HR_10s.pkl'
df_HR_10s = pd.read_csv( source_dir / file_name )


output_dir = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')
output_name = "df_pulse_binned_5s.pkl"
df_pulse_binned_5s

# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%

df_HR_10s
df_pulse_binned_5s

title_text = 'batch_4 , 2512058__SN_921536130'

# %%


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
ax2.set_ylim(0, 120) # Percentage scale

# --- 5. Add Shading for HR Dropouts ---
# Using the logic from your plot.txt to mark missing data zones
nan_mask = df_HR_10s['Average_Heart_Rate_BPM'].isna()
for i in range(len(nan_mask) - 1):
    if nan_mask[i]:
        ax1.axvspan(hr_time_min[i], hr_time_min[i+1], 
                    color='grey', alpha=0.2, label='Signal Dropout' if i == 0 else "")

# --- 6. Add Clinical Annotations (from plot.txt) ---
events = {
    (2 + 57/60): "Loss of Consciousness",
    (4 + 0/60): "Apnea"
}

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

# %%

output_dir_plot = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\terminal\2512058__SN_921536130')
file_name = 'joint_2512058.pdf'
plt.savefig( output_dir_plot / file_name )


# %%



