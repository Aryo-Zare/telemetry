

# %% HR_plot

# simple overview plot.

df_final['Average_Heart_Rate_BPM'].plot()
plt.title('batch_4 \n 2512054__SN_920336131 ')
plt.xlabel('bin' , loc='right')
plt.ylabel('Average_Heart_Rate_BPM' , loc='top')

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\HR.pdf' )

# %%'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%


# --- 1. Setup Parameters ---
bin_duration_sec = 10  # As specified for this run
# Calculate minutes for the x-axis
# df_final.index represents the bin numbers [0, 1, 2, ...]
time_minutes = (df_final.index * bin_duration_sec) / 60

# Define clinical events in "Minutes:Seconds" converted to decimal minutes
events = {
    (1 + 45/60): "Loss of Consciousness",
    (4 + 35/60): "Apnea",
    ( 358 / 60): "Ventricular Flutter"   # visually defined on the ECG signa.
}

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
             '\n batch_4  _ 2512054__SN_920336131', 
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

# %%

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\batch_4\HR_3.pdf' )


# %%



