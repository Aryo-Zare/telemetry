

# %% sanity check

# random trace : this randomly checks trace slices throughout the whole recording period, to detect various types of abnormal signals.

import os
import random
import matplotlib.pyplot as plt

# %%%'

# Define your path (using raw string for Windows paths)
save_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\blood_pressure'

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