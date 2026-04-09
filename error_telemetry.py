

# %%

# the error in the cell below was not reproducible !!
# this was related to file not beig able to be read properly.
    # the whole array was replaced with 0.s !

# %%

f.close()

file_path = r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512058__SN_921536130__.edf'

f = pyedflib.EdfReader(file_path)

# something like a warning is returned !
ecg = f.readSignal(1)    
    # read -1, less than 247000 requested!!!


ecg.sum()
# Out[99]: np.float64(0.0)


ecg[:20]
# Out[104]: 
# array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0.])


# %%

f = pyedflib.EdfReader(file_path)
labels = f.getSignalLabels()
n_channels = f.signals_in_file

print(f"Analyzing File: {file_path}")
print("="*40)

for i in range(n_channels):
    # Get the number of samples the header *claims* exist
    n_samples = f.getNSamples()[i]
    sfreq = f.getSampleFrequency(i)
    
    try:
        # The 'digital=True' flag reads the raw integers directly
        # without applying calibration math.
        digital_sig = f.readSignal(i, digital=True)
        physical_sig = f.readSignal(i, digital=False)
        
        print(f"Index {i}: Label='{labels[i]}' | {sfreq} Hz")
        print(f"   - Samples Claimed: {n_samples}")
        print(f"   - Digital Sum:  {np.sum(digital_sig)}")
        print(f"   - Physical Sum: {np.sum(physical_sig)}")
        
        # Check if the first 100 samples are all zeros
        if np.all(digital_sig[:100] == 0):
            print("   - [!] Warning: Leading samples are all zeros.")
            
    except Exception as e:
        print(f"Index {i}: ERROR reading signal - {str(e)}")
    
    print("-" * 30)

f.close()

# %%

# Analyzing File: F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\save_notocord\batch_4\conversion\sacrifice\2512058__SN_921536130__.edf
# ========================================
# Index 0: Label='STE20a1.SN_92153' | 1.0 Hz
#    - Samples Claimed: 494
#    - Digital Sum:  1460298
#    - Physical Sum: 1485.5608000000002
# ------------------------------
# Index 1: Label='STE20a1.SN_92153' | 500.0 Hz
#    - Samples Claimed: 247000
#    - Digital Sum:  509749552
#    - Physical Sum: -11089.852918842513
#    - [!] Warning: Leading samples are all zeros.
# ------------------------------
# Index 2: Label='STE20a1.SN_92153' | 250.0 Hz
#    - Samples Claimed: 123500
#    - Digital Sum:  149349653
#    - Physical Sum: 10945722.329785947
#    - [!] Warning: Leading samples are all zeros.
# ------------------------------
# Index 3: Label='STE20a1.SN_92153' | 1.0 Hz
#    - Samples Claimed: 494
#    - Digital Sum:  180676
#    - Physical Sum: 18265.199999999997
# ------------------------------

# %%


f = pyedflib.EdfReader(file_path)
sfreq = int(f.getSampleFrequency(1)) # 500 Hz
total_samples_claimed = f.getNSamples()[1]

print("Scanning for signal onset...")
found_start = False
chunk_size = sfreq # 1 second per step

for start_sample in range(0, total_samples_claimed, chunk_size):
    try:
        # Read a small 1-second window instead of the whole file
        # This bypasses the 'Read -1' full-file error
        chunk = f.readSignal(1, start=start_sample, n=chunk_size, digital=True)
        
        # Check if this chunk has any non-zero data
        if np.any(chunk != 0):
            # Find the exact first non-zero index within this chunk
            local_index = np.where(chunk != 0)[0][0]
            absolute_index = start_sample + local_index
            print(f"Success! Data detected at sample: {absolute_index}")
            print(f"This is {absolute_index / sfreq:.2f} seconds into the recording.")
            found_start = True
            break
            
    except Exception as e:
        print(f"Read error at sample {start_sample}: {e}")
        break

if not found_start:
    print("Could not find any non-zero data in this channel.")

f.close()

# %%

# Scanning for signal onset...
# Success! Data detected at sample: 883
# This is 1.77 seconds into the recording.

# %% save

# to avoid possible future recurrence of the error, I saved the corresponding numpy arrays.

channel_0 = f.readSignal(0) 
channel_1 = f.readSignal(1) 
channel_2 = f.readSignal(2)     
channel_3 = f.readSignal(3)     


# %%


from pathlib import Path

# loss-less

save_path = Path(r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\dataframe\batch_4\terminal\2512058__SN_921536130')

# suppose ecg is your NumPy array
np.save( save_path / "channel_0.npy", channel_0 )

# to load it later
ecg_loaded = np.load("ecg_signal.npy")

# %%



