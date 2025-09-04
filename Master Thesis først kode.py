#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries
import h5py          # For reading the HDF5 file
import numpy as np   # For numerical operations and array handling
import matplotlib.pyplot as plt  # For plotting

# Open the HDF5 file. Adjust the path if necessary.
filename = r"E:\Petroula Zygogianni\Master thesis\NorSat-1-POBC-LPR-20180315.tar/NorSat-1-POBC-LPR-20180315_00-c.h5f"
f = h5py.File(filename, 'r')

# Print out the keys (top-level groups/datasets) in the file to see its structure.
print("Keys in the file:", list(f.keys()))


# In[2]:


import h5py

filename = r"E:\Petroula Zygogianni\Master thesis\NorSat-1-POBC-LPR-20180315.tar/NorSat-1-POBC-LPR-20180315_00-c.h5f"

def explore_h5(group, prefix=''):
    """
    Εκτυπώνει όλα τα datasets και groups με το όνομα, το σχήμα και τον τύπο τους
    """
    for key in group.keys():
        item = group[key]
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            print(f"Dataset: {name}, shape: {item.shape}, dtype: {item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"Group: {name}")
            explore_h5(item, prefix=name)  # επανάκληση για subgroups

with h5py.File(filename, 'r') as f:
    print("Top-level keys:", list(f.keys()))
    for key in f.keys():
        explore_h5(f[key], prefix=key)


# In[3]:


import h5py
import numpy as np

file_path = r"E:\Petroula Zygogianni\Master thesis\NorSat-1-POBC-LPR-20180315.tar/NorSat-1-POBC-LPR-20180315_00-c.h5f"

with h5py.File(file_path, 'r') as f:
    # Loop through each bias dataset
    for i in range(1, 5):
        bias_data = f[f"m-NLP/bias{i}"][:]
        
        # Get unique voltages (rounded for stability)
        unique_vals = np.unique(np.round(bias_data, 3))
        
        print(f"Needle {i} bias values:")
        print(unique_vals)
        print("Min:", np.min(bias_data), "Max:", np.max(bias_data))
        print()


# In[4]:


import h5py
import numpy as np
import os

# -------------------------------
# Physical constants
# -------------------------------
e = 1.602e-19       # Elementary charge [C]
kB = 1.381e-23      # Boltzmann constant [J/K]
me = 9.109e-31      # Electron mass [kg]
Te = 3000           # Assumed electron temperature in Kelvin (~0.26 eV)
beta = 0.5          # Exponent from OML theory

# -------------------------------
# Folder containing 24 HDF5 files
# -------------------------------
folder_path = r"E:\Petroula Zygogianni\Master thesis\NorSat-1-POBC-LPR-20180315.tar"

# -------------------------------
# Build file list (00 to 23)
# -------------------------------
file_list = [
    os.path.join(folder_path, f"NorSat-1-POBC-LPR-20180315_{i:02d}-c.h5f")
    for i in range(24)
]

# -------------------------------
# Scan through files to find global min/max
# -------------------------------
global_min, global_max = np.inf, -np.inf

for file_path in file_list:
    with h5py.File(file_path, 'r') as f:
        # Read currents and biases from probes 2,3,4
        currents = [f[f'm-NLP/current{i}'][:] for i in [2,3,4]]
        biases = [f[f'm-NLP/bias{i}'][0] for i in [2,3,4]]
        
        # Compute electron density for each probe
        ne_all = []
        for i in range(3):
            Ip = currents[i]
            Vb = biases[i]
            C = e * np.sqrt(kB * Te / (2 * np.pi * me))  # proportionality constant
            ne_i = Ip / (C * (1 + (e * Vb) / (kB * Te))**beta)
            ne_all.append(ne_i)
        
        # Average over probes 2,3,4
        ne_avg = np.mean(ne_all, axis=0)
        
        # Update global min and max
        global_min = min(global_min, np.min(ne_avg))
        global_max = max(global_max, np.max(ne_avg))

# -------------------------------
# Print results
# -------------------------------
print("Global electron density range across all 24 files:")
print("Min =", global_min)
print("Max =", global_max)


# In[5]:


import h5py
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define physical constants
# -------------------------------
e = 1.602e-19       # Elementary charge [C]
kB = 1.381e-23      # Boltzmann constant [J/K]
me = 9.109e-31      # Electron mass [kg]
Te = 3000           # Assumed electron temperature in Kelvin (~0.26 eV)
A = 1.0             # Probe area in arbitrary units
beta = 0.5          # Exponent from OML theory

# -------------------------------
# 2. File path
# -------------------------------
file_path = r"E:\Petroula Zygogianni\Master thesis\NorSat-1-POBC-LPR-20180315.tar/NorSat-1-POBC-LPR-20180315_01-c.h5f"

# -------------------------------
# 3. Open HDF5 file and read data
# -------------------------------
with h5py.File(file_path, 'r') as f:
    timestamps = f['m-NLP/timestamp'][:]
    currents = [f[f'm-NLP/current{i}'][:] for i in range(2, 5)]  # only probes 2,3,4
    biases = [f[f'm-NLP/bias{i}'][0] for i in range(2, 5)]       # only probes 2,3,4

# -------------------------------
# 4. Convert timestamps to UTC datetime objects
# -------------------------------
t_utc = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]

# -------------------------------
# 5. Compute electron density for each needle
# -------------------------------
C = e * A * np.sqrt(kB * Te / (2 * np.pi * me))  # proportionality constant
ne_all = []

for Ip, Vb in zip(currents, biases):
    ne_i = Ip / (C * (1 + (e * Vb) / (kB * Te))**beta)
    ne_all.append(ne_i)

# -------------------------------
# 6. Average over the 3 needles
# -------------------------------
ne_avg = np.mean(ne_all, axis=0)

# Clip negative values (electron density cannot be < 0)
ne_avg = np.clip(ne_avg, 0, None)

# -------------------------------
# 7. Plot full time series with fixed y-axis
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(t_utc, ne_avg, color='black')
plt.xlabel('Time (UTC)')
plt.ylabel('Electron density (relative units)')
plt.title('Electron Density vs Time from NorSat-1 m-NLP (Probes 2–4)')
plt.ylim(-1.8e10, 7.3e12)  # <-- fixed global scale
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




