#DERIVE ELECTRON DENSITY FLUCTUATIONS THROUGH NORSAT-1 SATELLITE

import h5py
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt

# 1. Physical constants & probe geometry
e  = 1.602e-19
kB = 1.381e-23
me = 9.109e-31

d = 0.0005      # probe diameter [m]
L = 0.025       # probe length [m]
A = np.pi * d * L
Cgeom = 2/np.sqrt(np.pi)
B = Cgeom * A * e * np.sqrt(kB / (2*np.pi*me))
B2 = B**2

# 2. File path
file_path = r"filepath"


# 3. Load data for all 4 probes
with h5py.File(file_path, 'r') as f:
    timestamps = f['m-NLP/timestamp'][:]
    currents = [f[f'm-NLP/current{i}'][:] for i in range(1,5)]  # probes 1-4
    biases = [f[f'm-NLP/bias{i}'][0] for i in range(1,5)]

I = np.vstack(currents).T  # shape: (n_samples, 4)
V = np.array(biases)


# 4. Convert timestamps to UTC datetime
t_utc = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]


# 5. Compute electron density for each time point using 4 probes without assuming electron temperature
ne_all = np.empty(I.shape[0])
for t in range(I.shape[0]):
    I2 = I[t]**2
    slope, _ = np.polyfit(V, I2, 1)
    slope = max(slope, 0)  # avoid negative slopes
    ne_all[t] = np.sqrt((slope * kB) / (B2 * e))

# 6. Plot
plt.figure(figsize=(12,6))
plt.plot(t_utc, ne_all, color='black')
plt.xlabel('Time (UTC)')
plt.ylabel('Electron density [m^-3]')
plt.title('Electron Density vs Time from NorSat-1 m-NLP (4 Probes)')
plt.grid(True)
plt.tight_layout()
plt.xlim(t_utc[0], t_utc[-1])
plt.ylim(np.min(ne_all)*0.95, np.max(ne_all)*1.05)


#WHEN AND FOR HOW LONG A SATELLITE IS VISIBLE FROM A FIXED POINT ON EARTH

import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import os
import glob

# ------------------- CONFIG -------------------
folder_path = r"folderpath"
site_lat =       # Ground station latitude
site_lon =      # Ground station longitude
site_alt =      # Ground station altitude in meters
elevation_mask =  # Minimum elevation to count as visible in degrees

# WGS84 constants
a = 6378137.0 #Earth equatorial radius in meters
f = 1/298.257 #Earth`s flattening
e2 = f*(2-f) # Square of eccentricity

# ------------------- FUNCTIONS -------------------
def llh_to_ecef(lat_deg, lon_deg, h_m):              #From geodetic system (WGS84) to ECEF (Earth Centeres Earth Fixed) system
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    x = (N + h_m) * cos_lat * np.cos(lon)        # x points in the direction of prime meridian
    y = (N + h_m) * cos_lat * np.sin(lon)        # y is 90 degrees from the prim meridian
    z = ((1 - e2)*N + h_m) * sin_lat             # z points in the direction of true north 
    return np.vstack((x,y,z)).T

def ecef_to_enu(sat_ecef, site_lat, site_lon, site_ecef):   #From ECEF system to ENU (East-North-Up) system
    lat = np.radians(site_lat)
    lon = np.radians(site_lon)
    dx = sat_ecef[:,0] - site_ecef[0] #[:,0] all values of first column - x values
    dy = sat_ecef[:,1] - site_ecef[1] #[:,1] all values of second column - y values
    dz = sat_ecef[:,2] - site_ecef[2] #[:,2] all values of third column - z values
    R = np.array([[-np.sin(lon), np.cos(lon), 0],
                  [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                  [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]])
    enu = np.dot(np.vstack((dx,dy,dz)).T, R.T)
    return enu

def enu_to_el_az(enu):
    E,N,U = enu[:,0], enu[:,1], enu[:,2] # 0 - gives East component, 1 - gives North component and 2 - gives Up component
    horiz = np.hypot(E,N)  # horizontal distance, elevation and azimuth angles from geometery (article with coordinate conversion)
    elev = np.degrees(np.arctan2(U,horiz))
    az = np.degrees(np.arctan2(E,N)) % 360
    return elev, az

def find_passes(times, elev, mask_deg):
    above = elev >= mask_deg
    passes = []
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i-1
            seg_times = times[start:end+1]
            seg_elev = elev[start:end+1]
            passes.append({
                "AOS": seg_times[0],
                "LOS": seg_times[-1],
                "Duration_min": (seg_times[-1]-seg_times[0]).total_seconds() / 60.0,
                "MaxElev_deg": float(np.max(seg_elev)),
                "TCA": seg_times[np.argmax(seg_elev)]
            })
        i += 1
    return passes

# ------------------- MAIN SCRIPT -------------------
# Convert site once
site_ecef = llh_to_ecef(site_lat, site_lon, site_alt)[0]

# Find all .h5f files in folder
file_paths = glob.glob(os.path.join(folder_path, "*.h5f"))

for file_path in file_paths:
    try:
        with h5py.File(file_path, "r") as f:
            times_unix = f["NorSat-1/timestamp"][:]
            lat = f["NorSat-1/latitude"][:]
            lon = f["NorSat-1/longitude"][:]
            alt = f["NorSat-1/altitude"][:]  # km

        alt_m = alt * 1000.0
        times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in times_unix]

        sat_ecef = llh_to_ecef(lat, lon, alt_m)
        enu = ecef_to_enu(sat_ecef, site_lat, site_lon, site_ecef)
        elev, az = enu_to_el_az(enu)

        passes = find_passes(times, elev, elevation_mask)

        if len(passes) == 0:
            print(f"No passes detected above elevation mask for file {os.path.basename(file_path)}")
            continue

        df = pd.DataFrame(passes)

        # Create CSV name based on file name
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_csv = os.path.join(folder_path, f"{base_name}_passes.csv")

        # Avoid overwriting
        counter = 1
        base_csv = output_csv
        while os.path.exists(output_csv):
            output_csv = base_csv.replace(".csv", f"_{counter}.csv")
            counter += 1

        df.to_csv(output_csv, index=False)
        print(f"Done! Passes saved to {output_csv}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


#PLOTTING SATELLITE TRAJECTORY ON GLOBAL MAP AT A SPECIFIC TIME

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import h5py
import numpy as np
from datetime import datetime, timezone

def main():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()
    ax.coastlines()

    # Skibotn ground station
    skibotn_lat, skibotn_lon = 69.340, 20.313
    ax.plot(skibotn_lon, skibotn_lat, "ro", markersize=8,
            transform=ccrs.PlateCarree(), label="Skibotn")

    # Read your HDF5 file
    with h5py.File(r"filepath", "r") as f:
        times_unix = f["NorSat-1/timestamp"][:]   # UNIX timestamps
        lat = f["NorSat-1/latitude"][:]
        lon = f["NorSat-1/longitude"][:]

    # Convert timestamps → datetime
    times = np.array([datetime.fromtimestamp(t, tz=timezone.utc) for t in times_unix])

    # Filter for 1 May 2028
    start = datetime(2018, 5, 2, 0, 40, 0, tzinfo=timezone.utc) 
    end   = datetime(2018, 5, 2, 0, 52, 0, tzinfo=timezone.utc)
    mask = (times >= start) & (times <= end)

    # Plot trajectory
    ax.plot(lon[mask], lat[mask], color="blue", linewidth=1.5,
            transform=ccrs.PlateCarree(), label="NorSat-1 on 01.05.2028")

    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()




plt.show()
