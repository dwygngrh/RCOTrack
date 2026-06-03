"""
Copyright 2026 Dwiyoga Nugroho/Research Center For Oceanology-BRIN Indonesia. All rights reserved.

Non-Commercial, No-Derivatives Academic License Terms:

1. License Grant: You are granted a non-exclusive, non-transferable, royalty-free
   license to run and execute this LTM code solely for non-commercial, personal,
   educational, or academic research purposes.

2. Restrictions:
   - Commercial Use: You may not use this software, its source code, or any
     oceanographic outputs generated directly by it for commercial purposes,
     corporate research, or financial gain without explicit written permission.
   - No Derivatives: You may not modify, alter, transform, or build upon this
     source code. You may not distribute modified versions of the code.
   - Attribution: Any public use, academic publication, or presentation of results
     utilizing this code must prominently credit the author: Dwiyoga Nugroho/
     Research Center For Oceanology-BRIN Indonesia.

3. Distribution: You may redistribute exact, unmodified copies of this software
   repository to others, provided this license file remains intact.
"""


import os
import numpy as np
from netCDF4 import Dataset, date2num

class DataProvider:
    def __init__(self, cmems_dir, era5_dir):
        self.cmems_dir = cmems_dir
        self.era5_dir = era5_dir
        
        # 1. Load Current (CMEMS) Grid Coordinates
        cmems_files = sorted([f for f in os.listdir(cmems_dir) if f.endswith('.nc')])
        if not cmems_files:
            raise FileNotFoundError(f"No CMEMS NetCDF files found in {cmems_dir}")
            
        sample_cmems = os.path.join(cmems_dir, cmems_files[0])
        with Dataset(sample_cmems, 'r') as ds:
            self.lon_c = np.asarray(ds.variables['longitude'][:], dtype=np.float64)
            self.lat_c = np.asarray(ds.variables['latitude'][:], dtype=np.float64)
            # Load all 50 depth levels for 3D interpolation
            self.depth_levels = np.asarray(ds.variables['depth'][:], dtype=np.float64)
            self.cmems_time_units = ds.variables['time'].units
            self.cmems_calendar = ds.variables['time'].calendar

        # 2. Load Bathymetry data
        bathy_path = os.path.join(cmems_dir, "bathy_meter.nc")
        if not os.path.exists(bathy_path):
            self.lon_b, self.lat_b = self.lon_c, self.lat_c
            self.bathy_depth = np.zeros((len(self.lat_c), len(self.lon_c)), dtype=np.float64)
        else:
            with Dataset(bathy_path, 'r') as ds_b:
                self.lon_b = np.asarray(ds_b.variables['longitude'][:], dtype=np.float64)
                self.lat_b = np.asarray(ds_b.variables['latitude'][:], dtype=np.float64)
                self.bathy_depth = np.asarray(ds_b.variables['deptho'][:], dtype=np.float64)

        # 3. Load Wind (ERA5) Grid Coordinates
        era5_files = sorted([f for f in os.listdir(era5_dir) if f.endswith('.nc')])
        if not era5_files:
            self.lon_w, self.lat_w = self.lon_c, self.lat_c
        else:
            sample_era5 = os.path.join(era5_dir, era5_files[0])
            with Dataset(sample_era5, 'r') as ds:
                self.lon_w = np.asarray(ds.variables['longitude'][:], dtype=np.float64)
                self.lat_w = np.asarray(ds.variables['latitude'][:], dtype=np.float64)
                t_var = 'time' if 'time' in ds.variables else 'valid_time'
                self.era5_time_units = ds.variables[t_var].units
                self.era5_calendar = getattr(ds.variables[t_var], 'calendar', 'gregorian')

    def get_step_data(self, timestamp, mode='microplastic', dimension='3DUVW'):
        """
        Retrieves Current and Wind data using Step-Slicing.
        Supports: '2.5D', '3DUV', and '3DUVW'
        """
        cmems_file_name = f"GLORYS_Y{timestamp.year}M{timestamp.month:02d}.nc"
        cmems_path = os.path.join(self.cmems_dir, cmems_file_name)
        
        if not os.path.exists(cmems_path):
            raise FileNotFoundError(f"Missing CMEMS file: {cmems_path}")

        with Dataset(cmems_path, 'r') as ds:
            ds.set_auto_maskandscale(True)
            target_time = date2num(timestamp, units=self.cmems_time_units, calendar=self.cmems_calendar)
            time_idx = np.argmin(np.abs(ds.variables['time'][:] - target_time))
            
            # --- Dimensionality Logic ---
            if dimension == '3DUVW':
                # Full 3D: Load uo, vo, and wo
                u = np.asarray(ds.variables['uo'][time_idx, :, :, :].filled(np.nan), dtype=np.float32)
                v = np.asarray(ds.variables['vo'][time_idx, :, :, :].filled(np.nan), dtype=np.float32)
                w = np.asarray(ds.variables['wo'][time_idx, :, :, :].filled(np.nan), dtype=np.float32)
            elif dimension == '3DUV':
                # 3D UV: Load uo and vo, but provide empty wo
                u = np.asarray(ds.variables['uo'][time_idx, :, :, :].filled(np.nan), dtype=np.float32)
                v = np.asarray(ds.variables['vo'][time_idx, :, :, :].filled(np.nan), dtype=np.float32)
                w = np.zeros_like(u, dtype=np.float32) 
            else:
                # 2.5D: Load surface layer only
                u = np.asarray(ds.variables['uo'][time_idx, 0:1, :, :].filled(np.nan), dtype=np.float32)
                v = np.asarray(ds.variables['vo'][time_idx, 0:1, :, :].filled(np.nan), dtype=np.float32)
                w = np.zeros_like(u, dtype=np.float32)

        # --- Wind Extraction (ERA5) ---
        era5_file_name = f"ERA5_Y{timestamp.year}M{timestamp.month:02d}.nc"
        era5_path = os.path.join(self.era5_dir, era5_file_name)
        
        if os.path.exists(era5_path):
            with Dataset(era5_path, 'r') as ds_w:
                ds_w.set_auto_maskandscale(True)
                t_var = 'time' if 'time' in ds_w.variables else 'valid_time'
                target_time_w = date2num(timestamp, units=self.era5_time_units, calendar=self.era5_calendar)
                time_idx_w = np.argmin(np.abs(ds_w.variables[t_var][:] - target_time_w))
                uw = np.asarray(ds_w.variables['u10'][time_idx_w, :, :].filled(np.nan), dtype=np.float32)
                vw = np.asarray(ds_w.variables['v10'][time_idx_w, :, :].filled(np.nan), dtype=np.float32)
        else:
            uw = np.zeros((len(self.lat_w), len(self.lon_w)), dtype=np.float32)
            vw = np.zeros_like(uw)

        return u, v, w, uw, vw
