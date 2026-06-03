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
import netCDF4 as nc

class TrajectoryWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

    def create_file(self, name, cfg_ref):
        # Initializes the NetCDF file with dimensions, variables, and global metadata.
        file_path = os.path.join(self.output_dir, f"{name}.nc")
        
        # --- Metadata Extraction ---
        sim_cfg = cfg_ref.config['Simulation']
        rel_cfg = cfg_ref.config['Release']
        
        proj_name = sim_cfg.get('project_name', 'LTM_Project')
        start_sim = sim_cfg.get('start_simulation')
        end_sim = sim_cfg.get('end_simulation')
        dimension = sim_cfg.get('dimension', '2.5D')
        time_step = sim_cfg.get('dt')
        
        start_rel = rel_cfg.get('start_date_release')
        end_rel = rel_cfg.get('end_date_release')
        
        running_type = "forward" if int(time_step) > 0 else "backward"
        mode = sim_cfg.get('mode')

        with nc.Dataset(file_path, 'w', format='NETCDF4') as root:
            root.createDimension('obs', None) 

            time_var = root.createVariable('time', 'f8', ('obs',))
            time_var.units = "seconds since 1970-01-01 00:00:00"
            time_var.calendar = "gregorian"
            
            # FIX: Use 'f8' (float64) for coordinates to ensure sub-meter precision 
            # and prevent accumulation of rounding errors during long simulations.
            root.createVariable('lon', 'f8', ('obs',))
            root.createVariable('lat', 'f8', ('obs',))
            root.createVariable('z', 'f8', ('obs',))
            
            root.createVariable('status', 'i1', ('obs',))
            root.createVariable('particle_id', 'i4', ('obs',))

            # --- GLOBAL METADATA ATTRIBUTES ---
            root.project_name = proj_name
            root.start_date_simulation = start_sim
            root.end_date_simulation = end_sim
            root.start_particle_release = start_rel
            root.end_particle_release = end_rel
            root.dimension = dimension
            root.time_step = time_step
            root.direction = running_type
            
            # --- Standard Metadata ---
            root.Author = "Dr Dwiyoga Nugroho, RCO-BRIn 2026"
            root.mode = mode

    def write_step(self, file_name, time, lons, lats, depths, status, p_ids):
        # Appends a new time step of data to the NetCDF file.
        file_path = os.path.join(self.output_dir, f"{file_name}.nc")
        if not lons: return
        
        with nc.Dataset(file_path, 'a') as root:
            start_idx = len(root.dimensions['obs'])
            count = len(lons)
            end_idx = start_idx + count
            
            t_val = nc.date2num(time, units=root.variables['time'].units, 
                                calendar=root.variables['time'].calendar)
            
            root.variables['time'][start_idx:end_idx] = [t_val] * count
            root.variables['lon'][start_idx:end_idx] = lons
            root.variables['lat'][start_idx:end_idx] = lats
            root.variables['z'][start_idx:end_idx] = depths
            root.variables['status'][start_idx:end_idx] = status
            root.variables['particle_id'][start_idx:end_idx] = p_ids
