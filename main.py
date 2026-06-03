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


import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import gc 
from Namelist.config_parser import LTMConfig
from IO.nc_handler import DataProvider
from IO.output_writer import TrajectoryWriter
from Initial.seeding import spawn
from Hydrodynamic.kernels import update_3d_safe
from Model.microplastic import get_polymer_density

def run():
    print("--- [PROCESS] INITIALIZING INPUT DATA ---")
    cfg = LTMConfig()
    io = DataProvider(cfg.config['Files']['cmems_dir'], cfg.config['Files']['era5_dir'])
    writer = TrajectoryWriter(cfg.config['Files']['output_dir'])
    sources = cfg.get_sources()
    start_sim, end_sim = cfg.get_simulation_times()
    dt = int(cfg.config['Simulation']['dt'])
    write_freq = int(cfg.config['Simulation'].get('write_frequency', 3600))
    mode = cfg.config['Simulation']['mode']
    project_name = cfg.config['Simulation'].get('project_name', 'LTM_Simulation')
    dim_str = cfg.config['Simulation'].get('dimension', '2.5D').upper()
    dim_mode = 2 if dim_str == '3DUVW' else (1 if dim_str == '3DUV' else 0)
    scheme_str = cfg.config['Simulation'].get('scheme', 'euler').lower()
    scheme_idx = 1 if scheme_str == 'rk2' else (2 if scheme_str == 'rk4' else 0)
    kh = float(cfg.config['Diffusion'].get('Kh', 1.0))
    beaching_prob = float(cfg.config['MarineDebris'].get('beaching_probability', 0.05))
    is_forward = dt > 0
    running_type = "forward" if is_forward else "backward"
    start_str = start_sim.strftime('%Y%m%d')
    end_str = end_sim.strftime('%Y%m%d')
    if mode == 'marine_debris':
        kz = 0.0  
        debris_type = cfg.config['MarineDebris'].get('type', 'macro_plastic')
        leeway_key = f"leeway_{debris_type}"
        leeway = float(cfg.config['MarineDebris'].get(leeway_key, 0.03))
        p_rho, r0_mm, bf_rho, bf_rate, shape_idx, ws_mode, static_ws = 1025.0, 1.0, 1350.0, 0.0, 0, 0, 0.0
    else:
        kz = float(cfg.config['Diffusion'].get('Kz', 0.0000001))
        leeway = 0.0  
        mp_cfg = cfg.config['Microplastic']
        ws_mode = 1 if mp_cfg.get('ws_type') == 'dynamic' else 0
        static_ws = float(mp_cfg.get('ws_static', 0.0001))
        bf_rate = float(mp_cfg.get('biofilm_growth_rate', 0.01))
        bf_rho = float(mp_cfg.get('biofilm_density', 1350.0))
        p_rho = get_polymer_density(mp_cfg['polymer_type'])
        r0_mm = float(mp_cfg['diameter_mm']) / 2.0
        shape_idx = 0 if mp_cfg['shape'] == 'sphere' else 1

    if mode == 'marine_debris':
        print(f"--- [INFO] MODE: {mode} | DIRECTION: {running_type.upper()} | SCHEME: {scheme_str.upper()} ---")
    else:
        print(f"--- [INFO] MODE: {mode} | DIMENSION: {dim_str} | DIRECTION: {running_type.upper()} | SCHEME: {scheme_str.upper()} ---")

    output_filename = f"{project_name}_{mode}_{running_type}_{start_str}_{end_str}"
    writer.create_file(output_filename, cfg)
    global_p_counter = 0
    p_data = {s['name']: {'lon': [], 'lat': [], 'z': [], 'active': [], 'bf': [], 'p_id': []} for s in sources}
    cur = start_sim
    total_steps = int(abs((end_sim - start_sim).total_seconds() / dt)) + 1
    print("--- [PROCESS] STARTING SIMULATION LOOP ---")
    pbar = tqdm(total=total_steps, desc="Particles evolve")
    while (is_forward and cur <= end_sim) or (not is_forward and cur >= end_sim):
        u, v, w, uw, vw = io.get_step_data(cur, mode=mode, dimension=dim_str)
        elapsed_seconds = int(abs((cur - start_sim).total_seconds()))
        for s in sources:
            name = s['name']
            if s['start_rel'] <= cur <= s['end_rel']: 
                nl, na, nz = spawn(s, cur, dt) 
                p_data[name]['lon'].extend(nl)
                p_data[name]['lat'].extend(na)
                p_data[name]['z'].extend([0.0] * len(nl) if mode == 'marine_debris' else nz)
                p_data[name]['active'].extend([1] * len(nl))
                p_data[name]['bf'].extend([0.0] * len(nl))
                new_ids = list(range(global_p_counter, global_p_counter + len(nl)))
                p_data[name]['p_id'].extend(new_ids)
                global_p_counter += len(nl)
            if p_data[name]['lon']:
                lo = np.array(p_data[name]['lon'], dtype=np.float64)
                la = np.array(p_data[name]['lat'], dtype=np.float64)
                zz = np.array(p_data[name]['z'], dtype=np.float64)
                act = np.array(p_data[name]['active'], dtype=np.int32)
                bf = np.array(p_data[name]['bf'], dtype=np.float64)
                update_3d_safe(lo, la, zz, act, bf, u, v, w, uw, vw, 
                               io.lon_c, io.lat_c, io.depth_levels, io.lon_w, io.lat_w, 
                               dt, leeway, kh, kz,
                               ws_mode, static_ws, p_rho, r0_mm, bf_rho, bf_rate, shape_idx,
                               io.bathy_depth, io.lon_b, io.lat_b, mode, 
                               scheme_idx, dim_mode, beaching_prob)
                p_data[name].update({'lon': lo.tolist(), 'lat': la.tolist(), 'z': zz.tolist(), 'active': act.tolist(), 'bf': bf.tolist()})
        if elapsed_seconds % write_freq == 0:
            all_lons, all_lats, all_zs, all_acts, all_pids = [], [], [], [], []
            for name in p_data:
                all_lons.extend(p_data[name]['lon'])
                all_lats.extend(p_data[name]['lat'])
                all_zs.extend(p_data[name]['z'])
                all_acts.extend(p_data[name]['active'])
                all_pids.extend(p_data[name]['p_id'])
            if all_lons:
                writer.write_step(output_filename, cur, all_lons, all_lats, all_zs, all_acts, all_pids)
        del u, v, w, uw, vw
        if elapsed_seconds % 86400 == 0: gc.collect()
        cur += pd.Timedelta(seconds=dt)
        pbar.update(1)
    pbar.close()
    print("--- [PROCESS] SIMULATION COMPLETED SUCCESSFULLY ---")
    print("--- [INFO] AUTHOR: Dr Dwiyoga Nugroho, RCO-BRIn 2026 ---")

if __name__ == "__main__":
    run()
