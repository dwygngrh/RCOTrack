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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Configuration ---
# Update this to match your actual NetCDF filename
NC_FILE = 'output/test_run_microplastic_forward_20240101_20240210.nc'
PRODUCE_ANIMATION = False  # Set to False to generate the .jpg figure
FRAME_SKIP = 2  
TRAIL_LIMIT = 20 

def plot_trajectories():
    if not os.path.exists(NC_FILE):
        print(f'Error: File {NC_FILE} not found.')
        return

    # 1. Load Data and Metadata
    ds = nc.Dataset(NC_FILE)
    project_name = getattr(ds, 'project_name', 'LTM').replace(" ", "_")
    mode = getattr(ds, 'mode', 'microplastic')
    dimension = getattr(ds, 'dimension', '3DUV')
    
    # Extract dt to detect direction (backward if negative)
    dt_val = getattr(ds, 'dt', 1)
    obs_time = ds.variables['time'][:]
    if obs_time.size == 0:
        print("Error: NetCDF file contains no data.")
        return
        
    direction = 'backward' if (dt_val < 0 or obs_time[0] > obs_time[-1]) else 'forward'
    
    time_units = ds.variables['time'].units
    calendar = getattr(ds.variables['time'], 'calendar', 'standard')
    times_dt = nc.num2date(obs_time, units=time_units, calendar=calendar)
    
    # Stability fix for cftime objects
    times_raw = times_dt.compressed() if hasattr(times_dt, 'compressed') else times_dt
    t_min, t_max = np.min(times_raw), np.max(times_raw)
    start_str, end_str = t_min.strftime('%Y%m%d'), t_max.strftime('%Y%m%d')
    
    obs_lon, obs_lat = ds.variables['lon'][:], ds.variables['lat'][:]
    obs_z, obs_pid = ds.variables['z'][:], ds.variables['particle_id'][:]
    
    # 2. Setup Map
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([obs_lon.min()-1.5, obs_lon.max()+1.5, 
                   obs_lat.min()-1.5, obs_lat.max()+1.5])
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    ax.gridlines(draw_labels=True, dms=True, alpha=0.2, zorder=4)

    # UPDATED: Colormap 'jet'
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=0, vmax=np.max(obs_z) if np.max(obs_z) > 0 else 100)
    title_base = f"{project_name} {mode} {dimension} {direction} {start_str}, {end_str}"
    
    # 3. Plotting Logic
    if not PRODUCE_ANIMATION:
        # --- STATIC FIGURE LOGIC ---
        unique_pids = np.unique(obs_pid).astype(int)
        for pid in unique_pids:
            idx = np.where(obs_pid == pid)[0]
            sort_idx = np.argsort(obs_time[idx])
            lons, lats, zs = obs_lon[idx][sort_idx], obs_lat[idx][sort_idx], obs_z[idx][sort_idx]
            
            # Draw persistent trails with reduced linewidth
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.6, zorder=4, linewidth=0.5)
            lc.set_array(zs)
            ax.add_collection(lc)
            
            # Add small dot at the final position
            ax.scatter(lons[-1], lats[-1], c=[zs[-1]], cmap=cmap, norm=norm, s=5, edgecolors='none', zorder=5)

        # ADDED: Vertical inverted colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
        cbar.set_label('Depth (m)', fontsize=12)
        cbar.ax.invert_yaxis() # Surface (0) at top
            
        ax.set_title(title_base)
        output_jpg = f"{project_name}_{mode}_{dimension}_{direction}_{start_str},{end_str}.jpg"
        plt.show()
        plt.savefig(output_jpg, dpi=300, bbox_inches='tight')
        print(f"Static figure produced: {output_jpg}")

    else:
        # --- ANIMATION LOGIC ---
        # scat represents the small dots for particles
        scat = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=5, edgecolors='none', zorder=6)
        # trail_coll represents the thinner trails
        trail_coll = LineCollection([], cmap=cmap, norm=norm, alpha=0.4, zorder=5, linewidth=0.5)
        ax.add_collection(trail_coll)
        anim_title = ax.set_title('', fontsize=12)
        particle_paths = {}

        def update(frame_time):
            f_idx = np.where(obs_time == frame_time)[0]
            if len(f_idx) == 0: return scat, trail_coll, anim_title
            pids, cur_lons, cur_lats, cur_zs = obs_pid[f_idx], obs_lon[f_idx], obs_lat[f_idx], obs_z[f_idx]
            all_segs, all_zs = [], []

            for i, pid in enumerate(pids):
                if pid not in particle_paths: particle_paths[pid] = {'ln': [], 'lt': [], 'z': []}
                particle_paths[pid]['ln'].append(cur_lons[i])
                particle_paths[pid]['lt'].append(cur_lats[i])
                particle_paths[pid]['z'].append(cur_zs[i])
                if len(particle_paths[pid]['ln']) > TRAIL_LIMIT:
                    for key in ['ln', 'lt', 'z']: particle_paths[pid][key].pop(0)

                lns, lts = np.array(particle_paths[pid]['ln']), np.array(particle_paths[pid]['lt'])
                if len(lns) > 1:
                    pts = np.array([lns, lts]).T.reshape(-1, 1, 2)
                    all_segs.extend(np.concatenate([pts[:-1], pts[1:]], axis=1))
                    all_zs.extend(np.array(particle_paths[pid]['z'])[1:])

            trail_coll.set_segments(all_segs)
            trail_coll.set_array(np.array(all_zs))
            scat.set_offsets(np.c_[cur_lons, cur_lats])
            scat.set_array(cur_zs)
            anim_title.set_text(f"{title_base}\nTime: {nc.num2date(frame_time, time_units).strftime('%Y-%m-%d %H:%M')}")
            return scat, trail_coll, anim_title

        frames = np.unique(obs_time)[::FRAME_SKIP] if direction == 'forward' else np.unique(obs_time)[::-FRAME_SKIP]
        ani = FuncAnimation(fig, update, frames=frames, blit=True)
        output_gif = f"{project_name}_{mode}_{dimension}_{direction}_{start_str},{end_str}.gif"
        ani.save(output_gif, writer=PillowWriter(fps=12))
        print(f"Animation produced: {output_gif}")

    ds.close()

if __name__ == '__main__':
    plot_trajectories()
