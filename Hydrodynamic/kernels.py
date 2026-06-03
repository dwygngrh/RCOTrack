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


import numpy as np
from numba import njit, prange
from Model.microplastic import calculate_3d_ws 

@njit
def interp(x, y, xg, yg, data):
    # Bilinear interpolation for 2D grids with strict boundary safety.
    if x <= xg[0] or x >= xg[-1] or y <= yg[0] or y >= yg[-1]: 
        return np.nan
    i = np.searchsorted(xg, x) - 1
    j = np.searchsorted(yg, y) - 1
    if i < 0 or j < 0 or i >= len(xg)-1 or j >= len(yg)-1: 
        return np.nan
    dx = (x - xg[i]) / (xg[i+1] - xg[i])
    dy = (y - yg[j]) / (yg[j+1] - yg[j])
    v00, v10, v01, v11 = data[j, i], data[j, i+1], data[j+1, i], data[j+1, i+1]
    
    # FIX: Synchronized land-mask check for standard NetCDF FillValues
    if v00 > 1e+20 or v10 > 1e+20 or v01 > 1e+20 or v11 > 1e+20 or np.isnan(v00):
        return np.nan
    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v10 + (1-dx)*dy*v01 + dx*dy*v11

@njit
def interp3d(x, y, z, xg, yg, zg, data):
    # Trilinear interpolation for Longitude, Latitude, and Depth.
    if z <= zg[0]: 
        return interp(x, y, xg, yg, data[0, :, :])
    if z >= zg[-1]: 
        return interp(x, y, xg, yg, data[-1, :, :])
    k = np.searchsorted(zg, z) - 1
    v_upper = interp(x, y, xg, yg, data[k, :, :])
    v_lower = interp(x, y, xg, yg, data[k+1, :, :])
    if np.isnan(v_upper): return v_lower
    if np.isnan(v_lower): return v_upper
    dz = (z - zg[k]) / (zg[k+1] - zg[k])
    return v_upper + (v_lower - v_upper) * dz

@njit
def get_uvw_total(lon, lat, depth, u, v, w, uw, vw, l_c, a_c, z_c, l_w, a_w, is_debris, leeway, dim_mode):
    # Calculates 3D velocity vectors based on dimension mode.
    if dim_mode == 0:
        uc = interp(lon, lat, l_c, a_c, u[0, :, :])
        vc = interp(lon, lat, l_c, a_c, v[0, :, :])
        wc = 0.0
    elif dim_mode == 1:
        uc = interp3d(lon, lat, depth, l_c, a_c, z_c, u)
        vc = interp3d(lon, lat, depth, l_c, a_c, z_c, v)
        wc = 0.0
    else:
        uc = interp3d(lon, lat, depth, l_c, a_c, z_c, u)
        vc = interp3d(lon, lat, depth, l_c, a_c, z_c, v)
        wc = interp3d(lon, lat, depth, l_c, a_c, z_c, w)
    
    if np.isnan(uc) or np.isnan(vc):
        return np.nan, np.nan, np.nan
    
    ut, vt, wt = uc, vc, wc
    if is_debris:
        u_wind = interp(lon, lat, l_w, a_w, uw)
        v_wind = interp(lon, lat, l_w, a_w, vw)
        if not np.isnan(u_wind) and not np.isnan(v_wind):
            ut += leeway * u_wind
            vt += leeway * v_wind
    return ut, vt, wt

@njit(parallel=True)
def update_3d_safe(lons, lats, depths, active_mask, biofilms, u, v, w, uw, vw, 
                   l_c, a_c, z_c, l_w, a_w, dt, leeway, kh, kz,
                   ws_mode, static_ws, p_rho, r0_mm, bf_rho, bf_rate, shape_idx,
                   bathy_depth, l_b, a_b, mode, scheme_idx, dim_mode, beaching_prob=0.05):
    # Consolidated Lagrangian kernel with high-order numerical robustness.
    DEG_TO_M = 111320.0
    abs_dt = np.abs(dt)
    is_backward = dt < 0
    is_debris = (mode == 'marine_debris')

    for i in prange(len(lons)):
        if active_mask[i] == 0: continue
        
        # FIX: Explicit initialization to prevent uninitialized variables in parallel loops
        adv_lon, adv_lat, adv_z = 0.0, 0.0, 0.0
        cos_lat = np.cos(np.radians(lats[i]))

        # --- ADVECTION STEP ---
        if scheme_idx == 0: # Euler Forward
            u1, v1, w1 = get_uvw_total(lons[i], lats[i], depths[i], u, v, w, uw, vw, l_c, a_c, z_c, l_w, a_w, is_debris, leeway, dim_mode)
            if np.isnan(u1): active_mask[i] = 0; continue
            adv_lon, adv_lat, adv_z = (u1*dt)/(DEG_TO_M*cos_lat), (v1*dt)/DEG_TO_M, w1*dt

        elif scheme_idx == 1: # RK2
            u1, v1, w1 = get_uvw_total(lons[i], lats[i], depths[i], u, v, w, uw, vw, l_c, a_c, z_c, l_w, a_w, is_debris, leeway, dim_mode)
            if np.isnan(u1): active_mask[i] = 0; continue
            lp, ap, zp = lons[i]+(u1*dt)/(DEG_TO_M*cos_lat), lats[i]+(v1*dt)/DEG_TO_M, depths[i]+w1*dt
            u2, v2, w2 = get_uvw_total(lp, ap, zp, u, v, w, uw, vw, l_c, a_c, z_c, l_w, a_w, is_debris, leeway, dim_mode)
            if np.isnan(u2): u2, v2, w2 = u1, v1, w1
            adv_lon, adv_lat, adv_z = 0.5*(u1+u2)*dt/(DEG_TO_M*cos_lat), 0.5*(v1+v2)*dt/DEG_TO_M, 0.5*(w1+w2)*dt

        else: # RK4
            u1, v1, w1 = get_uvw_total(lons[i], lats[i], depths[i], u, v, w, uw, vw, l_c, a_c, z_c, l_w, a_w, is_debris, leeway, dim_mode)
            if np.isnan(u1): active_mask[i] = 0; continue
            l2, a2, z2 = lons[i]+(u1*dt/2)/(DEG_TO_M*cos_lat), lats[i]+(v1*dt/2)/DEG_TO_M, depths[i]+w1*dt/2
            u2, v2, w2 = get_uvw_total(l2, a2, z2, u, v, w, uw, vw, l_c, a_c, z_c, l_w, a_w, is_debris, leeway, dim_mode)
            if np.isnan(u2): u2, v2, w2 = u1, v1, w1
            l3, a3, z3 = lons[i]+(u2*dt/2)/(DEG_TO_M*cos_lat), lats[i]+(v2*dt/2)/DEG_TO_M, depths[i]+w2*dt/2
            u3, v3, w3 = get_uvw_total(l3, a3, z3, u, v, w, uw, vw, l_c, a_c, z_c, l_w, a_w, is_debris, leeway, dim_mode)
            if np.isnan(u3): u3, v3, w3 = u2, v2, w2
            l4, a4, z4 = lons[i]+(u3*dt)/(DEG_TO_M*cos_lat), lats[i]+(v3*dt)/DEG_TO_M, depths[i]+w3*dt
            u4, v4, w4 = get_uvw_total(l4, a4, z4, u, v, w, uw, vw, l_c, a_c, z_c, l_w, a_w, is_debris, leeway, dim_mode)
            if np.isnan(u4): u4, v4, w4 = u3, v3, w3
            adv_lon = (dt/6)*(u1+2*u2+2*u3+u4)/(DEG_TO_M*cos_lat)
            adv_lat = (dt/6)*(v1+2*v2+2*v3+v4)/DEG_TO_M
            adv_z = (dt/6)*(w1+2*w2+2*w3+w4)

        # --- DIFFUSION & UPDATE ---
        r_lon, r_lat = np.random.standard_normal(), np.random.standard_normal()
        diff_lon = (np.sqrt(2 * kh * abs_dt) * r_lon) / (DEG_TO_M * cos_lat)
        diff_lat = (np.sqrt(2 * kh * abs_dt) * r_lat) / DEG_TO_M
        
        next_lon, next_lat = lons[i] + adv_lon + diff_lon, lats[i] + adv_lat + diff_lat
        
        # Check if the new position is on land
        if np.isnan(interp(next_lon, next_lat, l_c, a_c, u[0,:,:])):
            if depths[i] <= 1.0 and np.random.random() < beaching_prob:
                active_mask[i] = 0
        else:
            lons[i], lats[i] = next_lon, next_lat

        # --- VERTICAL TRANSPORT ---
        if not is_debris: 
            ws = static_ws if ws_mode == 0 else calculate_3d_ws(p_rho, r0_mm*2, bf_rho, biofilms[i], shape_idx)
            depths[i] += adv_z + (ws * dt) + (np.sqrt(2 * kz * abs_dt) * np.random.standard_normal())
            
            if depths[i] < 0: depths[i] = 0.0
            
            # Grounding logic for forward tracking
            if not is_backward:
                lb = interp(lons[i], lats[i], l_b, a_b, bathy_depth)
                if not np.isnan(lb) and depths[i] >= lb:
                    depths[i], active_mask[i] = lb, 0
        else: 
            depths[i] = 0.0
