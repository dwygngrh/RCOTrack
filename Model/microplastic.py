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
from numba import njit

@njit
def get_polymer_density(polymer_type):
    """Specific density mapping"""
    if polymer_type == "PE": return 910.0 
    if polymer_type == "HDPE": return 950.0
    return 1025.0

@njit
def get_fouled_density(rho_p, r0_mm, rho_bf, bt_mm):
    """Calculates fouled density based on core-shell volume ratios."""
    r0 = r0_mm / 1000.0
    bt = bt_mm / 1000.0
    vol_ratio = (r0**3) / ((r0 + bt)**3)
    return (rho_p * vol_ratio) + (rho_bf * (1.0 - vol_ratio))

@njit
def calculate_3d_ws(rho_p, d_mm, rho_bf, bt_mm, shape_idx):
    """Calculates dynamic ws based on Jalón-Rojas et al. (2019)."""
    g, rho_w, nu = 9.81, 1025.0, 1e-6
    radius_tot_m = (d_mm / 2.0 + bt_mm) / 1000.0
    rho_fouled = get_fouled_density(rho_p, d_mm/2.0, rho_bf, bt_mm)
    
    # Buoyancy: positive for sinking, negative for rising
    d_star = 2 * radius_tot_m * (g * np.abs(rho_fouled - rho_w) / (rho_w * nu**2))**(1/3)
    
    if shape_idx == 0: # sphere
        w_val = (nu / (2 * radius_tot_m)) * d_star**3 * (38.1 + 0.93 * d_star**(12/7))**-0.875
    else: # cylinder/other
        w_val = (np.pi / 2) * (1/nu) * g * (np.abs(rho_fouled - rho_w)/rho_w) * (2*(radius_tot_m**2) / 55.0)
    
    return w_val if rho_fouled > rho_w else -w_val
