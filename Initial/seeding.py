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

def spawn(src, now, dt_seconds):
    """
    Spawns particles using a probabilistic threshold to handle low release rates.
    Ensures consistent total particle counts across different time-step (dt) configurations.
    """
    # Only release if current time is within the release window
    if not (src['start_rel'] <= now <= src['end_rel']):
        return [], [], []
    
    # Calculate the exact floating-point number of particles for this specific step
    # Scaling the daily rate to the current simulation time step
    n_float = (src['rate_day'] / 86400.0) * abs(dt_seconds)
    
    # 1. Get the base integer (number of guaranteed particles)
    n = int(n_float)
    
    # 2. Use the fractional part as a probability for one extra particle
    # This prevents 'lost' particles when the rate per step is less than 1
    if np.random.rand() < (n_float - n):
        n += 1
    
    if n <= 0: 
        return [], [], []
    
    # Random spatial distribution within the specified diameter
    # Convert diameter in meters to approximate degrees
    r_deg = (src['diameter'] / 2.0) / 111000.0
    
    # Uniform area distribution: sqrt(rand) ensures particles don't cluster at the center
    r = r_deg * np.sqrt(np.random.rand(n))
    theta = np.random.rand(n) * 2 * np.pi
    
    # Calculate final coordinates relative to the source point
    lons = (src['lon'] + r * np.cos(theta)).tolist()
    lats = (src['lat'] + r * np.sin(theta)).tolist()
    depths = [src['depth']] * n
    
    return lons, lats, depths
