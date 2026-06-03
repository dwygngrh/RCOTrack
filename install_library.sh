#!/bin/bash

# Create the conda environment with Python 3.10
conda create -n lagrangian python=3.10 -y

# Activate the environment
source activate lagrangian

# Install core data processing and oceanographic libraries
# xarray and netcdf4 are required for CMEMS and ERA5 data handling
# pandas handles the temporal indexing for daily/hourly data[cite: 1]
conda install -c conda-forge xarray netcdf4 pandas dask -y

# Install high-performance computing libraries
# Numba is required for the parallelized JIT-compiled kernels
# Numpy handles the heavy coordinate array math[cite: 1]
conda install -c conda-forge numpy numba -y

# Install additional utilities
conda install -c conda-forge configparser -y

echo "------------------------------------------------"
echo "Environment 'lagrangian' is ready."
echo "Activate it using: conda activate lagrangian"
echo "------------------------------------------------"
