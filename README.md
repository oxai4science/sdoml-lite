# SDOML-lite

SDOML-lite is a lightweight version of the SDOML dataset specifically designed for machine learning applications in solar physics, providing continuous full-disk images of the Sun with magnetic field and extreme ultraviolet data in several wavelength. The data source is the Solar Dynamics Observatory (SDO) space telescope, a NASA mission that has been in operation since 2010.

This repository contains:

- Self-contained code that can be used to create custom SDOML-lite datasets with any given date range, downloading data from original sources and processing into the SDOML-lite format. This is based entirely in Python.
- A PyTorch dataset implementation to work with the data.

IMPORTANT: SDOML and SDOML-lite datasets are different in structure and data distributions. SDOML-lite is inspired by SDOML, but there is no compatibility between the two formats.

## Data

The data is derived from the Helioseismic and Magnetic Imager (HMI) and the Atmospheric Imaging Assembly (AIA) instruments onboard SDO.

For each date represented, we provide ten channels of images containing the following data:
- One channel containing magnetic field data. A line-of-sight magnetogram based on the HMI data product "15-Minute Image Catalog" derived from data series hmi.M_720s.
- Nine channels containing AIA wavelengths: 94, 131, 171, 193, 211, 304, 335, 1600, and 1700 Å. 

By default the data is provided with an image resolution of 512x512 pixels and a time resolution of 15 minutes. The time resolution is limited by the underlying HMI data product that we use, which has a 15-minute cadence. The 15-minute HMI data is paired with the nearest available AIA data, which has a 12-second cadence.

## Creating your own custom SDOML-lite dataset

TO DO

```
python get_aia.py --target_dir ./dataset_raw
python get_hmi.py --target_dir ./dataset_raw

python process_aia.py --source_dir ./dataset_raw --target_dir ./dataset
python process_hmi.py --source_dir ./dataset_raw --target_dir ./dataset
```

Multi-node download usage example:
```
python get_aia.py --target_dir ./dataset_raw --max_workers 10 --worker_chunk_size 10 --total_nodes 10 --node_index 0
```

## Data sources

The data is downloaded from the Stanford / Joint Science Operations Center (JSOC). 

TO DO

## Information about the original SDOML format

Galvez, R., Fouhey, D.F., Jin, M., Szenicer, A., Muñoz-Jaramillo, A., Cheung, M.C., Wright, P.J., Bobra, M.G., Liu, Y., Mason, J. and Thomas, R., 2019. A machine-learning data set prepared from the NASA Solar Dynamics Observatory mission. The Astrophysical Journal Supplement Series, 242(1), p.7. https://iopscience.iop.org/article/10.3847/1538-4365/ab1005