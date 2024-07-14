# SDOML-lite

SDOML-lite is a lightweight version of the SDOML dataset specifically designed for machine learning applications in solar physics, providing continuous full-disk images of the Sun with magnetic field and extreme ultraviolet data in several wavelengths. The data source is the [Solar Dynamics Observatory (SDO)](https://sdo.gsfc.nasa.gov/) space telescope, a NASA mission that has been in operation since 2010.

This repository contains:

- Self-contained code that can be used to create custom SDOML-lite datasets with any given date range, downloading data from original sources and processing into the SDOML-lite format. This is based entirely in Python.
- A PyTorch dataset implementation to work with the data.

*IMPORTANT: SDOML and SDOML-lite datasets are different in structure and data distributions. SDOML-lite is inspired by SDOML, but there is no compatibility between the two formats.*

## Data

The data is derived from the Helioseismic and Magnetic Imager (HMI) and the Atmospheric Imaging Assembly (AIA) instruments onboard SDO. Our scripts download the data from Stanford [Joint Science Operations Center (JSOC)](http://jsoc.stanford.edu/).

The dataset contains ten image channels for each date:
- One channel containing a line-of-sight magnetogram. Based on the HMI "15-Minute Image Catalog" data in JPG format, 1024x1024 resolution, 15-minute cadence, derived from hmi.M_720s data series: http://jsoc.stanford.edu/data/hmi/images/
- Nine channels containing AIA wavelengths 94, 131, 171, 193, 211, 304, 335, 1600, and 1700 Å. Based on the AIA Synoptic data in FITS format, 1024x1024 resolution, 2-minute cadence: http://jsoc2.stanford.edu/data/aia/synoptic/

By default the data is provided with an image resolution of 512x512 pixels and a time resolution of 15 minutes. The scripts allow generation of SDOML-lite datasets with 1024x1024 resolution if needed.

The HMI data source we use has 15-minute cadence and the AIA data source we use has 2-minute cadence. Given the limiting nature of the 15-minute HMI data that we use, we pair these two datasets using the (HMI, AIA) pairs with times (HH:00, HH:00), (HH:15, HH:14), (HH:30, HH:30), (HH:45, HH:44) for any given hour HH. This is done to ensure that the AIA data is as close as possible to the HMI data in time.

### A note on data quality

The main motivation for SDOML-lite is to provide a lightweight dataset to be consumed as an input to machine learning pipelines, e.g., models that can predict Sun-dependent quantities in space weather, thermospheric density, or radiation domains. 

We believe that the data is of sufficient quality as an input for machine learning applications, but note that it is not intended to be used for scientific analysis of the HMI or AIA instruments.

### Data normalization

The data comes normalized within each image channel such that the pixel values are in the range [0, 1], making it ready for machine learning use out of the box. 

The HMI source we use is already normalized in the range [0, 1]. We normalize the AIA data based on the statistics of the actual AIA data processed during the generation of the dataset, in a two-phase processing pipeline where the first phase computes data statistics and the second phase applies normalization.

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

## Information about the original SDOML format

Galvez, R., Fouhey, D.F., Jin, M., Szenicer, A., Muñoz-Jaramillo, A., Cheung, M.C., Wright, P.J., Bobra, M.G., Liu, Y., Mason, J. and Thomas, R., 2019. A machine-learning data set prepared from the NASA Solar Dynamics Observatory mission. The Astrophysical Journal Supplement Series, 242(1), p.7. https://iopscience.iop.org/article/10.3847/1538-4365/ab1005