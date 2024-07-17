# SDOML-lite

SDOML-lite is a lightweight version of the SDOML dataset specifically designed for machine learning applications in solar physics, providing continuous full-disk images of the Sun with magnetic field and extreme ultraviolet data in several wavelengths. The data source is the [Solar Dynamics Observatory (SDO)](https://sdo.gsfc.nasa.gov/) space telescope, a NASA mission that has been in operation since 2010.

This repository contains:

- Self-contained code that can be used to create custom SDOML-lite datasets with any given date range, downloading data from original sources and processing into the SDOML-lite format. This is implemented entirely in Python.
- A PyTorch dataset implementation to work with the data.

*IMPORTANT: SDOML and SDOML-lite datasets are different in structure and data distributions. SDOML-lite is inspired by SDOML, but there is no compatibility between the two formats.*

## Creating your own custom SDOML-lite dataset

The following steps will download AIA and HMI data from the SDO mission, process it, and create a custom SDOML-lite dataset with the default date range and other default settings. The scripts will download data from the [Joint Science Operations Center (JSOC)](http://jsoc.stanford.edu/).

```
python get_aia.py --target_dir ./dataset_raw
python get_hmi.py --target_dir ./dataset_raw

python process_aia.py --source_dir ./dataset_raw --target_dir ./dataset
python process_hmi.py --source_dir ./dataset_raw --target_dir ./dataset

python make_dataset.py --source_dir ./dataset --target_dir ./sdoml-lite
```

### Custom date range

By default, the scripts download data for the date range from `2022-11-01T00:01:00` to `2024-05-14T19:44:00`. These are the dates for a subset of the NASA [BioSentinel](https://www.nasa.gov/centers-and-facilities/ames/what-is-biosentinel/) mission data, measuring space radiation. Working with BioSentinel data was the main motivation for the creation of the SDOML-lite dataset.

 You can specify a custom date range by using the `--date_start` and `--date_end` arguments. Example usage:
```
python get_aia.py --target_dir ./dataset_raw --date_start 2019-11-01T00:00:00 --date_end 2021-05-01T00:00:00
python get_hmi.py --target_dir ./dataset_raw --date_start 2019-11-01T00:00:00 --date_end 2021-05-01T00:00:00
```

### Optional: Parallel processing on a single compute node

The `get_aia.py`, `get_hmi.py`, `process_aia.py`, `process_hmi.py` scripts support parallel processing using multiple processes on the same compute node. Example usage:
```
python get_aia.py --target_dir ./dataset_raw --max_workers 8 --worker_chunk_size 10
```


### Optional: Chunking of the download across multiple compute nodes
The `get_aia.py` and `get_hmi.py` scripts support chunking of the files to be downloaded across multiple compute nodes. This allows the download phase to be completed faster by downloading separate chunks of data on different nodes. Example usage:
```
python get_aia.py --target_dir ./dataset_raw --total_nodes 10 --node_index 0
python get_aia.py --target_dir ./dataset_raw --total_nodes 10 --node_index 1
...
python get_aia.py --target_dir ./dataset_raw --total_nodes 10 --node_index 9
```

You then need to combine the downloaded chunks into one unified dataset directory. For example:
```
scp -r user@remote_host_0:/path/to/dataset_raw ./dataset_raw
scp -r user@remote_host_1:/path/to/dataset_raw ./dataset_raw
...
scp -r user@remote_host_9:/path/to/dataset_raw ./dataset_raw
```

*IMPORTANT: When using chunked data downloads across multiple compute nodes, it is crucial to apply the data processing step with `process_aia.py` only after all the downloaded data are unified into a single directory. This is because the `process_aia.py` script has a data normalization phase that depends on the data distribution the script reads from the files it processes. If `process_aia.py` file is applied to different data chunks separately, the normalization used between different chunks will be different and it will be invalid to combine the processed data chunks into a final dataset.*

## Data

The data is derived from the Helioseismic and Magnetic Imager (HMI) and the Atmospheric Imaging Assembly (AIA) instruments onboard SDO. Our scripts download the data from Stanford [Joint Science Operations Center (JSOC)](http://jsoc.stanford.edu/).

The dataset contains ten image channels for each date:
- One channel containing a line-of-sight magnetogram. Based on the HMI "15-Minute Image Catalog" data in JPG format, 1024x1024 resolution, 15-minute cadence, derived from hmi.M_720s data series: http://jsoc.stanford.edu/data/hmi/images/
- Seven channels containing AIA wavelengths 94, 131, 171, 193, 211, 1600, and 1700 Å. Based on the AIA Synoptic data in FITS format, 1024x1024 resolution, 2-minute cadence: http://jsoc2.stanford.edu/data/aia/synoptic/

For the AIA channels, we exclude the wavelengths 304, 335 Å from the default configuration, due to degradation-related issues, but these can be added to dataset creation with the argument `--wavelengths` of the `get_aia.py` and `process_aia.py` scripts. We also exclude channel 4500 Å.

By default the data is provided with an image resolution of 512x512 pixels and a time resolution of 15 minutes. The scripts allow generation of SDOML-lite datasets with 1024x1024 resolution if needed.

The HMI data source we use has 15-minute cadence and the AIA data source we use has 2-minute cadence. Given the limiting nature of the 15-minute HMI data that we use, we pair these two datasets using the (HMI, AIA) pairs with times (HH:00, HH:00), (HH:15, HH:14), (HH:30, HH:30), (HH:45, HH:44) for any given hour HH. This is done to ensure that the AIA data is as close as possible to the HMI data in time.

### Format

SDOML-lite uses the WebDataset convention for storing data. With the default settings, it has one day of data per tar file (representing a shard of the whole dataset), so for a date range of one year, there would be 365 tar files named `sdoml-lite-001.tar` to `sdoml-lite-365.tar`. These tar files can be used shard-based shuffling and distributed training.

Within each tar file, the data is stored using file names `YYYY/MM/DD/HHMM.AIA_WWWW.npy` for AIA and `YYYY/MM/DD/HHMM.HMI_M.npy` for HMI, where `YYYY` is the year, `MM` is the month, `DD` is the day, `HH` is the hour, `MM` is the minute, and `WWWW` is the AIA wavelength string. For example:
```
...
2024/03/08/2345.AIA_0094.npy
2024/03/08/2345.AIA_0131.npy
2024/03/08/2345.AIA_0171.npy
2024/03/08/2345.AIA_0193.npy
2024/03/08/2345.AIA_0211.npy
2024/03/08/2345.AIA_1600.npy
2024/03/08/2345.AIA_1700.npy
2024/03/08/2345.HMI_M.npy
...
```

WebDataset format treats each `YYYY/MM/DD/HHMM` name (until the first period in the file name) as a unit of data sample, and different channels for that data sample will be accessible by names `AIA_0094.npy`, `AIA_0131.npy`, ..., `HMI_M.npy` etc.

Reference material for information on the WebDataset format and its benefits for machine learning applications:
- https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/
- https://huggingface.co/docs/hub/en/datasets-webdataset
- Aizman, A., Maltby, G. and Breuel, T., 2019, December. High performance I/O for large scale deep learning. In 2019 IEEE International Conference on Big Data (Big Data) (pp. 5965-5967). IEEE. https://arxiv.org/abs/2001.01858

### Size on disk

With the default settings (512x512 resolution, 15-minute cadence), the data size is approximately 738 MiB per day, of which 96 MiB (13%) is HMI data and 642 MiB (87%) is AIA data. The dataset size can be reduced by using a lower resolution (e.g., 256x256) or by using a lower cadence (e.g., 30-minute cadence). Some applications can also work with HMI-only or AIA-only datasets.

### Data normalization

The data comes normalized within each image channel such that the pixel values are in the range [0, 1], making it ready for machine learning use out of the box. 

The HMI source we use is already normalized in the range [0, 1]. We normalize the AIA data based on the statistics of the actual AIA data processed during the generation of the dataset, in a two-phase processing pipeline where the first phase computes data statistics and the second phase applies normalization.

### A note on data quality

The main motivation for SDOML-lite is to provide a lightweight dataset to be consumed as an input to machine learning pipelines, e.g., models that can predict Sun-dependent quantities in space weather, thermospheric density, or radiation domains. 

We believe that the data is of sufficient quality as an input for machine learning applications, but note that it is not intended to conduct scientific analyses of the HMI or AIA instruments.

## Information about the original SDOML format

Galvez, R., Fouhey, D.F., Jin, M., Szenicer, A., Muñoz-Jaramillo, A., Cheung, M.C., Wright, P.J., Bobra, M.G., Liu, Y., Mason, J. and Thomas, R., 2019. A machine-learning data set prepared from the NASA Solar Dynamics Observatory mission. The Astrophysical Journal Supplement Series, 242(1), p.7. https://iopscience.iop.org/article/10.3847/1538-4365/ab1005
