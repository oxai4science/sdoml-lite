# SDOML-lite

SDOML-lite is a lightweight version of the SDOML dataset specifically designed for machine learning applications in solar physics, containing continuous full-disk images of the Sun across several extreme ultraviolet channels and magnetograms. The data source is the space-based telescope Solar Dynamics Observatory (SDO), a NASA mission that has been in operation since 2010.

IMPORTANT: SDOML and SDOML-lite datasets are different in structure and data distributions. SDOML-lite is inspired by SDOML, but there is no compatibility between the two formats.

## Data

TO DO

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