Experimental script to download SDO AIA Synoptic data files (.fits format) from Stanford/JSOC.


Create a custom SDOML-lite dataset:

```
python get_aia.py --local_root ./dataset_raw
python get_hmi.py --local_root ./dataset_raw
```

Multi-node download usage example:
```
python3 get_sdo.py --local_root ./radiation-sdo --max_workers 10 --worker_chunk_size 10 --total_nodes 10 --node_index 0
```
