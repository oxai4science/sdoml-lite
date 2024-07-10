Experimental script to download SDO AIA Synoptic data files (.fits format) from Stanford/JSOC.

Multi-node download usage example:
```
python3 get_sdo.py --local_root ./radiation-sdo --max_workers 10 --worker_chunk_size 10 --total_nodes 10 --node_index 0
```
