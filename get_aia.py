import argparse
import pprint
import sys
import datetime
import time
import os
import urllib.request
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import traceback

# BioSentinel dates
# From: 2022-11-01T00:01:00 
# To:   2024-05-14T19:44:00


def date_to_filename(date, wavelength):
    # wavelength is an integer that can be 94, 131, 171, 193, 211, 304, 335, 1600, 1700
    # zero-pad wavelength to 4 digits
    return 'AIA{:%Y%m%d_%H%M}_{:04d}.fits'.format(date, wavelength)


def process(file_names):
    remote_file_name, local_file_name = file_names

    print('Remote: {}'.format(remote_file_name), flush=True)
    os.makedirs(os.path.dirname(local_file_name), exist_ok=True)
    timeout = 5 # seconds
    retries = 5
    for i in range(retries):
        if i > 0:
            print('Retrying ({}/{}): {}'.format(i+1, retries, remote_file_name))
            time.sleep(0.5)
        try:
            r = urllib.request.urlopen(remote_file_name, timeout=timeout)
            open(local_file_name, 'wb').write(r.read())
            print('Local : {}'.format(local_file_name))
            return True
        except Exception as e:
            print('Error: {}'.format(e))
            traceback.print_exception(*sys.exc_info()) 
            print()
    if os.path.exists(local_file_name):
        os.remove(local_file_name)
    return False


def main():
    description = 'FDL-X 2024, Radiation Team, SDO AIA data downloader'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--date_start', type=str, default='2010-05-13T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2024-07-27T00:00:00', help='End date')
    parser.add_argument('--cadence', type=int, default=15, help='Cadence (minutes)')
    # parser.add_argument('--wavelengths', nargs='+', default=[94,131,171,193,211,1600,1700], help='Wavelengths')
    parser.add_argument('--wavelengths', nargs='+', default=[131,171,193,211,1600], help='Wavelengths')
    parser.add_argument('--remote_root', type=str, default='http://jsoc.stanford.edu/data/aia/synoptic/', help='Remote root')
    parser.add_argument('--target_dir', type=str, help='Local root', required=True)
    parser.add_argument('--max_workers', type=int, default=1, help='Max workers')
    parser.add_argument('--worker_chunk_size', type=int, default=1, help='Chunk size per worker')
    parser.add_argument('--total_nodes', type=int, default=1, help='Total number of nodes')
    parser.add_argument('--node_index', type=int, default=0, help='Node index')
    
    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    date_start = datetime.datetime.fromisoformat(args.date_start)
    date_end = datetime.datetime.fromisoformat(args.date_end)
    
    if (args.cadence % 2 != 0) and (args.cadence != 15):
        print('Cadence must be an even number (except when it is 15).')
        return
    elif args.cadence == 15:
        print('Special case: Cadence is 15 minutes. Will use a sequence of minutes :00, :14, :30, :44.')
        # Adjust starting date to the nearest minute that is 0, 14, 30 or 44
        if date_start.minute < 14:
            date_start = date_start.replace(minute=0)
        elif date_start.minute < 30:
            date_start = date_start.replace(minute=14)
        elif date_start.minute < 44:
            date_start = date_start.replace(minute=30)
        else:
            date_start = date_start.replace(minute=44)
        print('Adjusted start date: {}'.format(date_start))
    else:
        # Adjust starting date to the nearest minute that is even
        if date_start.minute % 2 != 0:
            date_start = date_start.replace(minute=date_start.minute + 1)
            print('Adjusted start date: {}'.format(date_start))

    current = date_start

    file_names = []
    while current < date_end:
        # Sample URL, the last suffix is the wavelength
        # http://jsoc2.stanford.edu/data/aia/synoptic/2024/01/02/H0100/AIA20240102_0100_0094.fits

        for wavelength in args.wavelengths:
            file_name = date_to_filename(current, wavelength)
            remote_file_name = os.path.join(args.remote_root, '{:%Y/%m/%d/H%H00}'.format(current), file_name)
            # print('Remote: {}'.format(remote_file_name))
            local_file_name = os.path.join(args.target_dir, '{:%Y/%m/%d}'.format(current), file_name)
            # print('Local : {}'.format(local_file_name))
            file_names.append((remote_file_name, local_file_name))

        if args.cadence == 15:
            if current.minute == 0:
                current += datetime.timedelta(minutes=14)
            elif current.minute == 14:
                current += datetime.timedelta(minutes=16)
            elif current.minute == 30:
                current += datetime.timedelta(minutes=14)
            elif current.minute == 44:
                current += datetime.timedelta(minutes=16)
        else:
           current += datetime.timedelta(minutes=args.cadence)


    if len(file_names) == 0:
        print('No files to download.')
        return
    
    if len(file_names) < args.total_nodes:
        print('Total number of files is less than the total number of nodes.')
        return

    files_per_node = len(file_names) // args.total_nodes
    # get the subset of file names for this node, based on the total number of nodes and the node index
    file_names_for_this_node = file_names[args.node_index * files_per_node : (args.node_index + 1) * files_per_node]
    
    print('Total nodes: {}'.format(args.total_nodes))
    print('Node index : {}'.format(args.node_index))
    print('Total files for all nodes : {}'.format(len(file_names)))
    print('Total files for this node : {}'.format(len(file_names_for_this_node)))
    
    if args.max_workers == 1:
        results = list(map(process, file_names_for_this_node))
    else:
        results = process_map(process, file_names_for_this_node, max_workers=args.max_workers, chunksize=args.worker_chunk_size)

    print('Files downloaded: {}'.format(results.count(True)))
    print('Files skipped   : {}'.format(results.count(False)))
    print('Files total     : {}'.format(len(results)))
    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()