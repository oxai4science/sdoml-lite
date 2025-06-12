import argparse
import pprint
import sys
import datetime
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from glob import glob

from util import hmi_process


def main():
    description = 'SDOML-lite, SDO HMI data processor'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--target_dir', type=str, help='Destination directory', required=True)
    parser.add_argument('--max_workers', type=int, default=1, help='Max workers')
    parser.add_argument('--worker_chunk_size', type=int, default=1, help='Chunk size per worker')
    parser.add_argument('--resolution', type=int, default=512, help='Pixel resolution of processed images. Should be a divisor of 1024.')
    parser.add_argument('--degradation_dir', type=str, default='./degradation/v9', help='Directory with degradation correction files')

    args = parser.parse_args()

    print(description)

    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    # walk through the source directory with glob, find all .jpg files, and create a corresponding file name ending in .npy in the target dir, keeping the directory structure

    # set the source and target directories, strip final slash if present
    source_dir = args.source_dir.rstrip('/')
    target_dir = args.target_dir.rstrip('/')

    # get all .fits files in the source directory
    jpg_files = glob(os.path.join(source_dir, '**', '*.jpg'), recursive=True)

    if len(jpg_files) == 0:
        print('No files found in source directory: {}'.format(source_dir))
        return
    
    # create a list of tuples with the source and target file names
    # be careful to strip or add slashes as needed
    file_names = []
    for source_file in jpg_files:
        target_file = source_file.replace(source_dir, target_dir).replace('.jpg', '.npy')
        target_file = target_file.replace('00_M_1k', '_M')
        target_file = target_file.replace(os.path.basename(target_file), 'HMI' + os.path.basename(target_file))
        
        file_names.append((source_file, target_file, args.resolution))

    # process the files
    results = process_map(hmi_process, file_names, max_workers=args.max_workers, chunksize=args.worker_chunk_size)

    print('Files processed: {}'.format(results.count(True)))
    print('Files failed   : {}'.format(results.count(False)))
    print('Files total    : {}'.format(len(results)))
    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()