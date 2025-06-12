import argparse
import pprint
import sys
import datetime
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np
from glob import glob

from util import aia_load_degradations, aia_process, aia_normalize


def main():
    description = 'SDOML-lite, SDO AIA data processor'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--target_dir', type=str, help='Destination directory', required=True)
    parser.add_argument('--max_workers', type=int, default=1, help='Max workers')
    parser.add_argument('--worker_chunk_size', type=int, default=1, help='Chunk size per worker')
    parser.add_argument('--resolution', type=int, default=512, help='Pixel resolution of processed images. Should be a divisor of 1024.')
    # parser.add_argument('--wavelengths', nargs='+', default=[94,131,171,193,211,1600,1700], help='Wavelengths')
    parser.add_argument('--wavelengths', nargs='+', default=[131,171,193,211,1600], help='Wavelengths')
    parser.add_argument('--degradation_dir', type=str, default='./degradation/v9', help='Directory with degradation correction files')

    args = parser.parse_args()

    print(description)

    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    print('**************************')
    print('** Phase 1: Postprocessing')
    print('**************************')

    print('Loading degradations')
    degradations = aia_load_degradations(args.degradation_dir, args.wavelengths)

    # walk through the source directory with glob, find all .fits files, and create a corresponding file name ending in .npy in the target dir, keeping the directory structure

    # set the source and target directories, strip final slash if present
    source_dir = args.source_dir.rstrip('/')
    target_dir = args.target_dir.rstrip('/')

    # get all .fits files in the source directory
    fits_files = glob(os.path.join(source_dir, '**', '*.fits'), recursive=True)

    if len(fits_files) == 0:
        print('No files found in source directory: {}'.format(source_dir))
        return

    # create a list of tuples with the source and target file names
    # be careful to strip or add slashes as needed
    file_names = []
    for source_file in fits_files:
        target_file = source_file.replace(source_dir, target_dir).replace('.fits', '_unnormalized.npy')
        file_names.append((source_file, target_file, args.resolution, degradations))

    # process the files
    results = process_map(aia_process, file_names, max_workers=args.max_workers, chunksize=args.worker_chunk_size)

    files_failed = results.count(False)
    print('Files processed: {}'.format(len(results) - files_failed))
    print('Files failed   : {}'.format(files_failed))
    print('Files total    : {}'.format(len(results)))

    print('*************************')
    print('** Phase 2: Normalization')
    print('*************************')
    # construct dictionary of wavelenghts, min values in a numpy array.
    min_values = {}
    max_values = {}
    wavelenghts = []
    for result in results:
        if result == False:
            continue
        wavelength, min_value, max_value = result
        wavelenghts.append(wavelength)
        if wavelength not in min_values:
            min_values[wavelength] = []
            max_values[wavelength] = []
        min_values[wavelength].append(min_value)
        max_values[wavelength].append(max_value)

    for wavelength in wavelenghts:
        min_values[wavelength] = np.array(min_values[wavelength]).min()
        max_values[wavelength] = np.array(max_values[wavelength]).max()

    file_names_normalize = []
    for source_file, target_file, args.resolution, degradations in file_names:
        file_names_normalize.append((target_file, max_values))
   
    results = process_map(aia_normalize, file_names_normalize, max_workers=args.max_workers, chunksize=args.worker_chunk_size)

    print('Normalization factors')
    print('Min values:')
    pprint.pprint(min_values)
    print('Max values:')
    pprint.pprint(max_values)

    files_failed = results.count(False)
    print('Files processed: {}'.format(len(results) - files_failed))
    print('Files failed   : {}'.format(files_failed))
    print('Files total    : {}'.format(len(results)))

    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()