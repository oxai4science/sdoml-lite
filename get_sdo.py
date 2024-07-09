import argparse
import pprint
import sys
import datetime
import os
import urllib.request
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# BioSentinel dates
# From: 2022-11-01T00:01:00 
# To:   2024-05-14T19:44:00

# SDO AIA dates
# From: 2022-11-01T00:02:00 
# To:   2024-05-14T19:44:00

def date_to_filename(date, wavelength):
    # wavelength is an integer that can be 94, 131, 171, 193, 211, 304, 335, 1600, 1700
    # zero-pad wavelength to 4 digits
    return 'AIA{:%Y%m%d_%H%M}_{:04d}.fits'.format(date, wavelength)


def process(file_names):
    remote_file_name, local_file_name = file_names

    print('Remote: {}'.format(remote_file_name))
    os.makedirs(os.path.dirname(local_file_name), exist_ok=True)
    try:
        urllib.request.urlretrieve(remote_file_name, local_file_name)
        print('Local : {}'.format(local_file_name))
        return True
    except Exception as e:
        print('Error: {}'.format(e))
        return False


def main():
    description = 'FDL-X 2024, Radiation team, SDO AIA data downloader and processor.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--date_start', type=str, default='2022-11-01T00:02:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2022-11-01T01:01:00', help='End date')
    parser.add_argument('--cadence', type=int, default=12*60, help='Cadence (seconds)')
    parser.add_argument('--wavelengths', nargs='+', default=[94,131,171,193,211,304,335,1600,1700], help='Wavelengths')
    parser.add_argument('--remote_root', type=str, default='http://jsoc.stanford.edu/data/aia/synoptic/', help='Remote root')
    parser.add_argument('--local_root', type=str, help='Local root', required=True)
    parser.add_argument('--max_workers', type=int, default=4, help='Max workers')
    parser.add_argument('--chunk_size', type=int, default=1, help='Chunk size')
    
    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    date_start = datetime.datetime.fromisoformat(args.date_start)
    date_end = datetime.datetime.fromisoformat(args.date_end)
    current = date_start
    
    file_names = []
    while current < date_end:
        # Sample pattern, the last suffix is the wavelength
        # http://jsoc2.stanford.edu/data/aia/synoptic/2024/01/02/H0100/AIA20240102_0100_0094.fits

        for wavelength in args.wavelengths:
            file_name = date_to_filename(current, wavelength)
            remote_file_name = os.path.join(args.remote_root, '{:%Y/%m/%d/H%H00}/'.format(current), file_name)
            # print('Remote: {}'.format(remote_file_name))
            local_file_name = os.path.join(args.local_root, '{:%Y/%m/%d}/'.format(current), file_name)
            # print('Local : {}'.format(local_file_name))
            file_names.append((remote_file_name, local_file_name))

        current += datetime.timedelta(seconds=args.cadence)

    print('Total files: {}'.format(len(file_names)))


    # for remote_file_name, local_file_name in tqdm(file_names):
        # process((remote_file_name, local_file_name))


    results = process_map(process, file_names, max_workers=args.max_workers, chunksize=args.chunk_size)

    print('Files downloaded: {}'.format(results.count(True)))
    print('Files skipped   : {}'.format(results.count(False)))
    print('Files total     : {}'.format(len(results)))
    print('End time: {}'.format(datetime.datetime.now()))
    print('Elapsed time: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()