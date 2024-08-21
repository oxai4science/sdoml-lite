import os
import sys
import argparse
import datetime
import pprint


def main():
    description = 'SDOML-lite, file inspector'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--min_size', type=int, default=400000, help='Threshold for minimum file size (bytes)')

    args = parser.parse_args()

    print(description)

    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    print()
    files_processed = 0
    files_reported = 0
    for root, _, filenames in os.walk(args.source_dir):
        for filename in filenames:
            files_processed += 1
            file_path = os.path.join(root, filename)
            size = os.path.getsize(file_path)
            if size < args.min_size:
                files_reported += 1
                print('File: {} Size: {}'.format(file_path, size))

    print()
    print('Files processed: {}'.format(files_processed))
    print('Files reported : {}'.format(files_reported))

    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == '__main__':
    main()