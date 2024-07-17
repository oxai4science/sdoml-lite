import argparse
import os
import datetime
import pprint
import sys
import tarfile
from glob import glob


def main():
    description = 'SDOML-lite WebDataset creation script'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--target_dir', type=str, help='Destination directory', required=True)
    parser.add_argument('--days_per_archive', type=int, default=1, help='Number of days per archive')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix for tar files')

    args = parser.parse_args()
    
    print(description)

    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    source_dir = os.path.abspath(args.source_dir)
    target_dir = os.path.abspath(args.target_dir)

    # Ensure source directory exists
    if not os.path.isdir(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        sys.exit
    
    # Ensure target directory exists or create it
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    # the last directory in the target path is the prefix
    prefix = os.path.basename(target_dir)

    # Find all files in the directory
    files = sorted([file for file in glob(os.path.join(source_dir, '**', '*'), recursive=True) if os.path.isfile(file)])

    print()
    # Size of all data under source directory
    total_size = sum(os.path.getsize(file) for file in files)
    print(f"Total number of files found: {len(files):,}")
    print(f"Total size of data         : {total_size:,} bytes")

    file_earliest = os.path.relpath(files[0], source_dir)
    file_latest = os.path.relpath(files[-1], source_dir)
    print(f"Earliest file: {file_earliest}")
    print(f"Latest file  : {file_latest}")

    date_earliest = datetime.datetime.strptime(file_earliest[:10], '%Y/%m/%d')
    date_latest = datetime.datetime.strptime(file_latest[:10], '%Y/%m/%d')

    #print earliest date without time information
    print(f"Earliest date: {date_earliest.strftime('%Y-%m-%d')}")
    print(f"Latest date  : {date_latest.strftime('%Y-%m-%d')}")
    print(f'Total days   : {(date_latest - date_earliest).days + 1:,}')

    # iterate over days

    num_archives = ((date_latest - date_earliest).days + args.days_per_archive)// args.days_per_archive
    print(f"Number of archives to be generated: {num_archives:,}")

    padding_length = len(str(num_archives))

    current = date_earliest
    while current <= date_latest:
        current_end = current + datetime.timedelta(days=args.days_per_archive - 1)
        if current_end > date_latest:
            current_end = date_latest

        # calculate the index for naming the tar file
        index = (current - date_earliest).days // args.days_per_archive + 1
        # format the index with zero padding
        formatted_index = str(index).zfill(padding_length)
        tar_filename = f"{prefix}-{formatted_index}.tar"
        tar_filepath = os.path.join(target_dir, tar_filename)

        print()
        print(f"Archive               : {tar_filename} ({index}/{num_archives})")
        print(f"Date range (inclusive): {current.strftime('%Y-%m-%d')} - {current_end.strftime('%Y-%m-%d')}")

        with tarfile.open(tar_filepath, "w") as tar:
            while current <= current_end:
                print(f"Adding date: {current.strftime('%Y-%m-%d')}")

                files_to_add = sorted([file for file in files if file.startswith(os.path.join(source_dir, current.strftime('%Y/%m/%d')))])
                if len(files_to_add) == 0:
                    print(f"No files found for {current.strftime('%Y-%m-%d')}")
                else:
                    for file in files_to_add:
                        arcname = os.path.relpath(file, source_dir)
                        # print('Source: {}'.format(arcname))
                        arcname_dir = os.path.dirname(arcname)
                        arcname_base = os.path.basename(arcname)

                        if arcname_base.startswith('AIA'):
                            _, time, wavelength = arcname_base.split('_')

                            if time.endswith('14'):
                                time = time[:2] + '15'
                            elif time.endswith('44'):
                                time = time[:2] + '45'

                            wavelength = wavelength.split('.')[0]
                            arcname_base = f"{time}.AIA_{wavelength}.npy"
                        elif arcname_base.startswith('HMI'):
                            _, time, _ = arcname_base.split('_')
                            arcname_base = f"{time}.HMI_M.npy"
                        else:
                            print(f"Unknown file format: {arcname_base}")
                            continue

                        arcname = os.path.join(arcname_dir, arcname_base)
                        # print('Target: {}'.format(arcname))

                        tar.add(file, arcname=arcname)
                current += datetime.timedelta(days=1)

            print(f"Archive complete      : {tar_filename} ({os.path.getsize(tar_filepath):,} bytes)")

    print("Archives created successfully.")

    end_time = datetime.datetime.now()
    print('End time: {}'.format(end_time))
    print('Duration: {}'.format(end_time - start_time))


if __name__ == "__main__":
    main()
