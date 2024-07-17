import argparse
import os
import datetime
import pprint
import sys
import tarfile
from glob import glob


def main():
    description = 'WebDataset creation script'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--target_dir', type=str, help='Destination directory', required=True)
    parser.add_argument('--files_per_archive', type=int, default=None, help='Number of files per archive')
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

    files_per_archive = args.files_per_archive
    if files_per_archive is None:
        files_per_archive = 1 + len(files) // 2
        print(f"Number of files per archive not specified, defaulting to {files_per_archive}")
    if files_per_archive < 1:
        print("Number of files per archive must be greater than 0.")
        sys.exit(1)
    elif files_per_archive > len(files):
        print("Number of files per archive must be less than the total number of files.")
        sys.exit(1)

    # Calculate how many digits needed to zero-pad file names
    total_files = len(files)
    num_archives = (total_files + files_per_archive - 1) // files_per_archive
    padding_length = len(str(num_archives))

    # Size of all data under source directory
    total_size = sum(os.path.getsize(file) for file in files)
    archive_size = total_size // num_archives

    print(f"Total number of files found       : {total_files:,}")
    print(f"Number of files per archive (max) : {files_per_archive:,}")
    print(f"Number of archives to be generated: {num_archives:,}")
    print(f"Total size of data                : {total_size:,} bytes")
    print(f"Approx. size of data per archive  : {archive_size:,} bytes")

    # Loop to create tar files
    for i in range(0, total_files, files_per_archive):
        # Calculate the index for naming the tar file
        index = (i // files_per_archive) + 1

        # Format the index with zero padding
        formatted_index = str(index).zfill(padding_length)

        tar_filename = f"{prefix}-{formatted_index}.tar"
        tar_filepath = os.path.join(target_dir, tar_filename)

        print(f"Generating file         : {tar_filename}")

        # Create a tar file for this chunk
        with tarfile.open(tar_filepath, "w") as tar:
            for file in files[i:i+files_per_archive]:
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

        print(f"Finished generating file: {tar_filename} ({index}/{num_archives}) ({os.path.getsize(tar_filepath):,} bytes)")

    print("Archives created successfully.")

    end_time = datetime.datetime.now()
    print('End time: {}'.format(end_time))
    print('Duration: {}'.format(end_time - start_time))


if __name__ == "__main__":
    main()
