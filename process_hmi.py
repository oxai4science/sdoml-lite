import argparse
import pprint
import sys
import datetime
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sunpy.coordinates import sun
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


# Mask to remove text printed lower left
# Example: http://jsoc.stanford.edu/data/hmi/images/2024/01/01/20240101_000000_M_1k.jpg
mask = np.ones((1024,1024))
mask[990:,:300] = 0.

def read_hmi_jpg(file_name):
    x = plt.imread(file_name)
    x = x.mean(axis=2)
    x *= mask
    x /= 255.
    return x

def process(args):
    source_file, target_file, resolution = args

    try:
        X = read_hmi_jpg(source_file)
        print('Source: {}'.format(source_file))
    except Exception as e:
        print('Error: {}'.format(e))
        return False
    
    fn = os.path.basename(source_file)
    # get datetime object from fn. Example fn: 20240101_000000_M_1k.jpg
    date = datetime.datetime.strptime(fn[:13], '%Y%m%d_%H%M')

    # Target angular size
    trgtAS = 976.0

    # Scale factor
    # rad = Xd.meta['RSUN_OBS']
    # Since we don't have the meta data, we'll use the sunpy library to get the angular radius of the sun (based on Earth's position instead of SDO's, but it should be close enough)
    rad = sun.angular_radius(date).to('arcsec').value

    scale_factor = trgtAS/rad

    #fix the translation
    t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
    #rescale and keep center
    XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
    Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))

    #figure out the integer factor to downsample by mean
    divideFactor = int(X.shape[0] / resolution)
    Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))

    #cast to fp32
    Xr = Xr.astype('float32')

    os.makedirs(os.path.dirname(target_file), exist_ok=True)    
    np.save(target_file, Xr)

    print('Target: {}'.format(target_file))
    return True


def main():
    description = 'FDL-X 2024, Radiation team, SDO HMI data processor'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--target_dir', type=str, help='Destination directory', required=True)
    parser.add_argument('--max_workers', type=int, default=1, help='Max workers')
    parser.add_argument('--worker_chunk_size', type=int, default=1, help='Chunk size per worker')
    parser.add_argument('--resolution', type=int, default=512, help='Pixel resolution of processed images. Should be a divisor of 1024.')
    parser.add_argument('--wavelengths', nargs='+', default=[94,131,171,193,211,304,335,1600,1700], help='Wavelengths')
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
    fits_files = glob(os.path.join(source_dir, '**', '*.jpg'), recursive=True)

    # create a list of tuples with the source and target file names
    # be careful to strip or add slashes as needed
    file_names = []
    for source_file in fits_files:
        target_file = source_file.replace(source_dir, target_dir).replace('.jpg', '.npy')
        target_file = target_file.replace('00_M_1k', '_M')
        target_file = target_file.replace(os.path.basename(target_file), 'HMI' + os.path.basename(target_file))
        
        file_names.append((source_file, target_file, args.resolution))

    # process the files
    results = process_map(process, file_names, max_workers=args.max_workers, chunksize=args.worker_chunk_size)

    print('Files processed: {}'.format(results.count(True)))
    print('Files failed   : {}'.format(results.count(False)))
    print('Files total    : {}'.format(len(results)))
    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()