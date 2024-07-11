import argparse
import pprint
import sys
import datetime
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sunpy.map import Map
import skimage.transform
import numpy as np
from glob import glob

def process(args):
    source_file, target_file, resolution = args

    try:
        Xd = Map(source_file)
        print('Source: {}'.format(source_file))
    except Exception as e:
        print('Error: {}'.format(e))
        return False
    
    X = Xd.data
    
    #make a valid mask; we'll use this to correct for downpush when interpolating AIA
    validMask = 1.0 * (X > 0) 
    X[np.where(X<=0.0)] = 0.0

    expTime = max(Xd.meta['EXPTIME'],1e-2)
    quality = Xd.meta['QUALITY']    

    if quality != 0:
        print('Quality flag is not zero: {}'.format(quality))
        return False
    
    # Target angular size
    trgtAS = 976.0

    # Scale factor
    rad = Xd.meta['RSUN_OBS']
    scale_factor = trgtAS/rad

    #fix the translation
    t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
    #rescale and keep center
    XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))
    Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
    Xd = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))

    #correct for interpolating over valid pixels
    Xr = np.divide(Xr,(Xd+1e-8))

    # SKIPPING DEGRADATION CORRECTION, MIGHT REVISIT
    #correct for exposure time and AIA degradation correction
    correction = 1.
    Xr = Xr / (expTime*correction)

    #figure out the integer factor to downsample by mean
    divideFactor = int(X.shape[0] / resolution)

    Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
    #make it a sum rather than a mean by multiplying by the number of pixels that were used
    Xr = Xr*divideFactor*divideFactor

    #cast to fp32
    Xr = Xr.astype('float32')

    os.makedirs(os.path.dirname(target_file), exist_ok=True)    
    np.savez_compressed(target_file, x=Xr)

    print('Target: {}'.format(target_file))
    return True


def main():
    description = 'FDL-X 2024, Radiation team, SDO AIA data process script'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--target_dir', type=str, help='Destination directory', required=True)
    parser.add_argument('--max_workers', type=int, default=4, help='Max workers')
    parser.add_argument('--worker_chunk_size', type=int, default=1, help='Chunk size per worker')
    parser.add_argument('--resolution', type=int, default=512, help='Pixel resolution of processed images. Should be a divisor of 1024.')

    args = parser.parse_args()

    print(description)

    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    # walk through the source directory with glob, find all .fits files, and create a corresponding file name ending in .npz in the target dir, keeping the directory structure

    # set the source and target directories, strip final slash if present
    source_dir = args.source_dir.rstrip('/')
    target_dir = args.target_dir.rstrip('/')

    # get all .fits files in the source directory
    fits_files = glob(os.path.join(source_dir, '**', '*.fits'), recursive=True)

    # create a list of tuples with the source and target file names
    # be careful to strip or add slashes as needed
    file_names = []
    for source_file in fits_files:
        target_file = source_file.replace(source_dir, target_dir).replace('.fits', '.npz')
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