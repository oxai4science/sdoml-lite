import argparse
import pprint
import sys
import datetime
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sunpy.coordinates import sun
from sunpy.map import Map
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


def find_sun_ratio(data):
    # Returns the ratio: diameter of the solar disk / length of the image side
    if data.shape[0] != data.shape[1]:
        raise ValueError('Expecting square image')
    size = data.shape[0]
    mid_i = size // 2
    color = data[mid_i, 0]
    space_left = 0
    for i in range(1, mid_i):
        if data[mid_i, i] == color:
            color = data[mid_i, i]
            space_left += 1
        else:
            break
    color = data[0, mid_i]
    space_top = 0
    for i in range(1, mid_i):
        if data[i, mid_i] == color:
            color = data[i, mid_i]
            space_top += 1
        else:
            break
    space = (space_left + space_top)/2.
    ratio = (size - 2*space)/size
    return ratio


# HMI postprocessing based on SDOML code, with some modifications
# https://github.com/SDOML/SDOML/blob/bea846347b2cd64d81fdcf1baf88a245a1bcb429/hmi_fits_to_np.py
def process(args):
    source_file, target_file, resolution = args

    try:
        X = read_hmi_jpg(source_file)
        print('\nSource: {}'.format(source_file))
    except Exception as e:
        print('Error: {}'.format(e))
        return False
    
    # The HMI data product we use is not in FITS format and does not provide the metadata RSUN_OBS, so we need to estimate the scale factor

    # Original scale factor calculation which we cannot do
    # rad = Xd.meta['RSUN_OBS']
    # scale_factor = trgtAS/rad
    
    # Method 1: Use the angular radius of the Sun as seen from Earth (instead of SDO, but it should be close enough))
    # Based on code: https://github.com/sunpy/sunpy/blob/934a4439d420a6edf0196cc9325e770121db3d39/sunpy/coordinates/sun.py#L53
    # trgtAS = 976.0
    # hmi_basename = os.path.basename(source_file)
    # date = datetime.datetime.strptime(hmi_basename[:13], '%Y%m%d_%H%M')    
    # rad = sun.angular_radius(date).to('arcsec').value
    # scale_factor = trgtAS/rad

    # Method 2: Use the ratio of the diameter of the solar disk to the length of the image side
    # target_sun_ratio = 0.8 # This is the end result of the AIA scaling (AIA images end up having 10% length on each side of the solar disk)
    # ratio = find_sun_ratio(X)
    # scale_factor = target_sun_ratio / ratio

    # Method 3: Use the RSUN_OBS metadata from corresponding AIA FITS files which we have. There seems to be a simple relationship between AIA RSUN_OBS and HMI RSUN_OBS which we assume to be constant.
    # aia_rsun_obs = Map(aia_file).meta['RSUN_OBS']
    # trgtAS = 976.0
    # aia_scale_factor = trgtAS / aia_rsun_obs
    # scale_factor = aia_scale_factor * 0.85

    hmi_basename = os.path.basename(source_file)
    hmi_dir = os.path.dirname(source_file)
    date = datetime.datetime.strptime(hmi_basename[:13], '%Y%m%d_%H%M')
    # Try to find a very close AIA file
    aia_found = False
    if date.minute == 15:
        date = date.replace(minute=14)
    elif date.minute == 45:
        date = date.replace(minute=44)
    aia_files_pattern_prefix = datetime.datetime.strftime(date, 'AIA%Y%m%d_%H%M')
    aia_files_pattern = aia_files_pattern_prefix + '*.fits'
    aia_files_found = glob(os.path.join(hmi_dir, aia_files_pattern))
    if len(aia_files_found) > 0:
        aia_file_preferences = [os.path.join(hmi_dir, aia_files_pattern_prefix + '_' + postfix + '.fits') for postfix in ['0131','0171','0193','0211','0094','1600','1700']]
        aia_files = [file for file in aia_file_preferences if file in aia_files_found]
        if len(aia_files) > 0:
            aia_file = aia_files[0]
            aia_found = True

    # If no close AIA file is found, use any AIA file from the same day
    if not aia_found:
        aia_files_found = glob(os.path.join(hmi_dir, 'AIA*.fits'))
        if len(aia_files_found) > 0:
            aia_file = aia_files_found[0]
            aia_found = True

    if aia_found:
        print('Using AIA file metadata for RSUN_OBS: {}'.format(os.path.basename(aia_file)))
        aia_rsun_obs = Map(aia_file).meta['RSUN_OBS']
        trgtAS = 976.0
        aia_scale_factor = trgtAS / aia_rsun_obs
        scale_factor = aia_scale_factor * 0.85 # This is a factor determined empirically by inspecting some images for various dates

    # If no AIA files are found, fall back to Method 2 (happens almost never)
    if not aia_found:
        print('No AIA files found for HMI file: {}'.format(source_file))
        target_sun_ratio = 0.8 # This is the end result of the AIA scaling (AIA images end up having 10% length on each side of the solar disk)
        ratio = find_sun_ratio(X)
        scale_factor = target_sun_ratio / ratio

    print('AIA scale factor                     : {}'.format(aia_scale_factor))
    print('Scale factor                         : {}'.format(scale_factor))

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
    results = process_map(process, file_names, max_workers=args.max_workers, chunksize=args.worker_chunk_size)

    print('Files processed: {}'.format(results.count(True)))
    print('Files failed   : {}'.format(results.count(False)))
    print('Files total    : {}'.format(len(results)))
    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()