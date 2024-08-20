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


def has_nan_or_inf(data):
    if np.isnan(data).any():
        return True
    if np.isinf(data).any():
        return True
    if np.isneginf(data).any():
        return True
    return False


def normalize(args):
    try:
        source_file, aia_cutoffs = args
        target_file = source_file.replace('_unnormalized.npy', '.npy')

        data = np.load(source_file)
        print('\nSource: {}'.format(source_file))

        fn = os.path.basename(source_file).replace("_unnormalized.npy","")
        wavelength = int(fn.split("_")[-1])
        
        data = np.sqrt(data)
        c = np.sqrt(aia_cutoffs[wavelength])
        data = np.clip(data, a_min=None, a_max=c)
        data = data / c

        np.save(target_file, data)
        print('Target: {}'.format(target_file))
        # Delete the unnormalized file
        os.remove(source_file)
        print('Deleted: {}'.format(source_file))
        return True
    except Exception as e:
        print('Error: {}'.format(e))
        return False


# AIA postprocessing based on SDOML code, with some modifications
# https://github.com/SDOML/SDOML/blob/bea846347b2cd64d81fdcf1baf88a245a1bcb429/aia_fits_to_np.py
def process(args):
    source_file, target_file, resolution, degradations = args

    try:
        Xd = Map(source_file)
        print('\nSource: {}'.format(source_file))
    except Exception as e:
        print('Error: {}'.format(e))
        return False
    
    X = Xd.data
    
    #make a valid mask; we'll use this to correct for downpush when interpolating AIA
    validMask = 1.0 * (X > 0) 
    X[np.where(X<=0.0)] = 0.0

    fn = os.path.basename(source_file)
    fn2 = fn.split("_")[0].replace("AIA","")
    datestring = "%s-%s-%s" % (fn2[:4],fn2[4:6],fn2[6:8])
    wavelength = int(fn.split("_")[-1].replace(".fits",""))

    expTime = max(Xd.meta['EXPTIME'],1e-2)
    quality = Xd.meta['QUALITY']
    degrad = degradations[wavelength]
    if datestring in degrad:
        correction = degrad[datestring]
    else:
        correction = degrad['last']
        print('Degradation correction not found for wavelength {} and date {}, using the last value {}'.format(wavelength, datestring, correction))

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
    Xm = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))

    # Note: scaling leaves 10% of image width on each side of the Sun (and likewise for the image height). Measured in 512x512 images but should be the same for 1024x1024 images.

    #correct for interpolating over valid pixels
    # Xr = np.divide(Xr,(Xm+1e-8))
    # The mask application above in the original SDOML code might be bad. It ends up multiplying invalid pixels (value zero in mask) by the large factor 1e+8, instead of nullifying them. Simply multiply by the mask instead.
    Xr = Xr * Xm

    #correct for exposure time and AIA degradation correction
    Xr = Xr / (expTime*correction)

    #figure out the integer factor to downsample by mean
    divideFactor = int(X.shape[0] / resolution)

    Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))
    #make it a sum rather than a mean by multiplying by the number of pixels that were used
    Xr = Xr*divideFactor*divideFactor

    #cast to fp32
    Xr = Xr.astype('float32')

    Xr = np.flipud(Xr)

    if has_nan_or_inf(Xr):
        print('NaN or Inf found in the processed data')
        print('Source: {}'.format(source_file))
        print('X: {}'.format(X))
        print('Xr: {}'.format(Xr))
        print('expTime: {}'.format(expTime))
        print('correction: {}'.format(correction))
        return False

    os.makedirs(os.path.dirname(target_file), exist_ok=True)    
    np.save(target_file, Xr)

    print('Target: {}'.format(target_file))
    return wavelength, Xr.min(), Xr.max()


def load_degradations(degradation_dir, wavelengths):
    def getDegrad(fn):
        #map YYYY-MM-DD -> degradation parameter
        lines = open(fn).read().strip().split("\n")
        degrad = {}
        for l in lines:
            d, f = l.split(",")
            f = float(f)
            degrad[d[1:11]] = f
            degrad['last'] = f
        return degrad     
    #return wavelength -> (date -> degradation dictionary)
    degrads = {} 
    for wl in wavelengths:
        degrads[wl] = getDegrad(os.path.join(degradation_dir, 'degrad_{}.csv'.format(wl)))
    return degrads 


def main():
    description = 'FDL-X 2024, Radiation team, SDO AIA data processor'
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
    degradations = load_degradations(args.degradation_dir, args.wavelengths)

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
    results = process_map(process, file_names, max_workers=args.max_workers, chunksize=args.worker_chunk_size)

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
   
    results = process_map(normalize, file_names_normalize, max_workers=args.max_workers, chunksize=args.worker_chunk_size)

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