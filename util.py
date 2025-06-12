import os
import datetime
import numpy as np
from glob import glob
from sunpy.map import Map
import skimage.transform
import matplotlib.pyplot as plt


def has_nan_or_inf(data):
    if np.isnan(data).any():
        return True
    if np.isinf(data).any():
        return True
    if np.isneginf(data).any():
        return True
    return False



# Mask to remove text printed lower left
# Example: http://jsoc.stanford.edu/data/hmi/images/2024/01/01/20240101_000000_M_1k.jpg
hmi_mask = np.ones((1024,1024))
hmi_mask[990:,:300] = 0.

def hmi_read_jpg(file_name):
    x = plt.imread(file_name)
    x = x.mean(axis=2)
    x *= hmi_mask
    x /= 255.
    return x


def hmi_find_sun_ratio(data):
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
def hmi_process(args):
    source_file, target_file, resolution = args

    try:
        X = hmi_read_jpg(source_file)
        print('\nSource: {}'.format(source_file))
    except Exception as e:
        print('Error: {}'.format(e))
        return False
    
    # Fail if the file is too small, indicates an empty image (all black with just the text label)
    # There are 91 such files between in 2010 May and July
    if os.path.getsize(source_file) < 30000:
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
        try:
            aia_rsun_obs = Map(aia_file).meta['RSUN_OBS']
            trgtAS = 976.0
            aia_scale_factor = trgtAS / aia_rsun_obs
            scale_factor = aia_scale_factor * 0.85 # This is a factor determined empirically by inspecting some images for various dates
            print('AIA scale factor                    : {}'.format(aia_scale_factor))
            print('Scale factor (based on AIA)         : {}'.format(scale_factor))
        except Exception as e:
            print('Error: {}'.format(e))
            print('Failed to read AIA file metadata: {}'.format(aia_file))
            aia_found = False

    # If no AIA files are found, fall back to Method 2 (should not happen often)
    if not aia_found:
        print('No AIA files found for HMI file     : {}'.format(source_file))
        target_sun_ratio = 0.8 # This is the end result of the AIA scaling (AIA images end up having 10% length on each side of the solar disk)
        ratio = hmi_find_sun_ratio(X)
        scale_factor = target_sun_ratio / ratio
        print('Scale factor (not based on AIA)     : {}'.format(scale_factor))

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



def aia_load_degradations(degradation_dir, wavelengths):
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


def aia_normalize(args):
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
def aia_process(args):
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