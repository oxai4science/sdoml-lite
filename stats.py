import argparse
import pprint
import sys
import os
import datetime
from glob import glob
from tqdm import tqdm
import numpy as np
import json


from util import aia_process, aia_load_degradations


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def main():
    description = "SDOML-lite, statistics for downloaded data"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing downloaded data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output statistics')
    parser.add_argument('--wavelengths', nargs='+', default=[131, 171, 193, 211, 1600], help='Wavelengths')
    parser.add_argument('--instrument', type=str, choices=['AIA', 'HMI'], required=True, help='Instrument type')
    parser.add_argument('--hist_bins', type=int, default=256, help='Number of histogram bins for pixel values')
    parser.add_argument('--resolution', type=int, default=512, help='Pixel resolution of processed images. Should be a divisor of 1024.')
    parser.add_argument('--degradation_dir', type=str, default='./degradation/v9', help='Directory with degradation correction files')
    parser.add_argument('--max_num_files', type=int, default=None, help='Maximum number of files to process (for testing purposes)')

    args = parser.parse_args()
    print(description)
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    if not os.path.exists(args.data_dir):
        print(f"Data directory '{args.data_dir}' does not exist.")
        return
    if args.instrument == 'AIA':
        wavelengths = args.wavelengths
        print(f"Calculating statistics for AIA data in {args.data_dir} for wavelengths: {wavelengths}")

        # AIA files are .fits files under subdirectories named by YYYY/MM/DD, each file named like AIAYYYMMDD_HHMM_WWWW.fits where WWWW is the wavelength pre-padded by zeros if needed
        # We need to make a histogram of pixel values for each wavelength
        # We also need to track the minimum, maximum, mean, variance

        # use aia_process to read and process the data, save to a temporary file, and then calculate statistics

        print('Loading degradations')
        degradations = aia_load_degradations(args.degradation_dir, wavelengths)

        stats = {}
        for wavelength in wavelengths:
            stats[wavelength] = {
                'min': [],
                'max': [],
                'mean': [],
                'var': [],
                'min_log10': [],
                'max_log10': [],
                'mean_log10': [],
                'var_log10': [],
                'min_sqrt': [],
                'max_sqrt': [],
                'mean_sqrt': [],
                'var_sqrt': [],
                'hist': np.zeros(args.hist_bins, dtype=int),
                'hist_log10': np.zeros(args.hist_bins, dtype=int),
                'hist_sqrt': np.zeros(args.hist_bins, dtype=int),
            }

        files = glob(os.path.join(args.data_dir, '**', '*.fits'), recursive=True)
        if args.max_num_files is not None:
            files = files[:args.max_num_files]

        if not files:
            print(f"No AIA files found in {args.data_dir}.")
            return
        
        print('Phase 1: Compute overall min, max, mean, var')
        for file in tqdm(files, desc='Phase 1: Stats', unit='file'):
            try:
                file = os.path.abspath(file)
                wavelength = int(file.split("_")[-1].replace(".fits",""))
                if wavelength not in wavelengths:
                    continue

                # Process the file and get pixel data
                # make temporary file name to save the processed data
                temp_file = file.replace('.fits', '_temp.npy')
                result = aia_process((file, temp_file, args.resolution, degradations))

                if result == False:
                    print(f"Failed to process file {file}. Skipping.")
                    continue
                pixel_data = np.load(temp_file)
                os.remove(temp_file)  # Clean up temporary file

                stats[wavelength]['min'].append(np.min(pixel_data))
                stats[wavelength]['max'].append(np.max(pixel_data))
                stats[wavelength]['mean'].append(np.mean(pixel_data))
                stats[wavelength]['var'].append(np.var(pixel_data))

                log_pixel_data = np.log10(pixel_data + 1)  # Avoid log(0)
                stats[wavelength]['min_log10'].append(np.min(log_pixel_data))
                stats[wavelength]['max_log10'].append(np.max(log_pixel_data))
                stats[wavelength]['mean_log10'].append(np.mean(log_pixel_data))
                stats[wavelength]['var_log10'].append(np.var(log_pixel_data))

                sqrt_pixel_data = np.sqrt(pixel_data)
                stats[wavelength]['min_sqrt'].append(np.min(sqrt_pixel_data))
                stats[wavelength]['max_sqrt'].append(np.max(sqrt_pixel_data))
                stats[wavelength]['mean_sqrt'].append(np.mean(sqrt_pixel_data))
                stats[wavelength]['var_sqrt'].append(np.var(sqrt_pixel_data))
                

                
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        for wavelength in wavelengths:
            if stats[wavelength]['min']:
                stats[wavelength]['min'] = np.array(stats[wavelength]['min'])
                stats[wavelength]['max'] = np.array(stats[wavelength]['max'])
                stats[wavelength]['mean'] = np.array(stats[wavelength]['mean'])
                stats[wavelength]['var'] = np.array(stats[wavelength]['var'])
                stats[wavelength]['min_log10'] = np.array(stats[wavelength]['min_log10'])
                stats[wavelength]['max_log10'] = np.array(stats[wavelength]['max_log10'])
                stats[wavelength]['mean_log10'] = np.array(stats[wavelength]['mean_log10'])
                stats[wavelength]['var_log10'] = np.array(stats[wavelength]['var_log10'])
                stats[wavelength]['min_sqrt'] = np.array(stats[wavelength]['min_sqrt'])
                stats[wavelength]['max_sqrt'] = np.array(stats[wavelength]['max_sqrt'])
                stats[wavelength]['mean_sqrt'] = np.array(stats[wavelength]['mean_sqrt'])
                stats[wavelength]['var_sqrt'] = np.array(stats[wavelength]['var_sqrt'])

                k = len(stats[wavelength]['mean'])
                global_min = np.min(stats[wavelength]['min'])
                global_max = np.max(stats[wavelength]['max'])
                global_mean = np.mean(stats[wavelength]['mean'])
                global_mean_of_second_moments = sum(v + mu**2 for v, mu in zip(stats[wavelength]['var'], stats[wavelength]['mean'])) / k
                global_var = global_mean_of_second_moments - global_mean**2
                stats[wavelength]['min'] = global_min
                stats[wavelength]['max'] = global_max
                stats[wavelength]['mean'] = global_mean
                stats[wavelength]['var'] = global_var

                # log data
                global_min_log10 = np.min(stats[wavelength]['min_log10'])
                global_max_log10 = np.max(stats[wavelength]['max_log10'])
                global_mean_log10 = np.mean(stats[wavelength]['mean_log10'])
                global_mean_of_second_moments_log10 = sum(v + mu**2 for v, mu in zip(stats[wavelength]['var_log10'], stats[wavelength]['mean_log10'])) / k
                global_var_log10 = global_mean_of_second_moments_log10 - global_mean_log10**2
                stats[wavelength]['min_log10'] = global_min_log10
                stats[wavelength]['max_log10'] = global_max_log10
                stats[wavelength]['mean_log10'] = global_mean_log10
                stats[wavelength]['var_log10'] = global_var_log10

                # sqrt data
                global_min_sqrt = np.min(stats[wavelength]['min_sqrt'])
                global_max_sqrt = np.max(stats[wavelength]['max_sqrt'])
                global_mean_sqrt = np.mean(stats[wavelength]['mean_sqrt'])
                global_mean_of_second_moments_sqrt = sum(v + mu**2 for v, mu in zip(stats[wavelength]['var_sqrt'], stats[wavelength]['mean_sqrt'])) / k
                global_var_sqrt = global_mean_of_second_moments_sqrt - global_mean_sqrt**2
                stats[wavelength]['min_sqrt'] = global_min_sqrt
                stats[wavelength]['max_sqrt'] = global_max_sqrt
                stats[wavelength]['mean_sqrt'] = global_mean_sqrt
                stats[wavelength]['var_sqrt'] = global_var_sqrt

                stats[wavelength]['hist_bin_edges'] = np.linspace(global_min, global_max, args.hist_bins + 1)
                stats[wavelength]['hist_bin_edges_log10'] = np.linspace(global_min_log10, global_max_log10, args.hist_bins + 1)
                stats[wavelength]['hist_bin_edges_sqrt'] = np.linspace(global_min_sqrt, global_max_sqrt, args.hist_bins + 1)

            else:
                print(f"No valid data for wavelength {wavelength}.")

        print('Phase 1: Overall statistics computed')
        for wavelength in wavelengths:
            print(f"Wavelength {wavelength} nm:")
            print(f"  Min  : {stats[wavelength]['min']}")
            print(f"  Max  : {stats[wavelength]['max']}")
            print(f"  Mean : {stats[wavelength]['mean']}")
            print(f"  Var  : {stats[wavelength]['var']}")

        print('Phase 2: Compute histograms')
        
        for file in tqdm(files, desc='Phase 2: Histograms', unit='file'):
            try:
                file = os.path.abspath(file)
                wavelength = int(file.split("_")[-1].replace(".fits",""))
                if wavelength not in wavelengths:
                    continue

                # Process the file and get pixel data
                temp_file = file.replace('.fits', '_temp.npy')
                aia_process((file, temp_file, args.resolution, degradations))
                pixel_data = np.load(temp_file)
                os.remove(temp_file)  # Clean up temporary file

                # Update histogram
                hist, _ = np.histogram(pixel_data, bins=stats[wavelength]['hist_bin_edges'])
                stats[wavelength]['hist'] += hist

                # Update log histogram
                log_pixel_data = np.log10(pixel_data + 1)  # Avoid log(0)
                hist_log10, _ = np.histogram(log_pixel_data, bins=stats[wavelength]['hist_bin_edges_log10'])
                stats[wavelength]['hist_log10'] += hist_log10
                
                # Update sqrt histogram
                sqrt_pixel_data = np.sqrt(pixel_data)
                hist_sqrt, _ = np.histogram(sqrt_pixel_data, bins=stats[wavelength]['hist_bin_edges_sqrt'])
                stats[wavelength]['hist_sqrt'] += hist_sqrt


            except Exception as e:
                print(f"Error processing file {file}: {e}")


        print('Phase 2: Histograms computed')
        for wavelength in wavelengths:
            if stats[wavelength]['hist'].sum() > 0:
                print(f"Wavelength {wavelength} nm histogram:")
                print(stats[wavelength]['hist'])
            else:
                print(f"No valid histogram data for wavelength {wavelength}.")


        # Save statistics to output directory
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_file = os.path.join(args.output_dir, 'aia_statistics.json')
        with open(output_file, 'w') as f:
            json.dump(to_serializable(stats), f, indent=4)
        print(f"Statistics saved to {output_file}")
        print("AIA statistics calculation completed successfully.")

        # also plot histograms
        import matplotlib.pyplot as plt
        for wavelength in wavelengths:
            if stats[wavelength]['hist'].sum() > 0:
                print(f"Plotting histograms for wavelength {wavelength} nm")
                plt.figure(figsize=(10, 6))
                plt.bar(stats[wavelength]['hist_bin_edges'][:-1], stats[wavelength]['hist'], width=np.diff(stats[wavelength]['hist_bin_edges']), align='edge', edgecolor='black')
                plt.title(f"Histogram of Pixel Values for AIA {wavelength} nm")
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                plt.grid()
                plt.savefig(os.path.join(args.output_dir, f'aia_histogram_{wavelength}.pdf'))
                plt.close()

                # Save log histogram
                plt.figure(figsize=(10, 6))
                plt.bar(stats[wavelength]['hist_bin_edges_log10'][:-1], stats[wavelength]['hist_log10'], width=np.diff(stats[wavelength]['hist_bin_edges_log10']), align='edge', edgecolor='black')
                plt.title(f"Log Histogram of Pixel Values for AIA {wavelength} nm")
                plt.xlabel("Log Pixel Value")
                plt.ylabel("Frequency")
                plt.grid()
                plt.savefig(os.path.join(args.output_dir, f'aia_histogram_log10_{wavelength}.pdf'))
                plt.close()

                # Save square root histogram
                plt.figure(figsize=(10, 6))
                plt.bar(stats[wavelength]['hist_bin_edges_sqrt'][:-1], stats[wavelength]['hist_sqrt'], width=np.diff(stats[wavelength]['hist_bin_edges_sqrt']), align='edge', edgecolor='black')
                plt.title(f"Square Root Histogram of Pixel Values for AIA {wavelength} nm")
                plt.xlabel("Square Root of Pixel Value")
                plt.ylabel("Frequency")
                plt.grid()
                plt.savefig(os.path.join(args.output_dir, f'aia_histogram_sqrt_{wavelength}.pdf'))
                plt.close()

            else:
                print(f"No valid histogram data for wavelength {wavelength}.")


        
    elif args.instrument == 'HMI':
        raise NotImplementedError("HMI statistics calculation is not implemented yet.")
    else:
        print(f"Unknown instrument type: {args.instrument}. Supported types are 'AIA' and 'HMI'.")

if __name__ == "__main__":
    main()