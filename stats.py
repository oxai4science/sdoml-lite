import argparse
import pprint
import sys
import os
import datetime
import random
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
    parser.add_argument('--hist_bins', type=int, default=1000, help='Number of histogram bins for pixel values')
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
                'num_files': 0,
                'min': [],
                'max': [],
                'mean': [],
                'var': [],
                'min_log': [],
                'max_log': [],
                'mean_log': [],
                'var_log': [],
                'hist': np.zeros(args.hist_bins, dtype=int),
                'hist_log': np.zeros(args.hist_bins, dtype=int),
                'normalized_min': [],
                'normalized_max': [],
                'normalized_mean': [],
                'normalized_var': [],
                'normalized_hist': np.zeros(args.hist_bins, dtype=int),
            }

        files = glob(os.path.join(args.data_dir, '**', '*.fits'), recursive=True)
        if args.max_num_files is not None:
            # select a random subset of files if max_num_files is specified
            if len(files) > args.max_num_files:
                print(f"Limiting to {args.max_num_files} files for processing.")
                random.seed(42)
                files = random.sample(files, args.max_num_files)
            else:
                print(f"Using all {len(files)} files found in {args.data_dir}.")

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
                temp_file_normalized = file.replace('.fits', '_temp_normalized.npy')
                result = aia_process((file, temp_file, args.resolution, degradations, False))
                result_normalized = aia_process((file, temp_file_normalized, args.resolution, degradations, True))

                if result == False or result_normalized == False:
                    print(f"Failed to process file {file}. Skipping.")
                    continue
                pixel_data = np.load(temp_file)
                pixel_data_normalized = np.load(temp_file_normalized)
                os.remove(temp_file)  # Clean up temporary file
                os.remove(temp_file_normalized)  # Clean up normalized temporary file

                stats[wavelength]['num_files'] += 1

                stats[wavelength]['min'].append(np.min(pixel_data))
                stats[wavelength]['max'].append(np.max(pixel_data))
                stats[wavelength]['mean'].append(np.mean(pixel_data))
                stats[wavelength]['var'].append(np.var(pixel_data))

                log_pixel_data = np.log1p(pixel_data)  # Avoid log(0)
                stats[wavelength]['min_log'].append(np.min(log_pixel_data))
                stats[wavelength]['max_log'].append(np.max(log_pixel_data))
                stats[wavelength]['mean_log'].append(np.mean(log_pixel_data))
                stats[wavelength]['var_log'].append(np.var(log_pixel_data))

                # Normalized statistics
                stats[wavelength]['normalized_min'].append(np.min(pixel_data_normalized))
                stats[wavelength]['normalized_max'].append(np.max(pixel_data_normalized))
                stats[wavelength]['normalized_mean'].append(np.mean(pixel_data_normalized))
                stats[wavelength]['normalized_var'].append(np.var(pixel_data_normalized))
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        for wavelength in wavelengths:
            if stats[wavelength]['min']:
                stats[wavelength]['min'] = np.array(stats[wavelength]['min'])
                stats[wavelength]['max'] = np.array(stats[wavelength]['max'])
                stats[wavelength]['mean'] = np.array(stats[wavelength]['mean'])
                stats[wavelength]['var'] = np.array(stats[wavelength]['var'])
                stats[wavelength]['min_log'] = np.array(stats[wavelength]['min_log'])
                stats[wavelength]['max_log'] = np.array(stats[wavelength]['max_log'])
                stats[wavelength]['mean_log'] = np.array(stats[wavelength]['mean_log'])
                stats[wavelength]['var_log'] = np.array(stats[wavelength]['var_log'])
                stats[wavelength]['normalized_min'] = np.array(stats[wavelength]['normalized_min'])
                stats[wavelength]['normalized_max'] = np.array(stats[wavelength]['normalized_max'])
                stats[wavelength]['normalized_mean'] = np.array(stats[wavelength]['normalized_mean'])
                stats[wavelength]['normalized_var'] = np.array(stats[wavelength]['normalized_var'])

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
                global_min_log = np.min(stats[wavelength]['min_log'])
                global_max_log = np.max(stats[wavelength]['max_log'])
                global_mean_log = np.mean(stats[wavelength]['mean_log'])
                global_mean_of_second_moments_log = sum(v + mu**2 for v, mu in zip(stats[wavelength]['var_log'], stats[wavelength]['mean_log'])) / k
                global_var_log = global_mean_of_second_moments_log - global_mean_log**2
                stats[wavelength]['min_log'] = global_min_log
                stats[wavelength]['max_log'] = global_max_log
                stats[wavelength]['mean_log'] = global_mean_log
                stats[wavelength]['var_log'] = global_var_log

                # Normalized statistics
                global_min_normalized = np.min(stats[wavelength]['normalized_min'])
                global_max_normalized = np.max(stats[wavelength]['normalized_max'])
                global_mean_normalized = np.mean(stats[wavelength]['normalized_mean'])
                global_mean_of_second_moments_normalized = sum(v + mu**2 for v, mu in zip(stats[wavelength]['normalized_var'], stats[wavelength]['normalized_mean'])) / k
                global_var_normalized = global_mean_of_second_moments_normalized - global_mean_normalized**2
                stats[wavelength]['normalized_min'] = global_min_normalized
                stats[wavelength]['normalized_max'] = global_max_normalized
                stats[wavelength]['normalized_mean'] = global_mean_normalized
                stats[wavelength]['normalized_var'] = global_var_normalized

                stats[wavelength]['hist_bin_edges'] = np.linspace(global_min, global_max, args.hist_bins + 1)
                stats[wavelength]['hist_bin_edges_log'] = np.linspace(global_min_log, global_max_log, args.hist_bins + 1)
                stats[wavelength]['normalized_hist_bin_edges'] = np.linspace(global_min_normalized, global_max_normalized, args.hist_bins + 1)

            else:
                print(f"No valid data for wavelength {wavelength}.")

        print('Phase 1: Overall statistics computed')
        for wavelength in wavelengths:
            print(f"Wavelength {wavelength} nm:")
            print(f"  Min  : {stats[wavelength]['min']}")
            print(f"  Max  : {stats[wavelength]['max']}")
            print(f"  Mean : {stats[wavelength]['mean']}")
            print(f"  Var  : {stats[wavelength]['var']}")
            print(f"  Min Log: {stats[wavelength]['min_log']}")
            print(f"  Max Log: {stats[wavelength]['max_log']}")
            print(f"  Mean Log: {stats[wavelength]['mean_log']}")
            print(f"  Var Log: {stats[wavelength]['var_log']}")
            print(f"  Normalized Min: {stats[wavelength]['normalized_min']}")
            print(f"  Normalized Max: {stats[wavelength]['normalized_max']}")
            print(f"  Normalized Mean: {stats[wavelength]['normalized_mean']}")
            print(f"  Normalized Var: {stats[wavelength]['normalized_var']}")
            print(f"  Number of files: {stats[wavelength]['num_files']}")

        print('Phase 2: Compute histograms')
        
        for file in tqdm(files, desc='Phase 2: Histograms', unit='file'):
            try:
                file = os.path.abspath(file)
                wavelength = int(file.split("_")[-1].replace(".fits",""))
                if wavelength not in wavelengths:
                    continue

                # Process the file and get pixel data
                temp_file = file.replace('.fits', '_temp.npy')
                temp_file_normalized = file.replace('.fits', '_temp_normalized.npy')
                result = aia_process((file, temp_file, args.resolution, degradations, False))
                result_normalized = aia_process((file, temp_file_normalized, args.resolution, degradations, True))
                pixel_data = np.load(temp_file)
                pixel_data_normalized = np.load(temp_file_normalized)
                os.remove(temp_file)  # Clean up temporary file
                os.remove(temp_file_normalized)  # Clean up normalized temporary file

                # Update histogram
                hist, _ = np.histogram(pixel_data, bins=stats[wavelength]['hist_bin_edges'])
                stats[wavelength]['hist'] += hist

                # Update log histogram
                log_pixel_data = np.log1p(pixel_data)  # Avoid log(0)
                hist_log, _ = np.histogram(log_pixel_data, bins=stats[wavelength]['hist_bin_edges_log'])
                stats[wavelength]['hist_log'] += hist_log

                # Update normalized histogram
                hist_normalized, _ = np.histogram(pixel_data_normalized, bins=stats[wavelength]['normalized_hist_bin_edges'])
                stats[wavelength]['normalized_hist'] += hist_normalized

            except Exception as e:
                print(f"Error processing file {file}: {e}")


        print('Phase 3: Compute 99.9th percentile from histograms')
        percentile = 0.999
        for wavelength in wavelengths:
            cumulative_hist = np.cumsum(stats[wavelength]['hist'])
            total_pixels = cumulative_hist[-1]
            percentile_index = np.searchsorted(cumulative_hist, total_pixels * percentile)
            stats[wavelength]['percentile_99_9'] = stats[wavelength]['hist_bin_edges'][percentile_index]
            print(f"Wavelength {wavelength} nm 99.9th percentile: {stats[wavelength]['percentile_99_9']} ({percentile_index}), max: {stats[wavelength]['max']}")

            stats[wavelength]['percentile_99_9_log'] = np.log(stats[wavelength]['percentile_99_9'])  # log of the 99.9th percentile value
            print(f"Wavelength {wavelength} nm log 99.9th percentile: {stats[wavelength]['percentile_99_9_log']}, max: {stats[wavelength]['max_log']}")

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
        # Plot combined histogram grids for all wavelengths
        n_wavelengths = len(wavelengths)
        fig, axes = plt.subplots(1, n_wavelengths, figsize=(6 * n_wavelengths, 5), squeeze=False)
        for idx, wavelength in enumerate(wavelengths):
            ax = axes[0, idx]
            if stats[wavelength]['hist'].sum() > 0:
                ax.bar(stats[wavelength]['hist_bin_edges'][:-1], stats[wavelength]['hist'],
                    width=np.diff(stats[wavelength]['hist_bin_edges']), align='edge', edgecolor='black')
                ax.set_title(f"{wavelength} nm")
                ax.set_xlabel("Pixel Value")
                ax.set_ylabel("Frequency")
                ax.grid()
            else:
                ax.set_title(f"{wavelength} nm (no data)")
                ax.axis('off')
        plt.suptitle("Histogram of Pixel Values for AIA")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(args.output_dir, 'aia_histogram.pdf'))
        plt.close(fig)

        # Log histograms
        fig, axes = plt.subplots(1, n_wavelengths, figsize=(6 * n_wavelengths, 5), squeeze=False)
        for idx, wavelength in enumerate(wavelengths):
            ax = axes[0, idx]
            if stats[wavelength]['hist_log'].sum() > 0:
                ax.bar(stats[wavelength]['hist_bin_edges_log'][:-1], stats[wavelength]['hist_log'],
                    width=np.diff(stats[wavelength]['hist_bin_edges_log']), align='edge', edgecolor='black')
                ax.set_title(f"{wavelength} nm")
                ax.set_xlabel("Log Pixel Value")
                ax.set_ylabel("Frequency")
                ax.grid()
            else:
                ax.set_title(f"{wavelength} nm (no data)")
                ax.axis('off')
        plt.suptitle("Log Histogram of Pixel Values for AIA")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(args.output_dir, 'aia_histogram_log.pdf'))
        plt.close(fig)

        # Normalized histograms
        fig, axes = plt.subplots(1, n_wavelengths, figsize=(6 * n_wavelengths, 5), squeeze=False)
        for idx, wavelength in enumerate(wavelengths):
            ax = axes[0, idx]
            if stats[wavelength]['normalized_hist'].sum() > 0:
                ax.bar(stats[wavelength]['normalized_hist_bin_edges'][:-1], stats[wavelength]['normalized_hist'],
                    width=np.diff(stats[wavelength]['normalized_hist_bin_edges']), align='edge', edgecolor='black')
                ax.set_title(f"{wavelength} nm (Normalized)")
                ax.set_xlabel("Normalized Pixel Value")
                ax.set_ylabel("Frequency")
                ax.grid()
            else:
                ax.set_title(f"{wavelength} nm (no data)")
                ax.axis('off')
        plt.suptitle("Normalized Histogram of Pixel Values for AIA")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(args.output_dir, 'aia_histogram_normalized.pdf'))
        plt.close(fig)

        
    elif args.instrument == 'HMI':
        raise NotImplementedError("HMI statistics calculation is not implemented yet.")
    else:
        print(f"Unknown instrument type: {args.instrument}. Supported types are 'AIA' and 'HMI'.")

if __name__ == "__main__":
    main()