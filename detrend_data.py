import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

def detrend_all_lightcurves(data_dir="data", plot_dir="plots", window_hours=12.0):
    """
    Reads all FITS light curves from data_dir, normalizes them, 
    and applies a Savitzky-Golay filter via lightkurve's flatten() method.
    Plots the results and checks for potential overfitting.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Find all FITS files downloaded in Stage 1
    all_fits = glob.glob(os.path.join(data_dir, "*.fits"))
    
    fits_files = []
    skipped_files = []
    for f in all_fits:
        if "_detrended" in os.path.basename(f):
            skipped_files.append(f)
        else:
            fits_files.append(f)
            
    if skipped_files:
        print(f"Skipped {len(skipped_files)} already-detrended files:")
        for sf in skipped_files:
            print(f"  - {os.path.basename(sf)}")

    if not fits_files:
        print(f"No raw FITS files found in '{data_dir}/' to detrend. Please run the download script first.")
        return

    print(f"\nFound {len(fits_files)} raw light curves to detrend.")

    for file_path in fits_files:
        filename = os.path.basename(file_path)
        print(f"\n{'-'*40}\nProcessing {filename}...")

        # Read the light curve
        try:
            lc = lk.read(file_path)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue
            
        # Normalize the raw light curve so we can fairly compare variance
        lc_norm = lc.normalize()

        # Remove outliers BEFORE flattening so they don't distort the local trend
        initial_points = len(lc_norm)
        lc_norm_orig = lc_norm.copy()
        
        # We use return_mask=True to see exactly what got removed
        lc_norm, outlier_mask = lc_norm.remove_outliers(sigma=5, return_mask=True)
        outliers = lc_norm_orig[outlier_mask]
        outliers_removed = len(outliers)
        print(f"Removed {outliers_removed} outlier points (sigma=5).")
        
        if outliers_removed > 0:
            print("  Removed points (Time, Flux):")
            for t, f in zip(outliers.time.value, outliers.flux.value):
                print(f"    - BTJD {t:.3f}: Flux {f:.4f}")
        
        # Find the point near BTJD 1417 in the ORIGINAL light curve
        target_time = 1417.0
        idx_1417 = np.argmin(np.abs(lc_norm_orig.time.value - target_time))
        t_1417 = lc_norm_orig.time.value[idx_1417]
        f_1417 = lc_norm_orig.flux.value[idx_1417]
        
        # Calculate its sigma distance from the median
        median_flux = np.nanmedian(lc_norm_orig.flux.value)
        std_flux = np.nanstd(lc_norm_orig.flux.value)
        sigma_1417 = abs(f_1417 - median_flux) / std_flux
        print(f"  Point near 1417 (BTJD {t_1417:.3f}): Flux {f_1417:.4f} is {sigma_1417:.2f} sigma from overall median.")
        
        # Confirm that the object passed into flatten is the cleaned one
        print(f"  Confirming light curve for flatten() has length {len(lc_norm)} (initial was {initial_points})")

        # Calculate cadence in hours to determine a physical window length
        # np.diff gets the time difference between consecutive data points
        cadence_days = np.nanmedian(np.diff(lc_norm.time.value))
        cadence_hours = cadence_days * 24.0
        print(f"Detected Cadence: {cadence_hours:.4f} hours.")
        
        test_windows = [window_hours, 24.0, 36.0]
        final_window_hours = window_hours
        
        for wh in test_windows:
            # Calculate how many data points make up our desired physical time window
            window_length = int(wh / cadence_hours)
            
            # Savitzky-Golay filter requires the window_length to be an odd integer
            if window_length % 2 == 0:
                window_length += 1
                
            # Ensure it's greater than some minimum (e.g., 3) and not larger than the dataset
            window_length = max(3, window_length)
            if window_length > len(lc_norm):
                window_length = len(lc_norm)
                if window_length % 2 == 0: 
                    window_length -= 1

            print(f"Trying window_length = {window_length} cadences (approx {wh} hours)...")

            # Detrend using flatten (Savitzky-Golay filter by default)
            lc_flat, trend = lc_norm.flatten(window_length=window_length, return_trend=True)

            # Check how much variance was removed to flag potential overfitting
            var_raw = np.var(lc_norm.flux.value)
            var_flat = np.var(lc_flat.flux.value)
            
            variance_removed_pct = ((var_raw - var_flat) / var_raw) * 100

            if variance_removed_pct <= 50.0:
                print(f"  Variance removed: {variance_removed_pct:.1f}% (Acceptable)")
                final_window_hours = wh
                break
            else:
                print(f"  --> WARNING: Detrending removed {variance_removed_pct:.1f}% of the flux variance!")
                if wh == test_windows[-1]:
                    print(f"  Reached maximum window of {wh} hours. Proceeding with this window anyway.")
                    final_window_hours = wh
                else:
                    print("  This might mean the filter is over-fitting. Retrying with a longer window...")

        # Gap detection logic (identify but do not remove yet)
        time_diffs = np.diff(lc_norm.time.value)
        gap_threshold_days = 0.5 / 24.0  # 0.5 hours
        gap_indices = np.where(time_diffs > gap_threshold_days)[0]
        
        window_time_days = final_window_hours / 24.0
        is_near_gap = np.zeros(len(lc_norm), dtype=bool)
        
        for idx in gap_indices:
            gap_start = lc_norm.time.value[idx]
            gap_end = lc_norm.time.value[idx + 1]
            
            before_gap_mask = (lc_norm.time.value <= gap_start) & (lc_norm.time.value >= gap_start - window_time_days)
            after_gap_mask = (lc_norm.time.value >= gap_end) & (lc_norm.time.value <= gap_end + window_time_days)
            
            is_near_gap |= before_gap_mask
            is_near_gap |= after_gap_mask
            
        num_flagged = np.sum(is_near_gap)
        if num_flagged > 0:
            print(f"  Flagged {num_flagged} points as 'near-gap' (within {final_window_hours}h of a >0.5h gap).")
            print("  Gaps found around BTJD:")
            for idx in gap_indices:
                print(f"    - Gap: {lc_norm.time.value[idx]:.2f} to {lc_norm.time.value[idx+1]:.2f}")

        # Plotting: Raw vs Detrended
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Top subplot: Raw Normalized + Trend
        # Note: lightkurve's .plot() returns an Axes object
        lc_norm.plot(ax=axes[0], color='black', alpha=0.5, label='Raw Normalized Flux', marker='.', linestyle='none')
        trend.plot(ax=axes[0], color='red', linewidth=2, label=f'Trend (S-G, {final_window_hours}h window)')
        axes[0].set_title(f"Detrending: {filename}")
        axes[0].set_ylabel("Normalized Flux")
        axes[0].legend()
        
        # Bottom subplot: Flattened
        lc_flat.plot(ax=axes[1], color='black', alpha=0.8, label='Flattened (Detrended) Flux', marker='.', linestyle='none')
        axes[1].set_ylabel("Relative Flux")
        axes[1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, filename.replace('.fits', '_detrended.png'))
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved sanity-check plot to {plot_path}")
        
        # Save the detrended light curve
        detrended_path = os.path.join(data_dir, filename.replace('.fits', '_detrended.fits'))
        try:
            lc_flat.to_fits(path=detrended_path, overwrite=True)
            print(f"Saved detrended light curve to {detrended_path}")
        except Exception as e:
            print(f"Error saving detrended FITS: {e}")

if __name__ == "__main__":
    # You can change window_hours here if needed. 
    # 12 hours is a good starting point for finding typical exoplanets.
    detrend_all_lightcurves(window_hours=12.0)
