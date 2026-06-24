import sys
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

def process_multisector(tic_id):
    print(f"Searching MAST for all available sectors for TIC {tic_id}...")
    
    # 1. Query all sectors
    search_result = lk.search_lightcurve(f"TIC {tic_id}", author='SPOC')
    
    if len(search_result) == 0:
        print(f"No SPOC light curves found for TIC {tic_id}.")
        return
        
    print(f"Found {len(search_result)} observations.")
    
    # 2. Download all available sectors
    lc_collection = search_result.download_all()
    
    if lc_collection is None or len(lc_collection) == 0:
        print("Failed to download light curves.")
        return
        
    # Get the first sector for comparison
    lc_single = lc_collection[0].normalize().remove_nans().remove_outliers()
    single_sector_num = lc_collection[0].sector
    
    print("Normalizing and stitching chronologically...")
    # 3 & 4. Normalize independently and Stitch
    # lightkurve's stitch method handles normalization natively if we pass corrector_func
    # By default, it normalizes and concatenates.
    lc_stitched = lc_collection.stitch()
    
    # 5 & 6. Remove duplicate cadences and handle gaps
    # Drop NaNs
    lc_stitched = lc_stitched.remove_nans()
    
    # Remove duplicate times (can happen at sector overlaps)
    _, unique_indices = np.unique(lc_stitched.time.value, return_index=True)
    lc_stitched = lc_stitched[unique_indices]
    
    # Remove extreme outliers that might be introduced by edge effects
    lc_stitched = lc_stitched.remove_outliers(sigma=5)
    
    print(f"Stitched light curve contains {len(lc_stitched)} cadences spanning {lc_stitched.time.value[-1] - lc_stitched.time.value[0]:.1f} days.")
    
    # Output to FITS
    out_fits = f"TIC_{tic_id}_stitched_lightcurve.fits"
    lc_stitched.to_fits(out_fits, overwrite=True)
    print(f"Saved stitched light curve to {out_fits}")
    
    # 7. Run BLS
    print("Running BLS on single sector...")
    # Baseline for single
    baseline_single = lc_single.time.value[-1] - lc_single.time.value[0]
    bls_single = lc_single.to_periodogram(method='bls', minimum_period=0.5, maximum_period=min(20.0, baseline_single / 2.0), frequency_factor=1.0)
    
    print("Running BLS on multi-sector stitched data...")
    # Baseline for multi-sector is much longer, allowing us to probe >20 day periods
    baseline_multi = lc_stitched.time.value[-1] - lc_stitched.time.value[0]
    max_period = min(365.0, baseline_multi / 3.0)
    bls_multi = lc_stitched.to_periodogram(method='bls', minimum_period=0.5, maximum_period=max_period, frequency_factor=5000)
    
    best_p_single = bls_single.period_at_max_power.value
    best_p_multi = bls_multi.period_at_max_power.value
    
    print(f"Single Sector {single_sector_num} Best Period: {best_p_single:.4f} days")
    print(f"Stitched Multi-Sector Best Period: {best_p_multi:.4f} days")
    
    # 8. Generate comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot light curves
    axes[0, 0].scatter(lc_single.time.value, lc_single.flux.value, s=1, color='black', alpha=0.5)
    axes[0, 0].set_title(f"TIC {tic_id} - Single Sector {single_sector_num} (Baseline: {baseline_single:.1f}d)")
    axes[0, 0].set_xlabel("Time (BTJD)")
    axes[0, 0].set_ylabel("Normalized Flux")
    
    axes[0, 1].scatter(lc_stitched.time.value, lc_stitched.flux.value, s=1, color='blue', alpha=0.5)
    axes[0, 1].set_title(f"TIC {tic_id} - Stitched {len(search_result)} Sectors (Baseline: {baseline_multi:.1f}d)")
    axes[0, 1].set_xlabel("Time (BTJD)")
    axes[0, 1].set_ylabel("Normalized Flux")
    
    # Plot periodograms
    axes[1, 0].plot(bls_single.period, bls_single.power, color='black')
    axes[1, 0].axvline(best_p_single, color='red', linestyle='--', alpha=0.5, label=f'P={best_p_single:.2f}d')
    axes[1, 0].set_title("Single Sector BLS Periodogram")
    axes[1, 0].set_xlabel("Period (days)")
    axes[1, 0].set_ylabel("BLS Power")
    axes[1, 0].legend()
    
    axes[1, 1].plot(bls_multi.period, bls_multi.power, color='blue')
    axes[1, 1].axvline(best_p_multi, color='red', linestyle='--', alpha=0.5, label=f'P={best_p_multi:.2f}d')
    axes[1, 1].set_title("Stitched Multi-Sector BLS Periodogram")
    axes[1, 1].set_xlabel("Period (days)")
    axes[1, 1].set_ylabel("BLS Power")
    axes[1, 1].legend()
    
    plt.tight_layout()
    out_plot = "single_sector_vs_multisector.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {out_plot}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Use a known multi-sector exoplanet as a default test if none provided
        # TIC 283722336 (TOI-134) or similar. Let's just use 283722336 for demonstration.
        target = "283722336"
        
    process_multisector(target)
