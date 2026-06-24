import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

def compute_snr(lc, period, t0, duration_hours, depth):
    phase = (lc.time.value - t0 + 0.5 * period) % period - 0.5 * period
    duration_days = duration_hours / 24.0
    
    in_transit_mask = np.abs(phase) < (duration_days / 2.0)
    out_of_transit_mask = ~in_transit_mask
    
    n_in_transit = np.sum(in_transit_mask)
    if n_in_transit == 0:
        return 0.0
        
    out_of_transit_std = np.nanstd(lc.flux.value[out_of_transit_mask])
    if out_of_transit_std == 0:
        return 0.0
        
    snr = (depth / out_of_transit_std) * np.sqrt(n_in_transit)
    return snr

def run_bls_search(data_dir="data", plot_dir="plots"):
    """
    Reads detrended FITS files, runs Box Least Squares (BLS) period search,
    and plots the periodogram and the phase-folded light curve.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Find detrended files
    detrended_files = glob.glob(os.path.join(data_dir, "*_detrended.fits"))
    if not detrended_files:
        print(f"No detrended FITS files found in '{data_dir}/'. Please ensure Stage 2 saves them.")
        return
        
    print(f"Found {len(detrended_files)} detrended light curves for BLS search.")

    summary_table = []

    for file_path in detrended_files:
        filename = os.path.basename(file_path)
        
        # Extract the exact TIC ID from the filename (e.g. "TIC_55525572")
        tic_id = filename.split('_TESS')[0]
        
        print(f"\n{'-'*50}\nRunning BLS for {tic_id}...")
        
        import hashlib
        import datetime
        file_mtime = os.path.getmtime(file_path)
        mtime_str = datetime.datetime.fromtimestamp(file_mtime).isoformat()
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        print(f"DEBUG FILE: {file_path}")
        print(f"DEBUG MTIME: {mtime_str}")
        print(f"DEBUG MD5: {file_hash}")
        
        try:
            lc = lk.read(file_path)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue
            
        # -------------------------------------------------------------
        # 1. Run Box Least Squares
        # -------------------------------------------------------------
        # We search periods between 0.5 days and half the observation baseline
        baseline = lc.time[-1].value - lc.time[0].value
        max_period = min(20.0, baseline / 2.0) 
        period_grid = np.linspace(0.5, max_period, 10000)
        
        print(f"DEBUG BASELINE START: {lc.time[0].value}")
        print(f"DEBUG BASELINE END: {lc.time[-1].value}")
        print(f"DEBUG GRID FIRST: {period_grid[0]}")
        print(f"DEBUG GRID LAST: {period_grid[-1]}")
        print(f"DEBUG GRID LENGTH: {len(period_grid)}")
        
        # Astropy's BLS is conveniently wrapped by lightkurve's to_periodogram
        periodogram = lc.to_periodogram(method='bls', period=period_grid)
        
        # -------------------------------------------------------------
        # 2. Extract best-fit parameters
        # -------------------------------------------------------------
        best_period = periodogram.period_at_max_power.value
        best_duration = periodogram.duration_at_max_power.value * 24.0 # Convert to hours
        best_depth = periodogram.depth_at_max_power
        max_power = periodogram.max_power
        print(f"Sanity check — max(periodogram.power) = {periodogram.power.max():.2f}, matches reported max_power: {np.isclose(periodogram.power.max(), max_power.value)}")
        best_t0 = periodogram.transit_time_at_max_power.value
        
        snr = compute_snr(lc, best_period, best_t0, best_duration, best_depth)
        
        print(f"Best-fit Period:   {best_period:.4f} days")
        print(f"Best-fit Duration: {best_duration:.2f} hours")
        print(f"Best-fit Depth:    {best_depth:.4f} (relative flux drop)")
        print(f"BLS Power Score:   {max_power:.2f}")
        print(f"SNR:               {snr:.2f}")
        
        summary_table.append({
            'tic': tic_id,
            'period': best_period,
            'duration': best_duration,
            'depth': best_depth,
            'snr': snr
        })
        

            
        # -------------------------------------------------------------
        # 4. Plotting
        # -------------------------------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Top: BLS Periodogram
        periodogram.plot(ax=axes[0], color='blue')
        axes[0].set_title(f"BLS Periodogram - {tic_id}")
        axes[0].axvline(x=best_period, color='red', linestyle='--', alpha=0.5, label=f'Best P={best_period:.2f}d')
        axes[0].legend()
        
        # Bottom: Phase-folded Light Curve
        lc_folded = lc.fold(period=best_period, epoch_time=best_t0)
        lc_folded.plot(ax=axes[1], color='black', marker='.', linestyle='none', alpha=0.5, label='Folded Data')
        
        # Overlay the BLS square-wave model
        bls_model = periodogram.get_transit_model(period=best_period,
                                                  transit_time=best_t0,
                                                  duration=periodogram.duration_at_max_power.value)
        model_folded = bls_model.fold(period=best_period, epoch_time=best_t0)
        
        # Sort model by phase to draw a clean line
        sort_idx = np.argsort(model_folded.time.value)
        axes[1].plot(model_folded.time.value[sort_idx], model_folded.flux.value[sort_idx], 
                     color='red', linewidth=2, label='BLS Model Fit')
                     
        axes[1].set_title(f"Phase-folded Light Curve (P={best_period:.4f} d)")
        axes[1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{tic_id}_bls_analysis.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved BLS plots to {plot_path}")

    # Print Summary Table
    if summary_table:
        print("\n" + "="*70)
        print("BLS SEARCH SUMMARY TABLE")
        print("="*70)
        print(f"{'Target':<15} | {'Period (d)':<12} | {'Duration (h)':<14} | {'Depth':<10} | {'SNR':<8}")
        print("-" * 70)
        for row in summary_table:
            print(f"{row['tic']:<15} | {row['period']:<12.4f} | {row['duration']:<14.2f} | {row['depth']:<10.4f} | {row['snr']:<8.2f}")
        print("="*70 + "\n")

if __name__ == "__main__":
    run_bls_search()
