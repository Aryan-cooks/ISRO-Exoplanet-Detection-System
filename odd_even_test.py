import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

def check_odd_even(lc, period, t0, duration_hours, tic_id, plot_dir="plots"):
    # Exclusion logic for TIC 38699825 near BTJD 1367.15
    if "38699825" in tic_id:
        exclude_mask = (lc.time.value > 1367.0) & (lc.time.value < 1367.3)
        num_excluded = np.sum(exclude_mask)
        if num_excluded > 0:
            lc = lc[~exclude_mask]
            print(f"  Note: Excluded {num_excluded} points near BTJD 1367.15 (Cycle #13) for TIC 38699825 due to adjacent data gap.")
            
    # Exclusion logic for TIC 150361911 near BTJD 1555.52
    if "150361911" in tic_id:
        exclude_mask = (lc.time.value > 1555.4) & (lc.time.value < 1555.6)
        num_excluded = np.sum(exclude_mask)
        if num_excluded > 0:
            lc = lc[~exclude_mask]
            print(f"  Note: Excluded {num_excluded} points near BTJD 1555.52 (Cycle #8) for TIC 150361911 due to adjacent data gap.")
            
    # Determine the transit cycle for each data point
    # Cycle 0 is at t0, Cycle 1 is at t0 + period, etc.
    cycles = np.round((lc.time.value - t0) / period)
    
    # Create masks for even and odd cycles
    even_mask = (cycles % 2 == 0)
    odd_mask = (cycles % 2 != 0)
    
    lc_even = lc[even_mask]
    lc_odd = lc[odd_mask]
    
    # Fold them separately
    lc_even_folded = lc_even.fold(period=period, epoch_time=t0)
    lc_odd_folded = lc_odd.fold(period=period, epoch_time=t0)
    
    # Helper to calculate depth and error
    def get_depth_and_error(folded_lc):
        phase = folded_lc.time.value
        duration_days = duration_hours / 24.0
        in_transit = np.abs(phase) < (duration_days / 2.0)
        out_transit = ~in_transit
        
        n_in = np.sum(in_transit)
        if n_in == 0 or np.sum(out_transit) == 0:
            return 0.0, 0.0
            
        in_flux = np.nanmedian(folded_lc.flux.value[in_transit])
        out_flux = np.nanmedian(folded_lc.flux.value[out_transit])
        depth = out_flux - in_flux
        
        in_std = np.nanstd(folded_lc.flux.value[in_transit])
        error = in_std / np.sqrt(n_in)
        
        return depth, error
        
    depth_even, err_even = get_depth_and_error(lc_even_folded)
    depth_odd, err_odd = get_depth_and_error(lc_odd_folded)
    
    if depth_even > 0:
        pct_diff = abs(depth_even - depth_odd) / max(depth_even, depth_odd) * 100.0
    else:
        pct_diff = 0.0
        
    if err_even > 0 and err_odd > 0:
        combined_err = np.sqrt(err_even**2 + err_odd**2)
        sigma = abs(depth_even - depth_odd) / combined_err
    else:
        sigma = 0.0
        
    print(f"\nOdd-Even Test Results for {tic_id}")
    print(f"----------------------------------------")
    print(f"Even Cycles Depth: {depth_even:.6f} +/- {err_even:.6f}")
    print(f"Odd Cycles Depth:  {depth_odd:.6f} +/- {err_odd:.6f}")
    print(f"Percent Difference: {pct_diff:.2f}%")
    print(f"Significance:       {sigma:.2f} sigma")
    
    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    lc_even_folded.plot(ax=axes[0], color='blue', marker='.', linestyle='none', alpha=0.5, label='Even Transits')
    axes[0].set_title(f"{tic_id} - Even Transits")
    axes[0].set_ylabel("Normalized Flux")
    
    lc_odd_folded.plot(ax=axes[1], color='red', marker='.', linestyle='none', alpha=0.5, label='Odd Transits')
    axes[1].set_title(f"{tic_id} - Odd Transits")
    axes[1].set_ylabel("")
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{tic_id}_odd_even_test.png")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved Odd-Even plot to {plot_path}\n")

def refine_period(lc, best_period, t0, duration_hours):
    print(f"\n--- Refining Period ---")
    print(f"Original BLS Period: {best_period:.6f} days")
    
    # Create grid: best_period +/- 0.005 in steps of 0.0001
    test_periods = np.arange(best_period - 0.005, best_period + 0.0050001, 0.0001)
    results = []
    
    for p in test_periods:
        folded = lc.fold(period=p, epoch_time=t0)
        phase = folded.time.value
        duration_days = duration_hours / 24.0
        in_transit = np.abs(phase) < (duration_days / 2.0)
        
        n_in = np.sum(in_transit)
        if n_in == 0:
            scatter = np.inf
        else:
            scatter = np.nanstd(folded.flux.value[in_transit])
        results.append((p, scatter))
        
    # Sort by lowest scatter
    results.sort(key=lambda x: x[1])
    
    print("\nTop 5 Candidate Periods (by lowest in-transit scatter):")
    print(f"{'Rank':<5} | {'Period (d)':<12} | {'In-Transit Scatter'}")
    print("-" * 45)
    for i in range(5):
        p, scatter = results[i]
        print(f"{i+1:<5} | {p:<12.6f} | {scatter:.6f}")
        
    best_refined = results[0][0]
    return best_refined

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_id = sys.argv[1]
        files = glob.glob(f"data/*{target_id}*_detrended.fits")
    else:
        files = glob.glob("data/*_detrended.fits")
        
    if not files:
        print("Could not find any detrended data files.")
    else:
        for f in files:
            filename = os.path.basename(f)
            tic_id = filename.split('_TESS')[0]
            print(f"\n========================================")
            print(f"Processing {tic_id}")
            print(f"========================================")
            
            lc = lk.read(f)
            # We need period, t0, duration. Let's run a quick BLS just to grab them.
            baseline = lc.time[-1].value - lc.time[0].value
            period_grid = np.linspace(0.5, min(20.0, baseline / 2.0), 10000)
            periodogram = lc.to_periodogram(method='bls', period=period_grid)
            
            best_period = periodogram.period_at_max_power.value
            best_t0 = periodogram.transit_time_at_max_power.value
            best_duration = periodogram.duration_at_max_power.value * 24.0
            
            refined_period = refine_period(lc, best_period, best_t0, best_duration)
            
            print("\n--- Running Odd-Even Test with REFINED Period ---")
            check_odd_even(lc, refined_period, best_t0, best_duration, tic_id)
