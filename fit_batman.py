import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit

# Important: We must import batman inside a try/except or normally. 
# We'll assume the user has it installed when they run the script.
try:
    import batman
except ImportError:
    print("The 'batman-package' is not installed. Please run: pip install batman-package")
    sys.exit(1)

def fit_batman_transit(lc, bls_period, bls_t0, bls_duration_hours, bls_depth, tic_id, plot_dir="plots"):
    print(f"\n{'-'*50}")
    print(f"Starting Batman Transit Fit for {tic_id}")
    print(f"{'-'*50}")
    
    # ---------------------------------------------------------
    # Target-Specific Data Exclusions
    # ---------------------------------------------------------
    # Exclude cycle #13 for TIC 38699825
    if "38699825" in tic_id:
        exclude_mask = (lc.time.value > 1367.0) & (lc.time.value < 1367.3)
        num_excluded = np.sum(exclude_mask)
        if num_excluded > 0:
            lc = lc[~exclude_mask]
            print(f"  Note: Excluded {num_excluded} points near BTJD 1367.15 (Cycle #13) for TIC 38699825 due to adjacent data gap.")
            
    # Exclude cycle #8 for TIC 150361911
    if "150361911" in tic_id:
        exclude_mask = (lc.time.value > 1555.4) & (lc.time.value < 1555.6)
        num_excluded = np.sum(exclude_mask)
        if num_excluded > 0:
            lc = lc[~exclude_mask]
            print(f"  Note: Excluded {num_excluded} points near BTJD 1555.52 (Cycle #8) for TIC 150361911 due to adjacent data gap.")

    # ---------------------------------------------------------
    # Eclipsing Binary Simplification
    # ---------------------------------------------------------
    if tic_id in ["TIC_38699825", "TIC_150361911"]:
        print("  Note: Target is an Eclipsing Binary. Batman is built for single-dip transits.")
        print("  Simplification: Filtering out the shallower secondary eclipses to fit ONLY the primary eclipse shape.")
        
        unfiltered_pts = len(lc.flux.value)
        unfiltered_min_flux = np.nanmin(lc.flux.value)
        
        cycles = np.round((lc.time.value - bls_t0) / bls_period)
        even_mask = (cycles % 2 == 0)
        odd_mask = (cycles % 2 != 0)
        
        duration_days = bls_duration_hours / 24.0
        phase = (lc.time.value - bls_t0 + 0.5 * bls_period) % bls_period - 0.5 * bls_period
        in_transit = np.abs(phase) < (duration_days / 2.0)
        
        # Using a simple mean of the lowest points as a robust proxy for depth
        even_flux = lc.flux.value[even_mask & in_transit]
        odd_flux = lc.flux.value[odd_mask & in_transit]
        
        even_depth = 1.0 - np.nanmean(even_flux) if len(even_flux) > 0 else 0
        odd_depth = 1.0 - np.nanmean(odd_flux) if len(odd_flux) > 0 else 0
        
        if even_depth > odd_depth:
            odd_transit_mask = odd_mask & in_transit
            lc = lc[~odd_transit_mask]
        else:
            even_transit_mask = even_mask & in_transit
            lc = lc[~even_transit_mask]
            
        filtered_pts = len(lc.flux.value)
        filtered_min_flux = np.nanmin(lc.flux.value)
        
        print("\n  --- Eclipsing Binary Filtering Diagnostics ---")
        print(f"  Data points before filtering: {unfiltered_pts}")
        print(f"  Data points after filtering:  {filtered_pts}")
        print(f"  Unfiltered minimum flux:      {unfiltered_min_flux:.6f}")
        print(f"  Filtered minimum flux:        {filtered_min_flux:.6f}")

    # ---------------------------------------------------------
    # 1. Parameterization & Initial Guesses
    # ---------------------------------------------------------
    rp_rs_init = np.sqrt(bls_depth) if bls_depth > 0 else 0.1
    bls_duration_days = bls_duration_hours / 24.0
    a_init = bls_period / (np.pi * bls_duration_days)
    if a_init < 1.0:
        a_init = 10.0 # Safety fallback

    print("\nInitial Guesses (from BLS):")
    print(f"  t0:     {bls_t0:.6f}")
    print(f"  Period: {bls_period:.6f} days")
    print(f"  Rp/Rs:  {rp_rs_init:.6f}")
    print(f"  a/Rs:   {a_init:.6f}")

    # ---------------------------------------------------------
    # 2. Define the curve_fit wrapper
    # ---------------------------------------------------------
    def transit_model(t, t0, period, rp, a):
        params = batman.TransitParams()
        params.t0 = t0
        params.per = period
        params.rp = rp
        params.a = a
        params.inc = 90.0 # Fixed edge-on to avoid degeneracy
        params.ecc = 0.0  # Assumed circular orbit
        params.w = 90.0   # Doesn't matter for circular orbit
        params.u = [0.3, 0.2] 
        params.limb_dark = "quadratic"
        
        m = batman.TransitModel(params, t)
        return m.light_curve(params)

    # ---------------------------------------------------------
    # 3. Fit via Scipy
    # ---------------------------------------------------------
    p0 = [bls_t0, bls_period, rp_rs_init, a_init]
    
    bounds = (
        [bls_t0 - 0.5, bls_period - 0.5, 0.0001, 1.0],
        [bls_t0 + 0.5, bls_period + 0.5, 0.9999, 500.0]
    )

    t = lc.time.value
    flux = lc.flux.value

    try:
        popt, pcov = curve_fit(transit_model, t, flux, p0=p0, bounds=bounds)
        print("\nFit Converged Successfully!")
    except RuntimeError as e:
        print("\nFIT FAILED TO CONVERGE!")
        print(f"Error returned: {e}")
        print("\nWhat does this mean?")
        print("Scipy's `curve_fit` could not find a set of parameters that minimized the difference between the batman model and your data.")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return

    # ---------------------------------------------------------
    # 4. Extract & Print Results
    # ---------------------------------------------------------
    fit_t0, fit_period, fit_rp, fit_a = popt
    err_t0, err_period, err_rp, err_a = np.sqrt(np.diag(pcov))

    fit_depth = fit_rp**2
    fit_duration_days = (fit_period / np.pi) * np.arcsin(1.0 / fit_a)
    fit_duration_hours = fit_duration_days * 24.0

    print("\nFitted Parameters (with 1-sigma uncertainty):")
    print(f"  t0     = {fit_t0:.6f} +/- {err_t0:.6f}")
    print(f"  Period = {fit_period:.6f} +/- {err_period:.6f} days")
    print(f"  Rp/Rs  = {fit_rp:.6f} +/- {err_rp:.6f}")
    print(f"  a/Rs   = {fit_a:.6f} +/- {err_a:.6f}")

    print("\n--- Comparison: BLS vs. Batman Fit ---")
    print(f"{'Parameter':<10} | {'BLS Stage 3':<15} | {'Batman Fit':<15}")
    print("-" * 45)
    print(f"{'Depth':<10} | {bls_depth:<15.6f} | {fit_depth:<15.6f}")
    print(f"{'Duration':<10} | {bls_duration_hours:<15.4f} | {fit_duration_hours:<15.4f} hours")

    # ---------------------------------------------------------
    # 5. Plotting
    # ---------------------------------------------------------
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    folded_lc = lc.fold(period=fit_period, epoch_time=fit_t0)
    phase = folded_lc.time.value
    
    phase_t = np.linspace(-0.5 * fit_period, 0.5 * fit_period, 1000)
    model_t = phase_t + fit_t0 
    smooth_flux = transit_model(model_t, fit_t0, fit_period, fit_rp, fit_a)

    fig, ax = plt.subplots(figsize=(10, 6))
    folded_lc.plot(ax=ax, color='black', marker='.', linestyle='none', alpha=0.3, label='Detrended Data')
    ax.plot(phase_t, smooth_flux, color='red', linewidth=3, label='Batman Model Fit')
    
    ax.set_title(f"Batman Transit Model Fit - {tic_id}")
    ax.set_xlabel("Phase (days)")
    ax.set_ylabel("Normalized Flux")
    ax.legend()
    
    ax.set_xlim(-fit_duration_days * 2, fit_duration_days * 2)
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{tic_id}_batman_fit.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"\nSaved Batman fit plot to {plot_path}")

if __name__ == "__main__":
    targets_info = {
        "283722336": 3.0934,
        "38699825": 1.0427,
        "150361911": 1.2464
    }
    
    for tic_id_num, fixed_period in targets_info.items():
        tic_id = f"TIC_{tic_id_num}"
        files = glob.glob(f"data/*{tic_id_num}*_detrended.fits")
        if not files:
            print(f"Could not find detrended data for {tic_id}")
            continue
            
        lc = lk.read(files[0])
        
        # Extract initial t0, duration, depth by doing a localized BLS around the exact given period
        period_grid = np.linspace(fixed_period - 0.01, fixed_period + 0.01, 100)
        periodogram = lc.to_periodogram(method='bls', period=period_grid)
        
        bls_period = fixed_period  # Exact requested value
        bls_t0 = periodogram.transit_time_at_max_power.value
        bls_duration = periodogram.duration_at_max_power.value * 24.0
        bls_depth = periodogram.depth_at_max_power
        
        fit_batman_transit(lc, bls_period, bls_t0, bls_duration, bls_depth, tic_id)
