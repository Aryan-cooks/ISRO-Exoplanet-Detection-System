import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit
try:
    import batman
except ImportError:
    print("batman-package not installed.")

def get_batman_fit(lc, period, bls_t0, duration_hours, depth, tic_id):
    # Same data exclusion and EB filtering as fit_batman.py
    if "38699825" in tic_id:
        exclude_mask = (lc.time.value > 1367.0) & (lc.time.value < 1367.3)
        lc = lc[~exclude_mask]
    if "150361911" in tic_id:
        exclude_mask = (lc.time.value > 1555.4) & (lc.time.value < 1555.6)
        lc = lc[~exclude_mask]

    if tic_id in ["TIC_38699825", "TIC_150361911"]:
        cycles = np.round((lc.time.value - bls_t0) / period)
        even_mask = (cycles % 2 == 0)
        odd_mask = (cycles % 2 != 0)
        duration_days = duration_hours / 24.0
        phase = (lc.time.value - bls_t0 + 0.5 * period) % period - 0.5 * period
        in_transit = np.abs(phase) < (duration_days / 2.0)
        
        even_flux = lc.flux.value[even_mask & in_transit]
        odd_flux = lc.flux.value[odd_mask & in_transit]
        even_depth = 1.0 - np.nanmean(even_flux) if len(even_flux) > 0 else 0
        odd_depth = 1.0 - np.nanmean(odd_flux) if len(odd_flux) > 0 else 0
        
        if even_depth > odd_depth:
            lc = lc[~(odd_mask & in_transit)]
        else:
            lc = lc[~(even_mask & in_transit)]

    rp_rs_init = np.sqrt(depth) if depth > 0 else 0.1
    duration_days = duration_hours / 24.0
    a_init = period / (np.pi * duration_days)
    if a_init < 1.0: 
        a_init = 10.0

    def transit_model(t, t0, p, rp, a):
        params = batman.TransitParams()
        params.t0 = t0
        params.per = p
        params.rp = rp
        params.a = a
        params.inc = 90.0
        params.ecc = 0.0
        params.w = 90.0
        params.u = [0.3, 0.2]
        params.limb_dark = "quadratic"
        m = batman.TransitModel(params, t)
        return m.light_curve(params)

    p0 = [bls_t0, period, rp_rs_init, a_init]
    bounds = (
        [bls_t0 - 0.5, period - 0.5, 0.0001, 1.0],
        [bls_t0 + 0.5, period + 0.5, 0.9999, 500.0]
    )

    t = lc.time.value
    flux = lc.flux.value

    try:
        popt, _ = curve_fit(transit_model, t, flux, p0=p0, bounds=bounds)
        return popt, transit_model
    except:
        return None, None

def create_summary_plots(csv_path="feature_table.csv", out_dir="summary_plots"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        tic_id = row['TIC_ID']
        period = row['Period_days']
        duration = row['Duration_hours']
        depth = row['Depth']
        snr = row['SNR']
        label = row['Label']
        
        print(f"Generating summary plot for {tic_id}...")
        
        # Load detrended light curve
        tic_num = tic_id.replace("TIC_", "")
        files = glob.glob(f"data/*{tic_num}*_detrended.fits")
        if not files:
            print(f"  Missing data file for {tic_id}")
            continue
        lc = lk.read(files[0])
        
        # Extract initial t0 accurately using a local BLS grid around the established period
        period_grid = np.linspace(period - 0.01, period + 0.01, 100)
        pg = lc.to_periodogram(method='bls', period=period_grid)
        t0 = pg.transit_time_at_max_power.value
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Format the title based on constraints
        if "55525572" in tic_id:
            title_label = "non-detection / undetected long-period planet"
        else:
            title_label = label
            
        fig.suptitle(f"Target: {tic_id} | Classification: {title_label} | SNR: {snr:.1f}", fontsize=16, fontweight='bold')
        
        # --- Top Panel: Raw (Detrended) Light Curve ---
        ax1.plot(lc.time.value, lc.flux.value, 'k.', alpha=0.4, markersize=3)
        ax1.set_xlabel("Time (BTJD)")
        ax1.set_ylabel("Normalized Flux")
        ax1.set_title("Full Detrended Light Curve")
        
        # --- Bottom Panel: Phase-Folded Light Curve & Model ---
        if "55525572" in tic_id:
            # BLS Box Model for non-detection
            folded_lc = lc.fold(period=period, epoch_time=t0)
            ax2.plot(folded_lc.time.value, folded_lc.flux.value, 'k.', alpha=0.3, markersize=3, label="Folded Data")
            
            bls_model = pg.get_transit_model(period=period, transit_time=t0, duration=duration/24.0)
            model_folded = bls_model.fold(period=period, epoch_time=t0)
            sort_idx = np.argsort(model_folded.time.value)
            ax2.plot(model_folded.time.value[sort_idx], model_folded.flux.value[sort_idx], 'r-', lw=2, label="BLS Box Model")
            
            ax2.set_xlabel(f"Phase (days) folded at P={period:.4f} d")
            ax2.set_xlim(-0.5, 0.5) # Zoom slightly
        else:
            # Batman Fit Model for real transits/eclipses
            popt, transit_model = get_batman_fit(lc, period, t0, duration, depth, tic_id)
            if popt is not None:
                fit_t0, fit_period, fit_rp, fit_a = popt
                
                # Fold the true data
                folded_lc = lc.fold(period=fit_period, epoch_time=fit_t0)
                ax2.plot(folded_lc.time.value, folded_lc.flux.value, 'k.', alpha=0.3, markersize=4, label="Folded Data")
                
                # Generate and plot model
                phase_t = np.linspace(-0.5 * fit_period, 0.5 * fit_period, 1000)
                model_t = phase_t + fit_t0 
                smooth_flux = transit_model(model_t, fit_t0, fit_period, fit_rp, fit_a)
                ax2.plot(phase_t, smooth_flux, 'r-', lw=3, label="Batman Fit")
                
                ax2.set_xlabel(f"Phase (days) folded at P={fit_period:.6f} d")
                
                # Zoom in on the transit duration
                fit_duration_days = (fit_period / np.pi) * np.arcsin(1.0 / fit_a)
                ax2.set_xlim(-fit_duration_days * 3, fit_duration_days * 3)
            else:
                # Fallback if fit fails
                folded_lc = lc.fold(period=period, epoch_time=t0)
                ax2.plot(folded_lc.time.value, folded_lc.flux.value, 'k.', alpha=0.3, markersize=3, label="Folded Data")
                ax2.text(0, 1, "Batman Fit Failed", color='red')
                ax2.set_xlim(-duration/24.0 * 2, duration/24.0 * 2)
                
        ax2.set_ylabel("Normalized Flux")
        ax2.set_title("Phase-Folded View")
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(out_dir, f"{tic_id}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

if __name__ == "__main__":
    create_summary_plots()
