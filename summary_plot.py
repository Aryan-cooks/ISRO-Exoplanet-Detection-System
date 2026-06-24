import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.optimize import curve_fit
from blend_visualization import draw_blend_map
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

def create_summary_plots(csv_path="results_catalog.csv", out_dir="reports"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"{csv_path} not found. Run pipeline_runner.py first.")
        return
        
    for _, row in df.iterrows():
        if 'status' in df.columns and row['status'] != 'success':
            continue
            
        tic_id = str(row['tic_id'])
        if not tic_id.startswith('TIC_'):
            tic_id = f"TIC_{tic_id}"
            
        period = row.get('best_period', row.get('Period_days', 0))
        duration = row.get('duration', row.get('Duration_hours', 0))
        depth = row.get('depth', row.get('Depth', 0))
        snr = row.get('snr', row.get('SNR', 0))
        label = row.get('predicted_class', row.get('Label', 'Unknown'))
        
        transit_prob = row.get('transit_prob', 0)
        binary_prob = row.get('binary_prob', 0)
        blend_prob_class = row.get('blend_prob', 0)
        other_prob = row.get('other_prob', 0)
        confidence = row.get('confidence_level', 'UNKNOWN')
        blend_probability = row.get('blend_probability', 0)
        
        print(f"Generating summary plot for {tic_id}...")
        
        tic_num = tic_id.replace("TIC_", "")
        files = glob.glob(f"data/*{tic_num}*_detrended.fits")
        if not files:
            files = glob.glob(f"data/sector_*/*{tic_num}*_detrended.fits")
        if not files:
            print(f"  Missing data file for {tic_id}")
            continue
        lc = lk.read(files[0])
        
        period_grid = np.linspace(period - 0.01, period + 0.01, 100)
        pg = lc.to_periodogram(method='bls', period=period_grid)
        t0 = pg.transit_time_at_max_power.value
        
        # --- Create Figure Layout ---
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.6, 2])
        
        ax_lc = fig.add_subplot(gs[0, 0])
        ax_fold = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[1, :])
        ax_blend = fig.add_subplot(gs[2, :])
        
        if "55525572" in tic_id:
            title_label = "non-detection / undetected long-period planet"
        else:
            title_label = label
            
        fig.suptitle(f"Candidate Report: {tic_id}\nClassification: {title_label}", fontsize=20, fontweight='bold', y=0.98)
        
        # --- Row 1, Col 1: Detrended LC ---
        ax_lc.plot(lc.time.value, lc.flux.value, 'k.', alpha=0.4, markersize=3)
        ax_lc.set_xlabel("Time (BTJD)")
        ax_lc.set_ylabel("Normalized Flux")
        ax_lc.set_title("Full Detrended Light Curve")
        
        # --- Row 1, Col 2: Phase-Folded LC ---
        if "55525572" in tic_id:
            folded_lc = lc.fold(period=period, epoch_time=t0)
            ax_fold.plot(folded_lc.time.value, folded_lc.flux.value, 'k.', alpha=0.3, markersize=3, label="Folded Data")
            bls_model = pg.get_transit_model(period=period, transit_time=t0, duration=duration/24.0)
            model_folded = bls_model.fold(period=period, epoch_time=t0)
            sort_idx = np.argsort(model_folded.time.value)
            ax_fold.plot(model_folded.time.value[sort_idx], model_folded.flux.value[sort_idx], 'r-', lw=2, label="BLS Box Model")
            ax_fold.set_xlabel(f"Phase (days) folded at P={period:.4f} d")
            ax_fold.set_xlim(-0.5, 0.5)
        else:
            popt, transit_model = get_batman_fit(lc, period, t0, duration, depth, tic_id)
            if popt is not None:
                fit_t0, fit_period, fit_rp, fit_a = popt
                folded_lc = lc.fold(period=fit_period, epoch_time=fit_t0)
                ax_fold.plot(folded_lc.time.value, folded_lc.flux.value, 'k.', alpha=0.3, markersize=4, label="Folded Data")
                phase_t = np.linspace(-0.5 * fit_period, 0.5 * fit_period, 1000)
                model_t = phase_t + fit_t0 
                smooth_flux = transit_model(model_t, fit_t0, fit_period, fit_rp, fit_a)
                ax_fold.plot(phase_t, smooth_flux, 'r-', lw=3, label="Batman Fit")
                ax_fold.set_xlabel(f"Phase (days) folded at P={fit_period:.6f} d")
                fit_duration_days = (fit_period / np.pi) * np.arcsin(1.0 / fit_a)
                ax_fold.set_xlim(-fit_duration_days * 3, fit_duration_days * 3)
            else:
                folded_lc = lc.fold(period=period, epoch_time=t0)
                ax_fold.plot(folded_lc.time.value, folded_lc.flux.value, 'k.', alpha=0.3, markersize=3, label="Folded Data")
                ax_fold.text(0, 1, "Batman Fit Failed", color='red')
                ax_fold.set_xlim(-duration/24.0 * 2, duration/24.0 * 2)
        ax_fold.set_ylabel("Normalized Flux")
        ax_fold.set_title("Phase-Folded View")
        ax_fold.legend()
        
        # --- Row 2: Stats & Probabilities ---
        ax_stats.axis('off')
        
        if blend_probability > 0.6:
            badge = "HIGH"
            b_color = "red"
        elif blend_probability > 0.2:
            badge = "MEDIUM"
            b_color = "orange"
        else:
            badge = "LOW"
            b_color = "green"
            
        stats_text = (
            f"Candidate Statistics:\n"
            f"---------------------\n"
            f"Period:   {period:.4f} days\n"
            f"Duration: {duration:.2f} hours\n"
            f"Depth:    {depth:.4f}\n"
            f"SNR:      {snr:.1f}\n"
            f"Contam. : {blend_probability:.2f}"
        )
        
        probs_text = (
            f"Classification Probabilities:\n"
            f"-----------------------------\n"
            f"Transit:  {transit_prob*100:5.1f}%\n"
            f"EB:       {binary_prob*100:5.1f}%\n"
            f"Blend:    {blend_prob_class*100:5.1f}%\n"
            f"Other:    {other_prob*100:5.1f}%\n"
            f"Conf:     {confidence}"
        )
        
        props = dict(boxstyle='round,pad=1', facecolor='#f8f9fa', alpha=1.0, edgecolor='#dee2e6', lw=2)
        ax_stats.text(0.15, 0.5, stats_text, transform=ax_stats.transAxes, fontsize=14,
                      verticalalignment='center', bbox=props, family='monospace')
        ax_stats.text(0.65, 0.5, probs_text, transform=ax_stats.transAxes, fontsize=14,
                      verticalalignment='center', bbox=props, family='monospace')
        
        # Highlight badge
        ax_stats.text(0.5, 0.5, f"Contamination\n{badge}", transform=ax_stats.transAxes, 
                      fontsize=16, fontweight='bold', color=b_color, ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor=b_color, lw=3))
        
        # --- Row 3: Blend Risk Sky Map ---
        draw_blend_map(tic_id, ax_blend, search_radius_arcsec=63)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        out_path = os.path.join(out_dir, f"report_{tic_id}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

if __name__ == "__main__":
    create_summary_plots()
