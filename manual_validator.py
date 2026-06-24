import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from batch_processor import compute_transit_shape_features

def validate_specific_periods(tic_id, periods):
    print(f"Downloading stitched lightcurve for TIC {tic_id}...")
    search_result = lk.search_lightcurve(f"TIC {tic_id}", author='SPOC')
    lc_collection = search_result.download_all()
    lc = lc_collection.stitch().remove_nans()
    _, unique_indices = np.unique(lc.time.value, return_index=True)
    lc = lc[unique_indices].remove_outliers(sigma=5)
    
    fig, axes = plt.subplots(len(periods), 1, figsize=(10, 5 * len(periods)))
    
    for i, period in enumerate(periods):
        # We need an epoch time (t0). To find it, let's run a quick narrow BLS around this period
        print(f"Finding best t0 for period ~{period}...")
        bls = lc.to_periodogram(method='bls', period=np.linspace(period-0.05, period+0.05, 500))
        t0 = bls.transit_time_at_max_power.value
        dur = bls.duration_at_max_power.value * 24.0
        depth = bls.depth_at_max_power
        
        # Calculate shape features
        ing_dur, eg_dur, sym_score, v_score, u_score = compute_transit_shape_features(lc, period, t0, dur)
        
        print(f"\n--- Period: {period} d ---")
        print(f"Depth: {depth:.5f}")
        print(f"Duration: {dur:.2f} hrs")
        print(f"Symmetry Score: {sym_score:.4f}")
        
        # Fold and bin
        folded = lc.fold(period=period, epoch_time=t0)
        binned = folded.bin(time_bin_size=0.01)
        
        ax = axes[i]
        ax.scatter(folded.time.value, folded.flux.value, s=1, color='gray', alpha=0.3, label='Unbinned')
        ax.scatter(binned.time.value, binned.flux.value, s=20, color='red', label='Binned')
        ax.axvline(0, color='blue', linestyle='--', label='Transit Center')
        ax.set_title(f"Period: {period} d | Depth: {depth:.4f} | Symmetry: {sym_score:.2f}")
        ax.set_xlabel("Phase (days)")
        ax.set_ylabel("Normalized Flux")
        ax.legend()
        ax.set_xlim(-0.5, 0.5)
        
    plt.tight_layout()
    plt.savefig("explicit_comparison.png", dpi=300)
    print("\nSaved explicit_comparison.png")

if __name__ == "__main__":
    validate_specific_periods("283722336", [3.0929, 10.1675])
