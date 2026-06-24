import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

def generate_visual_report(tic_id, result_row, data_dir="data"):
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load light curve
    # Search for the FITS file
    search_pattern = os.path.join(data_dir, f"{tic_id}_*.fits")
    files = glob.glob(search_pattern)
    
    if len(files) == 0:
        print(f"Could not find FITS file for TIC {tic_id}")
        return
        
    file_path = files[0]
    try:
        lc = lk.read(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
        
    period = result_row.get('best_period', 0.0)
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)
    
    # Panel 1: Full Light Curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(lc.time.value, lc.flux.value, s=1, color='black', alpha=0.5)
    ax1.set_title(f"TIC {tic_id} - Full Light Curve", fontsize=14)
    ax1.set_xlabel("Time (BTJD)")
    ax1.set_ylabel("Normalized Flux")
    
    # Panel 2: Folded Light Curve
    ax2 = fig.add_subplot(gs[1, :])
    if period and not pd.isna(period) and period > 0:
        # Recompute t0 for fold since we didn't save it
        bls = lc.to_periodogram('bls', period=[period-0.01, period, period+0.01])
        t0 = bls.transit_time_at_max_power.value
        folded = lc.fold(period=period, epoch_time=t0)
        ax2.scatter(folded.time.value, folded.flux.value, s=2, color='blue', alpha=0.5)
        ax2.set_title(f"Phase-Folded (P = {period:.4f} d)", fontsize=14)
        ax2.set_xlabel("Phase (days)")
        ax2.set_ylabel("Normalized Flux")
    else:
        ax2.text(0.5, 0.5, "No Period Detected", ha='center', va='center', fontsize=14)
        ax2.axis('off')
        
    # Panel 3: Probabilities Bar Chart
    ax3 = fig.add_subplot(gs[2, 0])
    classes = ['Transit', 'Eclipsing Binary', 'Blend', 'Other']
    probs = [
        result_row.get('transit_prob', 0),
        result_row.get('binary_prob', 0),
        result_row.get('blend_prob', 0),
        result_row.get('other_prob', 0)
    ]
    
    colors = ['green', 'orange', 'red', 'gray']
    bars = ax3.bar(classes, probs, color=colors)
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel("Probability")
    ax3.set_title("Classification Probabilities")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f"{height*100:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    # Panel 4: Metadata & Confidence
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    pred_class = result_row.get('predicted_class', 'Unknown')
    confidence = result_row.get('confidence_level', 'LOW')
    
    # Color code confidence
    conf_color = 'green' if confidence == 'HIGH' else ('orange' if confidence == 'MEDIUM' else 'red')
    
    textstr = '\n'.join((
        f"Target: TIC {tic_id}",
        f"Predicted Class: {pred_class}",
        "",
        f"Confidence Level: {confidence}",
        "",
        f"Best Period: {period:.4f} d" if pd.notna(period) else "Best Period: N/A",
        f"SNR: {result_row.get('snr', 0):.1f}" if pd.notna(result_row.get('snr')) else "SNR: N/A",
        f"Symmetry: {result_row.get('transit_symmetry_score', 0):.2f}" if pd.notna(result_row.get('transit_symmetry_score')) else "Symmetry: N/A",
    ))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax4.text(0.1, 0.8, textstr, transform=ax4.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
            
    # Add big colored confidence badge
    ax4.text(0.1, 0.2, f"[{confidence}]", transform=ax4.transAxes, fontsize=24,
            color=conf_color, fontweight='bold', verticalalignment='top')
            
    plt.tight_layout()
    out_path = os.path.join("reports", f"report_TIC_{tic_id}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated {out_path}")

def generate_all_reports():
    print("Loading results_catalog.csv...")
    if not os.path.exists("results_catalog.csv"):
        print("results_catalog.csv not found!")
        return
        
    df = pd.read_csv("results_catalog.csv")
    df_success = df[df['status'] == 'success']
    
    print(f"Generating visual reports for {len(df_success)} targets...")
    for idx, row in df_success.iterrows():
        tic_id = row['tic_id']
        generate_visual_report(tic_id, row)

if __name__ == "__main__":
    generate_all_reports()
