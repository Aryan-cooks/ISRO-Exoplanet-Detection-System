import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from astroquery.mast import Catalogs
import astropy.units as u

warnings.filterwarnings('ignore', module='astroquery.utils.commons')

def calculate_flux(mag):
    return 10 ** (-0.4 * mag)

def draw_blend_map(tic_str, ax, search_radius_arcsec=63):
    tic_num = int(tic_str.replace("TIC_", ""))
    try:
        target = Catalogs.query_criteria(catalog='Tic', ID=tic_num)
        if len(target) == 0:
            ax.text(0.5, 0.5, f"Target TIC {tic_num} not found.", ha='center')
            return
            
        target_ra = target['ra'][0]
        target_dec = target['dec'][0]
        target_mag = target['Tmag'][0]
        if np.ma.is_masked(target_mag) or np.isnan(target_mag):
            target_mag = 10.0
            
        target_flux = calculate_flux(target_mag)
        
        neighbors = Catalogs.query_region(f'{target_ra} {target_dec}', radius=search_radius_arcsec*u.arcsec, catalog='TIC')
        
        # Plot target
        ax.scatter(target_ra, target_dec, s=300, marker='*', color='red', label=f'Target ({target_mag:.2f} mag)', zorder=10)
        
        # Collect valid neighbors
        valid_neighbors = []
        for row in neighbors:
            if row['ID'] != str(tic_num) and not np.ma.is_masked(row['Tmag']) and not np.isnan(row['Tmag']):
                valid_neighbors.append(row)
                
        if valid_neighbors:
            ras = [row['ra'] for row in valid_neighbors]
            decs = [row['dec'] for row in valid_neighbors]
            mags = [row['Tmag'] for row in valid_neighbors]
            dists = [row['dstArcSec'] for row in valid_neighbors]
            fluxes = [calculate_flux(m) for m in mags]
            
            sizes = [max(10, min(800, 200 * (f / target_flux))) for f in fluxes]
            
            # Color code by pixel ring
            colors = []
            for d in dists:
                if d <= 21:
                    colors.append('crimson') # High risk, 1px
                elif d <= 42:
                    colors.append('orange')  # Med risk, 2px
                elif d <= 63:
                    colors.append('gold')    # Low risk, 3px
                else:
                    colors.append('grey')    # Outside
                    
            scatter = ax.scatter(ras, decs, s=sizes, c=colors, alpha=0.8, edgecolors='k', zorder=5)
            
            # Annotate brightest neighbor
            brightest_idx = np.argmin(mags)
            ax.annotate(f"Brightest\n{mags[brightest_idx]:.1f} mag", 
                        xy=(ras[brightest_idx], decs[brightest_idx]), 
                        xytext=(10, 10), textcoords="offset points", 
                        arrowprops=dict(arrowstyle="->", color='black'),
                        fontsize=9, zorder=15)
                        
        # Draw pixel-scale rings
        cos_dec = np.cos(np.radians(target_dec))
        for r_arcsec, label, ls in [(21, '1 Pixel (21")', '-'), (42, '2 Pixels (42")', '--'), (63, '3 Pixels (63")', ':')]:
            radius_deg = r_arcsec / 3600.0
            circle = plt.Circle((target_ra, target_dec), radius_deg, color='blue', fill=False, linestyle=ls, alpha=0.6, label=label, zorder=2)
            ax.add_patch(circle)
            
        ax.set_aspect(1.0 / cos_dec)
        
        # Explicitly set limits so rings are visible
        max_rad_deg = search_radius_arcsec / 3600.0 * 1.1
        ax.set_xlim(target_ra + max_rad_deg/cos_dec, target_ra - max_rad_deg/cos_dec) # Invert RA
        ax.set_ylim(target_dec - max_rad_deg, target_dec + max_rad_deg)
        
        ax.set_xlabel('Right Ascension (deg)')
        ax.set_ylabel('Declination (deg)')
        ax.set_title(f'TESS Pixel-Scale Blend Map for {tic_str}\nTarget Mag: {target_mag:.2f}')
        
        # Custom legend for rings and neighbor colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='Target Star'),
            Line2D([0], [0], color='blue', lw=1.5, linestyle='-', label='1 Pixel (21")'),
            Line2D([0], [0], color='blue', lw=1.5, linestyle='--', label='2 Pixels (42")'),
            Line2D([0], [0], color='blue', lw=1.5, linestyle=':', label='3 Pixels (63")'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=8, label='Neighbor < 21"'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Neighbor < 42"'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=8, label='Neighbor < 63"')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error generating plot for {tic_num}: {e}", ha='center')

def plot_blend_map(tic_str, search_radius_arcsec=63, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fig, ax = plt.subplots(figsize=(10, 10))
    print(f"Generating sky map for {tic_str}...")
    draw_blend_map(tic_str, ax, search_radius_arcsec)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{tic_str}_blend_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sky map to {output_path}")

def main():
    df = pd.read_csv("feature_table.csv")
    for tic_id in df['TIC_ID'].unique():
        plot_blend_map(tic_id)

if __name__ == "__main__":
    main()
