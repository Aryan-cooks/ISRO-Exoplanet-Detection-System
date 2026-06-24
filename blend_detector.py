import pandas as pd
import numpy as np
import warnings
from astroquery.mast import Catalogs
import astropy.units as u

warnings.filterwarnings('ignore', module='astroquery.utils.commons')

def calculate_flux(mag):
    return 10 ** (-0.4 * mag)

def calculate_gaussian_weight(distance_arcsec, sigma=15.0):
    """Calculate distance weight using a Gaussian profile approximating TESS PSF."""
    return np.exp(-(distance_arcsec**2) / (2 * sigma**2))

def detect_blends(input_csv="feature_table.csv", output_csv="blend_features.csv", search_radius_arcsec=60):
    df = pd.read_csv(input_csv)
    tic_ids = df['TIC_ID'].unique()
    
    results = []
    
    print(f"Starting blend detection for {len(tic_ids)} targets...")
    
    for tic_str in tic_ids:
        tic_num = int(tic_str.replace("TIC_", ""))
        print(f"Querying MAST for TIC {tic_num}...")
        
        try:
            target = Catalogs.query_criteria(catalog='Tic', ID=tic_num)
            if len(target) == 0:
                print(f"Warning: Target TIC {tic_num} not found in MAST.")
                continue
                
            target_ra = target['ra'][0]
            target_dec = target['dec'][0]
            target_mag = target['Tmag'][0]
            
            if np.ma.is_masked(target_mag) or np.isnan(target_mag):
                target_mag = 10.0
                
            target_flux = calculate_flux(target_mag)
            
            # Query neighbors
            neighbors = Catalogs.query_region(f'{target_ra} {target_dec}', radius=search_radius_arcsec*u.arcsec, catalog='TIC')
            neighbors = neighbors[neighbors['ID'] != str(tic_num)]
            
            valid_neighbors = []
            for row in neighbors:
                if not np.ma.is_masked(row['Tmag']) and not np.isnan(row['Tmag']):
                    valid_neighbors.append(row)
                    
            neighbor_count = len(valid_neighbors)
            
            # Pixel-scale occupancy metrics (TESS is ~21 arcsec/pixel)
            neighbors_within_1px = sum(1 for r in valid_neighbors if r['dstArcSec'] <= 21)
            neighbors_within_2px = sum(1 for r in valid_neighbors if r['dstArcSec'] <= 42)
            neighbors_within_3px = sum(1 for r in valid_neighbors if r['dstArcSec'] <= 63)
            
            if neighbor_count > 0:
                mags = [row['Tmag'] for row in valid_neighbors]
                dists = [row['dstArcSec'] for row in valid_neighbors]
                
                brightest_mag = min(mags)
                closest_arcsec = min(dists)
                
                # V1 Contamination
                total_neighbor_flux = sum(calculate_flux(m) for m in mags)
                contamination_ratio = total_neighbor_flux / target_flux
                
                # V2 Distance-Weighted Contamination
                weighted_fluxes = [calculate_flux(m) * calculate_gaussian_weight(d) for m, d in zip(mags, dists)]
                weighted_contamination_ratio = sum(weighted_fluxes) / target_flux
                
                # Magnitude Difference: target_mag - brightest_neighbor_mag
                # Positive delta_mag means neighbor is fainter? No!
                # If target is 10 and neighbor is 8 (brighter), delta is 10 - 8 = +2.
                # So positive delta_mag means neighbor is brighter than target -> Dangerous!
                brightest_neighbor_delta_mag = target_mag - brightest_mag
            else:
                brightest_mag = np.nan
                closest_arcsec = np.nan
                contamination_ratio = 0.0
                weighted_contamination_ratio = 0.0
                brightest_neighbor_delta_mag = np.nan
                
            # V2 Enhanced Blend Probability Model
            # Base probability derived from the weighted contamination
            base_prob = 1.0 - np.exp(-20 * weighted_contamination_ratio)
            
            # Penalize highly if there are stars within 1 pixel
            if neighbors_within_1px > 0:
                base_prob += 0.15 * neighbors_within_1px
                
            # Penalize if the brightest neighbor is brighter than the target
            if not np.isnan(brightest_neighbor_delta_mag) and brightest_neighbor_delta_mag > 0:
                base_prob += 0.2
                
            blend_prob = min(1.0, max(0.0, base_prob))
            
            # Continuous risk class for backward compatibility
            if blend_prob < 0.3:
                blend_flag = "LOW_BLEND_RISK"
            elif blend_prob < 0.7:
                blend_flag = "MEDIUM_BLEND_RISK"
            else:
                blend_flag = "HIGH_BLEND_RISK"
                
            results.append({
                "TIC_ID": tic_str,
                "neighbor_count": neighbor_count,
                "neighbors_within_1px": neighbors_within_1px,
                "neighbors_within_2px": neighbors_within_2px,
                "neighbors_within_3px": neighbors_within_3px,
                "brightest_neighbor_mag": brightest_mag,
                "closest_neighbor_arcsec": closest_arcsec,
                "brightest_neighbor_delta_mag": brightest_neighbor_delta_mag,
                "contamination_ratio": contamination_ratio, # original
                "weighted_contamination_ratio": weighted_contamination_ratio, # v2 new
                "blend_probability": blend_prob,
                "blend_flag": blend_flag
            })
            
        except Exception as e:
            print(f"Error querying TIC {tic_num}: {e}")
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved blend features to {output_csv}")
    print(results_df[['TIC_ID', 'weighted_contamination_ratio', 'blend_probability', 'blend_flag']])

if __name__ == "__main__":
    detect_blends()
