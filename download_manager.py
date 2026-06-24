import os
import glob
import time
from astroquery.mast import Observations

def discover_targets(sector, limit=100):
    print(f"Querying MAST for TESS targets in Sector {sector}...")
    obs = Observations.query_criteria(
        obs_collection='TESS', 
        sequence_number=sector, 
        dataproduct_type='timeseries', 
        project='TESS'
    )
    
    targets = list(set(obs['target_name']))
    valid_targets = [t for t in targets if t.isdigit()]
    print(f"Discovered {len(valid_targets)} targets in Sector {sector}. Limiting to {limit}.")
    return valid_targets[:limit]

def download_lightcurves(sector, tic_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    success = 0
    failed = 0
    
    print(f"Starting download of {len(tic_ids)} targets to {output_dir}/")
    
    for i, tic in enumerate(tic_ids):
        # We enforce the TIC_{tic}_TESS_Sector_{sector}.fits naming convention
        expected_filename = f"TIC_{tic}_TESS_Sector_{sector:02d}.fits"
        expected_path = os.path.join(output_dir, expected_filename)
        
        if os.path.exists(expected_path):
            success += 1
            if i % 10 == 0:
                print(f"[{i+1}/{len(tic_ids)}] {expected_filename} already exists. Skipping.")
            continue
            
        print(f"[{i+1}/{len(tic_ids)}] Downloading TIC {tic}...")
        try:
            obs = Observations.query_criteria(
                obs_collection='TESS', 
                sequence_number=sector, 
                target_name=tic, 
                dataproduct_type='timeseries'
            )
            if len(obs) == 0:
                failed += 1
                continue
                
            dp = Observations.get_product_list(obs)
            lc_dp = Observations.filter_products(dp, productSubGroupDescription="LC")
            
            if len(lc_dp) > 0:
                # Download exactly the first LC product
                manifest = Observations.download_products(lc_dp[0:1], download_dir=output_dir, flat=True)
                
                if manifest is not None and len(manifest) > 0:
                    downloaded_file = manifest['Local Path'][0]
                    # Rename to our convention
                    os.rename(downloaded_file, expected_path)
                    success += 1
                else:
                    failed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"Error downloading TIC {tic}: {e}")
            failed += 1
            
    return success, failed
