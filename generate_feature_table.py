import os
import glob
import pandas as pd
from batch_processor import process_target

def generate_feature_table(data_dir="data", output_csv="feature_table.csv"):
    # Find all detrended FITS files across all sector directories
    files = []
    # Check both data/ and data/sector_*/
    files.extend(glob.glob(os.path.join(data_dir, "*_detrended.fits")))
    files.extend(glob.glob(os.path.join(data_dir, "sector_*", "*_detrended.fits")))
    
    print(f"Found {len(files)} detrended FITS files for feature extraction.")
    
    all_features = []
    
    for file_path in files:
        print(f"Processing {file_path}...")
        result = process_target(file_path)
        
        if result['status'] == 'success':
            feat_dict = {
                'TIC_ID': result['tic_id'],
                'Period_days': result['best_period'],
                'Duration_hours': result['duration'],
                'Depth': result['depth'],
                'SNR': result['snr'],
                'Odd_Even_Sigma': result.get('odd_even_sigma', 0),
                'blend_probability': result.get('blend_probability', 0),
                'neighbor_count': result.get('neighbor_count', 0),
                'bls_peak_power': result.get('bls_peak_power', 0),
                'transit_symmetry_score': result.get('transit_symmetry_score', 0),
                'v_shape_score': result.get('v_shape_score', 0),
                'u_shape_score': result.get('u_shape_score', 0),
                'ingress_duration': result.get('ingress_duration', 0),
                'egress_duration': result.get('egress_duration', 0),
                'num_observed_transits': result.get('num_observed_transits', 0),
                'Label': result.get('predicted_class', 'Unknown')
            }
            all_features.append(feat_dict)
        else:
            print(f"Skipping {file_path} due to error: {result.get('error')}")
            
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_csv, index=False)
        print(f"Successfully generated {output_csv} with {len(all_features)} rows and {len(df.columns)} columns.")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    generate_feature_table()
