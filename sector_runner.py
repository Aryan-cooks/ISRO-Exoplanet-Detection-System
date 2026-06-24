import os
import glob
import time
import json
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from batch_processor import process_target
import subprocess

RESULTS_FILE = "results_catalog.csv"
SUMMARY_FILE = "processing_summary.json"

def main():
    parser = argparse.ArgumentParser(description="TESS Sector Batch Runner")
    parser.add_argument('--sector', type=int, help="TESS Sector number to process (e.g. 2)")
    parser.add_argument('--download', action='store_true', help="Download the sector data before processing")
    parser.add_argument('--limit', type=int, default=100, help="Target limit if downloading")
    args = parser.parse_args()
    
    if args.sector:
        DATA_DIR = f"data/sector_{args.sector}"
        if args.download:
            print(f"--- Running Download Manager for Sector {args.sector} ---")
            subprocess.run(["python", "sector_ingest.py", "--sector", str(args.sector), "--limit", str(args.limit)])
    else:
        DATA_DIR = "data"
        
    print(f"--- Starting Sector Batch Runner ---")
    start_time = time.time()
    
    # 1. Find all FITS files
    files = glob.glob(os.path.join(DATA_DIR, "*.fits"))
    print(f"Found {len(files)} total FITS files in {DATA_DIR}/")
    
    # 2. Checkpoint / Safe Resume
    processed_tics = set()
    if os.path.exists(RESULTS_FILE):
        try:
            df_existing = pd.read_csv(RESULTS_FILE)
            processed_tics = set(df_existing['tic_id'].astype(str))
            print(f"Loaded existing catalog. Resuming... ({len(processed_tics)} targets already processed)")
        except Exception as e:
            print(f"Could not read existing {RESULTS_FILE}: {e}")
            
    # Filter files that haven't been processed
    files_to_process = []
    for f in files:
        basename = os.path.basename(f)
        tic_id = basename.split('_TESS')[0]
        if tic_id not in processed_tics:
            files_to_process.append(f)
            
    print(f"Files to process in this run: {len(files_to_process)}")
    
    if len(files_to_process) == 0:
        print("No new files to process. Generating summary and exiting.")
        generate_summary(start_time)
        return
        
    # 3. Setup CSV writer for appending
    file_exists = os.path.exists(RESULTS_FILE)
    columns = [
        'tic_id', 'status', 'best_period', 'depth', 'duration', 'snr',
        'predicted_class', 'classification_probability',
        'transit_prob', 'binary_prob', 'blend_prob', 'other_prob', 'confidence_level',
        'blend_probability', 'error'
    ]
    
    if not file_exists:
        pd.DataFrame(columns=columns).to_csv(RESULTS_FILE, index=False)
        
    # 4. Parallel Processing
    # We use ProcessPoolExecutor to utilize all available CPU cores.
    max_workers = os.cpu_count() or 4
    print(f"Launching {max_workers} parallel workers...")
    
    processed_count = 0
    errors_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_target, f): f for f in files_to_process}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                
                # Append to CSV safely
                df_res = pd.DataFrame([result])
                # Only write core columns to avoid schema mismatch if batman adds extra columns occasionally
                for col in columns:
                    if col not in df_res.columns:
                        df_res[col] = None
                
                df_res[columns].to_csv(RESULTS_FILE, mode='a', header=False, index=False)
                
                if result['status'] == 'success':
                    processed_count += 1
                    pred = result.get('predicted_class', 'Unknown')
                    print(f"[{processed_count}/{len(files_to_process)}] Processed {result['tic_id']} -> {pred}")
                else:
                    errors_count += 1
                    print(f"[{processed_count}/{len(files_to_process)}] Error on {result['tic_id']}: {result.get('error')}")
                    
            except Exception as exc:
                errors_count += 1
                print(f"Task generated an exception: {exc}")
                
    print(f"\nBatch processing complete! Processed {processed_count} successfully, {errors_count} errors.")
    
    # 5. Generate Summary
    generate_summary(start_time)

def generate_summary(start_time):
    if not os.path.exists(RESULTS_FILE):
        return
        
    df = pd.read_csv(RESULTS_FILE)
    df_success = df[df['status'] == 'success']
    
    total_processed = len(df)
    total_success = len(df_success)
    
    counts = df_success['predicted_class'].value_counts().to_dict()
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    
    summary = {
        "total_targets_in_catalog": total_processed,
        "successfully_processed": total_success,
        "errors": total_processed - total_success,
        "class_breakdown": {
            "Transit": counts.get('Transit', 0),
            "Eclipsing Binary": counts.get('Eclipsing Binary', 0),
            "Blend": counts.get('Blend', 0),
            "Other": counts.get('Other', 0)
        },
        "processing_time": time_str
    }
    
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Generated {SUMMARY_FILE}")
    
    # Export classification report
    try:
        report_cols = ['tic_id', 'predicted_class', 'transit_prob', 'binary_prob', 'blend_prob', 'other_prob', 'confidence_level']
        df_report = df_success[report_cols]
        df_report.to_csv("classification_report.csv", index=False)
        print("Generated classification_report.csv")
    except KeyError as e:
        print(f"Could not generate classification report: Missing column {e}")

if __name__ == "__main__":
    main()
