import os
import sys
import subprocess
import argparse
import time

def run_step(name, command):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    start_time = time.time()
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed at step: {name}")
        print(f"Command returned non-zero exit status {e.returncode}")
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"[{name}] completed in {elapsed:.2f} seconds.\n")

def main():
    parser = argparse.ArgumentParser(description="Exoplanet Detection Pipeline Runner")
    parser.add_argument('--sector', type=int, default=2, help="TESS Sector to download and process")
    parser.add_argument('--limit', type=int, default=5, help="Limit number of targets for quick testing (0 for all)")
    args = parser.parse_args()
    
    print("Starting Exoplanet Detection Pipeline...")
    total_start_time = time.time()
    
    # 1. Download and Ingest Data
    cmd_download = [sys.executable, "sector_ingest.py", "--sector", str(args.sector)]
    if args.limit > 0:
        cmd_download.extend(["--limit", str(args.limit)])
    run_step("1. Download/load light curves", cmd_download)
    
    # Optional: Multisector stitching can be run here if specified, 
    # but by default we process the downloaded single sector batch.
    print("\n[INFO] Step 2. Stitch sectors: Handled internally per target if multi-sector data is present.")
    
    # 3-6. Sector Processing (BLS, Features, Blend, Classification, Export)
    # sector_runner.py handles running batch_processor on all targets
    cmd_process = [sys.executable, "sector_runner.py", "--sector", str(args.sector)]
    run_step("3-6. Run BLS, Generate Features, Blend Analysis, and Classification", cmd_process)
    
    # 7. Generate offline feature table (Feature Audit consistency)
    cmd_features = [sys.executable, "generate_feature_table.py"]
    run_step("7. Export comprehensive feature_table.csv", cmd_features)
    
    # 8. Train Classifier (Updates metrics.json with dataset detection)
    cmd_train = [sys.executable, "train_classifier.py"]
    run_step("8. Model Training & Dataset Validation", cmd_train)
    
    # 9. Generate Reports & Visualizations
    cmd_reports = [sys.executable, "summary_plot.py"]
    run_step("9. Generate Reports & Export Final Results", cmd_reports)
    
    total_elapsed = time.time() - total_start_time
    print(f"\n{'*'*60}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS")
    print("Check 'reports/' for candidate summaries and 'classification_report.csv' for details.")
    print(f"{'*'*60}\n")

if __name__ == "__main__":
    main()
