import argparse
import time
import json
import os
from download_manager import discover_targets, download_lightcurves

def main():
    parser = argparse.ArgumentParser(description="TESS Sector Ingestion Workflow")
    parser.add_argument('--sector', type=int, required=True, help="TESS Sector number to ingest")
    parser.add_argument('--limit', type=int, default=100, help="Maximum number of targets to download")
    args = parser.parse_args()
    
    sector = args.sector
    limit = args.limit
    output_dir = f"data/sector_{sector}"
    
    print(f"=== Starting Ingestion for Sector {sector} ===")
    start_time = time.time()
    
    # 1. Discover Targets
    tic_ids = discover_targets(sector, limit=limit)
    
    if not tic_ids:
        print("No targets found or MAST query failed.")
        return
        
    # 2. Download Light Curves
    success, failed = download_lightcurves(sector, tic_ids, output_dir)
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    
    # 3. Produce Report
    report = {
        "sector": sector,
        "target_limit_requested": limit,
        "targets_discovered": len(tic_ids),
        "successful_downloads": success,
        "failed_downloads": failed,
        "output_directory": output_dir,
        "total_download_time": time_str
    }
    
    report_file = "sector_download_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"\n=== Ingestion Complete ===")
    print(f"Success: {success} | Failed: {failed}")
    print(f"Time elapsed: {time_str}")
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()
