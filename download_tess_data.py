import os
import argparse
import lightkurve as lk

# Hardcoded list of TIC IDs (mix of TOIs, eclipsing binaries, false positives)
TIC_IDS = [
    "TIC 283722336",  # HD 219134 b (TOI-1469.01)
    "TIC 55525572",   # TOI-813 b
    "TIC 261136679",  # Pi Mensae / TOI-144 — confirmed planet system
    "TIC 164726147",  # Example Target
    "TIC 100100827",  # Wasp-18b (HD 219134 b)
    "TIC 38699825",   # Eclipsing Binary
    "TIC 150361911"   # Eclipsing Binary
]

def download_and_save_lightcurve(tic_id, sector=None, output_dir="data"):
    """
    Downloads PDCSAP flux light curve for a given TIC ID and Sector,
    prints a summary, and saves the data to FITS and CSV formats.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nSearching for {tic_id}...")
    
    # Search for SPOC pipeline light curves (which contain PDCSAP flux)
    search_result = lk.search_lightcurve(tic_id, author="SPOC")
    
    if len(search_result) == 0:
        print(f"No SPOC light curves found for {tic_id}.")
        return

    # Filter by sector if provided
    if sector is not None:
        # Usually lightkurve mission strings are like "TESS Sector 01" or "TESS Sector 1"
        # To be safe, we just check if the sector number is in the mission string
        filtered_indices = [i for i, res in enumerate(search_result) if f"Sector {sector:02d}" in res.mission[0] or f"Sector {sector}" in res.mission[0]]
        
        if len(filtered_indices) == 0:
            available_sectors = [res.mission[0] for res in search_result]
            print(f"Sector {sector} not found for {tic_id}. Available: {available_sectors}")
            return
        
        # Slice the search result to only include the requested sector
        search_result = search_result[filtered_indices[0]]

    # If multiple sectors exist and no specific sector was requested, let the user choose
    if len(search_result) > 1:
        print(f"Multiple sectors found for {tic_id}:")
        for i in range(len(search_result)):
            row = search_result.table[i]
            mission = row['mission']
            author = row['author']
            exptime = row['exptime']
            print(f"  [{i}] Mission: {mission} | Author: {author} | Exptime: {exptime} s")
        
        while True:
            choice = input(f"Enter the index to download for {tic_id} (or 'skip' to skip): ")
            if choice.lower() == 'skip':
                return
            try:
                selected_idx = int(choice)
                if 0 <= selected_idx < len(search_result):
                    search_result = search_result[selected_idx]
                    break
                else:
                    print("Index out of range. Try again.")
            except ValueError:
                print("Invalid input. Please enter an integer or 'skip'.")
    elif len(search_result) == 1:
        search_result = search_result[0]

    # Download the light curve
    # search_result is now a single item SearchResult
    mission_val = search_result.mission
    mission_name = str(mission_val[0]) if hasattr(mission_val, '__len__') and not isinstance(mission_val, str) else str(mission_val)
    print(f"Downloading {mission_name} for {tic_id}...")
    lc = search_result.download()

    if lc is None:
        print(f"Failed to download light curve for {tic_id}.")
        return

    # Handle missing/NaN values in PDCSAP_FLUX
    # The remove_nans() method drops any cadence where the flux is NaN
    initial_points = len(lc)
    
    lc_clean = lc.remove_nans()
    final_points = len(lc_clean)
    num_nans = initial_points - final_points
    
    # Calculate time span in days
    time_span = lc_clean.time[-1].value - lc_clean.time[0].value

    # Print summary
    print(f"\n--- Summary for {tic_id} ({mission_name}) ---")
    print(f"Initial data points: {initial_points}")
    print(f"Removed {num_nans} missing/NaN flux values.")
    print(f"Final data points: {final_points}")
    print(f"Time span: {time_span:.2f} days")

    # Save to FITS and CSV
    safe_tic = tic_id.replace(' ', '_')
    safe_mission = mission_name.replace(' ', '_')
    fits_filename = os.path.join(output_dir, f"{safe_tic}_{safe_mission}.fits")
    csv_filename = os.path.join(output_dir, f"{safe_tic}_{safe_mission}.csv")
    
    try:
        lc_clean.to_fits(path=fits_filename, overwrite=True)
        print(f"Saved FITS to {fits_filename}")
    except Exception as e:
        print(f"Error saving FITS: {e}")

    try:
        lc_clean.to_csv(path_or_buf=csv_filename)
        print(f"Saved CSV to {csv_filename}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TESS light curves for a hardcoded list of TIC IDs.")
    parser.add_argument("--sector", type=int, default=None, help="Specific TESS sector to download. If not provided and multiple exist, you will be prompted.")
    parser.add_argument("--output", type=str, default="data", help="Output directory for saved light curves.")
    
    args = parser.parse_args()
    
    print("Starting TESS data download script...")
    for tic in TIC_IDS:
        download_and_save_lightcurve(tic, sector=args.sector, output_dir=args.output)
