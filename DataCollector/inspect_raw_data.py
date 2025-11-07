# inspect_raw_data.py (Corrected)

import pandas as pd
from pathlib import Path
import logging

# --- Configuration (Copied from create_features.py) ---
BASE_RAW_DATA_DIR = Path("./data/mar_raw_data")
RUN_ID_TO_PROCESS = "run_20250920_000825"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inspect_liquidations(run_dir: Path):
    """Scans raw liquidation files and prints the first few non-zero entries from EACH file that contains them."""
    print("\n" + "="*80)
    print("  INSPECTING: Raw Liquidation Data")
    print("="*80)
    
    liquidations_dir = run_dir / "liquidations"
    if not liquidations_dir.exists():
        logging.error(f"Liquidation directory not found at: {liquidations_dir}")
        return

    total_found_data = False
    # Use glob to search through all day/month/year partitions
    for file_path in sorted(liquidations_dir.glob("**/*.parquet")):
        try:
            df = pd.read_parquet(file_path, columns=['timestamp', 'symbol', 'side', 'orderQty', 'price'])
            if df.empty:
                continue

            non_zero_liqs = df[df['orderQty'] != 0]

            if not non_zero_liqs.empty:
                logging.info(f"SUCCESS: Found {len(non_zero_liqs)} non-zero liquidation events in: {file_path.name}")
                print("--- Sample Events ---")
                print(non_zero_liqs.head(3).to_string())
                print("-" * 23)
                total_found_data = True
        except Exception as e:
            logging.warning(f"Could not read or process file {file_path.name}: {e}")
            
    if not total_found_data:
        print("\n[!] No non-zero liquidation events found in ANY of the raw parquet files.")

def inspect_funding_rates(run_dir: Path):
    """Scans raw instrument files and prints the first few non-zero funding rates from EACH file that contains them."""
    print("\n" + "="*80)
    print("  INSPECTING: Raw Funding Rate Data (from Instruments)")
    print("="*80)
    
    instruments_dir = run_dir / "instruments"
    if not instruments_dir.exists():
        logging.error(f"Instruments directory not found at: {instruments_dir}")
        return

    total_found_data = False
    for file_path in sorted(instruments_dir.glob("**/*.parquet")):
        try:
            df = pd.read_parquet(file_path, columns=['timestamp', 'symbol', 'fundingRate', 'openInterest'])
            if df.empty:
                continue
                
            non_zero_funding = df[df['fundingRate'].notna() & (df['fundingRate'] != 0)]

            if not non_zero_funding.empty:
                logging.info(f"SUCCESS: Found {len(non_zero_funding)} non-zero funding rate updates in: {file_path.name}")
                print("--- Sample Updates ---")
                print(non_zero_funding.head(3).to_string())
                print("-" * 24)
                total_found_data = True
        except Exception as e:
            logging.warning(f"Could not read or process file {file_path.name}: {e}")

    if not total_found_data:
        print("\n[!] No non-zero funding rate updates found in ANY of the raw parquet files.")


if __name__ == "__main__":
    run_directory = BASE_RAW_DATA_DIR / RUN_ID_TO_PROCESS
    
    if not run_directory.exists():
        logging.error(f"FATAL: Run directory not found at {run_directory}")
        exit()
        
    logging.info(f"Starting inspection for run: {RUN_ID_TO_PROCESS}")

    inspect_liquidations(run_directory)
    inspect_funding_rates(run_directory)

    print("\n" + "="*80)
    print("                      INSPECTION COMPLETE")
    print("="*80)