# add_mid_price_column.py
"""
Adds mid-price columns to existing feature CSVs.

This script reads CSVs from mar_feature_data_with_regimes/, calculates the true mid-price
from the reconstructed orderbook data, and adds mid_price columns for each asset.

IMPORTANT: Handles multiple orderbook updates within a single 100ms interval correctly
by using the LAST (most recent) mid-price value in that interval after ffill.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = Path("./mar_feature_data_with_regimes")
OUTPUT_DIR = Path("./mar_feature_data_with_regimes_v2")
BASE_RAW_DATA_DIR = Path("./data/mar_raw_data")
RUN_ID_TO_PROCESS = "run_20250920_000825"

# Assets to process (must match the CSVs)
ASSETS = ["XBTUSD", "ETHUSD", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_orderbook_for_date(run_data_dir: Path, year: int, month: int, day: int, symbol: str) -> pd.DataFrame:
    """
    Load raw orderbook data for a specific date and symbol.

    Returns a DataFrame with timestamp index and orderbook columns (action, id, side, price, size).
    """
    try:
        filters = [('year', '==', year), ('month', '==', month), ('day', '==', day), ('symbol', '==', symbol)]
        orderbook_df = pd.read_parquet(run_data_dir / "orderbook", filters=filters)

        if not orderbook_df.empty:
            orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'], unit='s', utc=True)
            orderbook_df = orderbook_df.set_index('timestamp')
            orderbook_df.index = orderbook_df.index.tz_localize(None)

        return orderbook_df
    except Exception as e:
        logging.warning(f"Could not load orderbook for {symbol} on {year}{month:02d}{day:02d}: {e}")
        return pd.DataFrame()


def reconstruct_mid_prices(orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct mid-prices from orderbook updates.

    This function processes orderbook updates sequentially to maintain the book state
    and calculates mid-price at each update. Returns a DataFrame with timestamp index
    and 'mid_price' column.

    IMPORTANT: Multiple updates can occur within the same timestamp. We preserve all
    of them and let the resampling logic handle selecting the correct value.
    """
    if orderbook_df.empty:
        return pd.DataFrame(columns=['mid_price'])

    orderbook_df = orderbook_df.sort_index()

    n_rows = len(orderbook_df)
    timestamps = orderbook_df.index.values
    mid_prices = np.full(n_rows, np.nan, dtype=np.float64)

    # Use dictionaries to maintain book state
    # Format: {price: {id: size}}
    bids = {}
    asks = {}

    # Extract columns as numpy arrays for faster access
    actions = orderbook_df['action'].values
    ids = orderbook_df['id'].values
    sides = orderbook_df['side'].values
    prices = orderbook_df['price'].values
    sizes = orderbook_df['size'].values

    for idx in tqdm(range(n_rows), desc="Reconstructing Mid-Prices", disable=False):
        action = actions[idx]
        item_id = ids[idx]
        side = sides[idx]
        price = prices[idx]
        size = sizes[idx]

        is_buy = side == 'Buy'
        book_side = bids if is_buy else asks

        # Update the orderbook based on action
        if action == 'partial':
            bids.clear()
            asks.clear()
            if price > 0 and size > 0:
                if price not in book_side:
                    book_side[price] = {}
                book_side[price][item_id] = size
        elif action == 'insert':
            if price > 0 and size > 0:
                if price not in book_side:
                    book_side[price] = {}
                book_side[price][item_id] = size
        elif action == 'update':
            # Find the price level for this ID
            for p in list(book_side.keys()):
                if item_id in book_side[p]:
                    if size > 0:
                        book_side[p][item_id] = size
                    else:
                        del book_side[p][item_id]
                        if len(book_side[p]) == 0:
                            del book_side[p]
                    break
        elif action == 'delete':
            # Find and delete the order
            for p in list(book_side.keys()):
                if item_id in book_side[p]:
                    del book_side[p][item_id]
                    if len(book_side[p]) == 0:
                        del book_side[p]
                    break

        # Calculate mid-price from current book state
        if bids and asks:
            best_bid = max(bids.keys())
            best_ask = min(asks.keys())
            mid_prices[idx] = (best_bid + best_ask) * 0.5

    # Create DataFrame with all mid-price updates
    result_df = pd.DataFrame({'mid_price': mid_prices}, index=timestamps)

    return result_df


def calculate_true_mid_price_for_interval(mid_price_updates: pd.Series, interval_start: pd.Timestamp,
                                          interval_end: pd.Timestamp) -> float:
    """
    Calculate the TRUE mid-price for a 100ms interval.

    Key insight: When multiple orderbook updates occur in a single 100ms interval,
    we want the LAST (most recent) mid-price, as that's the true price at the end
    of the interval.

    Args:
        mid_price_updates: Series of mid-price values with timestamp index
        interval_start: Start of the 100ms interval
        interval_end: End of the 100ms interval

    Returns:
        The mid-price value to use for this interval
    """
    # Get all updates in this interval (exclusive of end to match pandas resample behavior)
    mask = (mid_price_updates.index >= interval_start) & (mid_price_updates.index < interval_end)
    updates_in_interval = mid_price_updates[mask]

    if len(updates_in_interval) == 0:
        return np.nan

    # Return the LAST value (most recent update in the interval)
    # This is the true mid-price at the end of the interval
    return updates_in_interval.iloc[-1]


def add_mid_price_to_csv(csv_path: Path, run_data_dir: Path) -> pd.DataFrame:
    """
    Load a CSV, calculate mid-prices for all assets, and add mid_price columns.

    Returns the modified DataFrame with new columns added.
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing: {csv_path.name}")
    logging.info(f"{'='*80}")

    # Load the existing CSV
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    logging.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

    # Extract date from filename (format: 100features_output_YYYYMMDD.csv)
    date_str = csv_path.stem.split('_')[-1]  # e.g., "20250919"
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    logging.info(f"Date: {year}-{month:02d}-{day:02d}")

    # Process each asset
    new_columns = {}

    for asset in ASSETS:
        logging.info(f"\nProcessing {asset}...")

        # Load raw orderbook data
        orderbook_df = load_orderbook_for_date(run_data_dir, year, month, day, asset)

        if orderbook_df.empty:
            logging.warning(f"No orderbook data for {asset}, filling with NaN")
            new_columns[f'{asset}_mid_price'] = np.nan
            continue

        # Reconstruct mid-prices from orderbook
        logging.info(f"Reconstructing mid-prices from {len(orderbook_df)} orderbook updates...")
        mid_price_df = reconstruct_mid_prices(orderbook_df)

        # Handle multiple updates per 100ms interval correctly:
        # 1. Forward fill to propagate mid-prices
        # 2. Resample to 100ms using 'last' to get the most recent value in each interval
        # 3. This matches the behavior of book_resampled.ffill() in create_features.py

        mid_price_filled = mid_price_df['mid_price'].ffill()

        # Resample to 100ms intervals, taking the LAST value in each interval
        # This ensures we get the most recent mid-price, handling multiple updates correctly
        mid_price_resampled = mid_price_filled.resample('100ms').last()

        # Align with the CSV's timestamp index
        mid_price_aligned = mid_price_resampled.reindex(df.index)

        # Forward fill any remaining NaNs (for intervals with no updates)
        mid_price_aligned = mid_price_aligned.ffill()

        # Log statistics
        valid_count = mid_price_aligned.notna().sum()
        logging.info(f"  Valid mid-prices: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")

        if valid_count > 0:
            logging.info(f"  Mid-price range: {mid_price_aligned.min():.2f} - {mid_price_aligned.max():.2f}")

        new_columns[f'{asset}_mid_price'] = mid_price_aligned

    # Now insert the mid_price columns in the correct positions
    # Format: regime, XBTUSD_f0..f6, XBTUSD_volume24h, XBTUSD_mid_price, ETHUSD_f0..f6, ...

    logging.info("\nInserting mid_price columns into DataFrame...")
    result_df = df.copy()

    # Track column positions as we insert
    for asset in ASSETS:
        # Find the position after {asset}_volume24h
        volume_col = f'{asset}_volume24h'

        if volume_col not in result_df.columns:
            logging.warning(f"Column {volume_col} not found, skipping {asset}_mid_price insertion")
            continue

        # Get the position to insert (right after volume24h)
        volume_idx = result_df.columns.get_loc(volume_col)
        insert_pos = volume_idx + 1

        # Insert the mid_price column
        mid_price_col = f'{asset}_mid_price'
        result_df.insert(insert_pos, mid_price_col, new_columns[mid_price_col])

        logging.info(f"  Inserted {mid_price_col} at position {insert_pos}")

    logging.info(f"\nFinal DataFrame: {len(result_df)} rows, {len(result_df.columns)} columns")

    return result_df


def main():
    """Main execution function."""
    logging.info("="*80)
    logging.info("Starting Mid-Price Column Addition")
    logging.info("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")

    # Get run directory
    run_dir = BASE_RAW_DATA_DIR / RUN_ID_TO_PROCESS
    if not run_dir.exists():
        logging.error(f"FATAL: Run directory not found at {run_dir}")
        return

    # Find all CSV files in input directory
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        logging.error(f"No CSV files found in {INPUT_DIR}")
        return

    logging.info(f"Found {len(csv_files)} CSV files to process")

    # Process each CSV
    for csv_path in csv_files:
        try:
            result_df = add_mid_price_to_csv(csv_path, run_dir)

            # Save to output directory with same filename
            output_path = OUTPUT_DIR / csv_path.name
            result_df.to_csv(output_path, date_format='%Y-%m-%d %H:%M:%S.%f')
            logging.info(f"\n✓ Saved to: {output_path}")

        except Exception as e:
            logging.error(f"\n✗ Error processing {csv_path.name}: {e}", exc_info=True)
            continue

    logging.info("\n" + "="*80)
    logging.info("Mid-Price Column Addition Complete!")
    logging.info("="*80)


if __name__ == "__main__":
    main()
