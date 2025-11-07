# create_features.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from numba import jit, prange
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# Try to import sortedcontainers, fallback to dict if not available
try:
    from sortedcontainers import SortedDict
    HAVE_SORTEDCONTAINERS = True
except ImportError:
    HAVE_SORTEDCONTAINERS = False
    logging.warning("sortedcontainers not installed. Install with: pip install sortedcontainers")
    logging.warning("Falling back to slower dict-based orderbook reconstruction")

# --- Configuration ---
BASE_RAW_DATA_DIR = Path("./data/mar_raw_data")
RUN_ID_TO_PROCESS = "run_20250920_000825"
FEATURE_OUTPUT_DIR = Path("./mar_feature_data")

SAMPLING_INTERVAL = "100ms"
LOB_DEPTH_FOR_FEATURES = 15
VOLATILITY_WINDOW_STR = "5s"
REGIME_WINDOW_STR = "120min"
PERFORMANCE_MODE = True  # Set to False to enable tqdm progress bars
NUM_WORKERS = max(1, cpu_count() - 1)  # Leave one core free

# Calculate the integer window sizes
vol_window_td = pd.to_timedelta(VOLATILITY_WINDOW_STR)
sampling_td = pd.to_timedelta(SAMPLING_INTERVAL)
VOLATILITY_WINDOW_INT = int(vol_window_td / sampling_td)

regime_window_td = pd.to_timedelta(REGIME_WINDOW_STR)
REGIME_WINDOW_INT = int(regime_window_td / sampling_td)

# =============================================================================
# FEATURE NORMALIZATION STRATEGY
# =============================================================================
# This script uses a "Transform & Scale" approach for feature engineering:
#
# 1. TRANSFORMATION (This Script):
#    - Features are transformed to be stationary (log returns, ratios, etc.)
#    - NO clipping or outlier filtering is applied
#    - Raw distributions are preserved to capture true market dynamics
#
# 2. STANDARDIZATION (Training Script):
#    - Apply StandardScaler (Z-score normalization) after loading all features
#    - Fit scaler ONLY on training data to prevent data leakage
#    - Transform both train and test data using the fitted scaler
#
# Example code for training script:
# ```python
# from sklearn.preprocessing import StandardScaler
#
# # Load all feature CSVs
# all_features_df = pd.concat([pd.read_csv(f) for f in feature_files], ignore_index=True)
#
# # Split train/test
# train_df = all_features_df[all_features_df['timestamp'] < split_date]
# test_df = all_features_df[all_features_df['timestamp'] >= split_date]
#
# # Fit scaler on training data only
# feature_columns = [col for col in train_df.columns if col != 'timestamp']
# scaler = StandardScaler()
# scaler.fit(train_df[feature_columns])
#
# # Transform both sets
# train_scaled = scaler.transform(train_df[feature_columns])
# test_scaled = scaler.transform(test_df[feature_columns])
# ```
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ALL_FEATURES from your MAR model's config.py
MAR_ALL_FEATURES = {
    "Price vs. Slow EMA (5min)": 0, "Mid-Price (Log Return)": 1, "Relative Spread": 2,
    "Multi-Level Cumulative OBI": 3, "Taker Imbalance": 4, "Realized Volatility (dt)": 5,
    "Multi-Level WAP": 6,
    # "Open Interest (Log Return)": 7, "Liquidation Volume": 8, "Funding Rate": 9  # DISABLED
    }

MAR_TARGET_FEATURE_NAME = "Price vs. Slow EMA (5min)"
MAR_PNL_FEATURE_NAME = "Mid-Price (Log Return)"

def filter_outliers(series, n_std=5, method='mad'):
    """
    Remove outliers using MAD (Median Absolute Deviation) or IQR method.
    MAD is more robust to extreme outliers than standard deviation.
    """
    if series.empty or series.isna().all():
        return series
    
    if method == 'mad':
        median = series.median()
        mad = (series - median).abs().median()
        # MAD to std conversion factor for normal distribution
        std_robust = 1.4826 * mad
        
        if std_robust < 1e-10:  # If no variation, return original
            return series
            
        lower = median - n_std * std_robust
        upper = median + n_std * std_robust
        
    elif method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - n_std * iqr
        upper = q3 + n_std * iqr
    
    else:  # Standard deviation method
        mean = series.mean()
        std = series.std()
        if std < 1e-10:
            return series
        lower = mean - n_std * std
        upper = mean + n_std * std
    
    # Log outliers for monitoring
    outliers_count = ((series < lower) | (series > upper)).sum()
    if outliers_count > 0:
        logging.debug(f"Filtering {outliers_count} outliers ({outliers_count/len(series)*100:.2f}%)")
    
    return series.clip(lower, upper)

def validate_raw_data(run_data_dir: Path, symbols: list, date_parts: tuple) -> bool:
    """Validates that the raw data is in the expected format before processing."""
    year, month, day = date_parts
    date_str = f"{year}{month:02d}{day:02d}"
    
    logging.info(f"Validating data structure for {date_str}...")
    
    required_dirs = ['trades', 'orderbook', 'instruments', 'liquidations']
    for dir_name in required_dirs:
        dir_path = run_data_dir / dir_name
        if not dir_path.exists():
            logging.error(f"Missing required directory: {dir_path}")
            return False
    
    sample_symbol = symbols[0] if symbols else None
    if not sample_symbol:
        logging.error("No symbols specified for processing")
        return False
    
    try:
        filters = [('year', '==', year), ('month', '==', month), ('day', '==', day), ('symbol', '==', sample_symbol)]
        
        trades_sample = pd.read_parquet(run_data_dir / "trades", filters=filters, columns=['timestamp', 'symbol'])
        if trades_sample.empty:
            logging.warning(f"No trades found for {sample_symbol} on {date_str}")
            return False
        
        ob_sample = pd.read_parquet(run_data_dir / "orderbook", filters=filters, columns=['timestamp', 'symbol', 'action'])
        if ob_sample.empty:
            logging.warning(f"No orderbook data found for {sample_symbol} on {date_str}")
            return False
        
        ts_sample = trades_sample['timestamp'].iloc[0]
        if not isinstance(ts_sample, (int, float)):
            logging.error(f"Unexpected timestamp format: {type(ts_sample)}")
            return False
            
        logging.info(f"✓ Data validation passed for {date_str}")
        return True
        
    except Exception as e:
        logging.error(f"Error during validation: {e}")
        return False

@jit(nopython=True, parallel=True)
def fast_rolling_std(values, window):
    """Numba-accelerated rolling standard deviation using Welford's algorithm"""
    n = len(values)
    result = np.zeros(n)

    if n < window:
        return result

    # Initialize first window
    mean = 0.0
    m2 = 0.0
    for i in range(window):
        delta = values[i] - mean
        mean += delta / (i + 1)
        delta2 = values[i] - mean
        m2 += delta * delta2

    # Safety clip to prevent sqrt of small negative number due to float inaccuracy
    m2_safe = max(0.0, m2)
    result[window - 1] = np.sqrt(m2_safe / window)

    # Slide the window
    for i in range(window, n):  # <-- Changed from prange to range
        old_val = values[i - window]
        new_val = values[i]

        # This check is crucial for stability with real-world data
        if old_val == new_val:
            result[i] = result[i-1]
            continue

        # Remove old value and add new value
        old_mean = mean
        mean = old_mean + (new_val - old_val) / window
        
        # Update variance using recurrence relation
        m2 = m2 + (new_val - old_val) * (new_val - mean + old_val - old_mean)
        
        m2_safe = max(0.0, m2)
        result[i] = np.sqrt(m2_safe / window)

    return result

@jit(nopython=True, parallel=True)
def fast_percentile_rank(values, window):
    """Numba-accelerated rolling percentile rank"""
    n = len(values)
    result = np.zeros(n)
    for i in prange(window - 1, n):
        window_values = values[i - window + 1:i + 1]
        current_value = values[i]
        rank = np.sum(window_values <= current_value) / window
        result[i] = rank
    return result

@jit(nopython=True)
def fast_ffill(arr):
    """Fast forward fill for numpy arrays"""
    result = arr.copy()
    last_valid = np.nan
    for i in range(len(arr)):
        if not np.isnan(arr[i]) and arr[i] != 0:
            last_valid = arr[i]
        elif not np.isnan(last_valid):
            result[i] = last_valid
    return result

def calculate_base_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Calculates the 9 stationary features for a single symbol.

    IMPORTANT: This function uses a "Transform & Scale" approach:
    - Features are transformed to be stationary (e.g., log returns, ratios)
    - NO clipping or outlier filtering is applied here
    - Raw distributions are preserved to capture true market dynamics
    - Normalization/standardization should be applied in the training script using StandardScaler

    This approach allows the MAR model to:
    - See the true magnitude of extreme events (e.g., 5-sigma events)
    - Adapt to changing market regimes dynamically
    - Avoid information loss from static bounds
    """
    logging.info(f"Calculating features for {symbol}...")

    # Pre-calculate mid price with vectorized operations
    bid_l1 = df['bid_price_L1'].ffill()
    ask_l1 = df['ask_price_L1'].ffill()
    mid_price = (bid_l1 + ask_l1) * 0.5

    # Fix zero/invalid mid prices
    valid_price_mask = mid_price > 0
    if not valid_price_mask.all():
        logging.warning(f"Found {(~valid_price_mask).sum()} zero/invalid mid prices for {symbol}")
        mid_price_median = mid_price[valid_price_mask].median()
        mid_price = mid_price.where(valid_price_mask, mid_price_median)

    # Cache mid_price values for reuse
    mid_price_vals = mid_price.values

    # 1. Relative Spread (Enhanced with basis spread)
    spread = np.abs(ask_l1.values - bid_l1.values)
    bid_ask_spread = np.where(mid_price_vals > 0, spread / mid_price_vals, 0)

    # Enhancement 4: Add basis spread (markPrice - indexPrice)
    if 'markPrice' in df.columns and 'indexPrice' in df.columns:
        # Calculate basis spread: divergence between mark and index
        index_price_vals = df['indexPrice'].values
        mark_price_vals = df['markPrice'].values
        basis_spread = np.where(
            index_price_vals > 0,
            (mark_price_vals - index_price_vals) / index_price_vals,
            0
        )
        # Combine bid-ask spread with absolute basis spread
        df['relative_spread'] = bid_ask_spread + np.abs(basis_spread)
    else:
        df['relative_spread'] = bid_ask_spread

    # No clipping - preserve raw distribution for scaling in training script

    # 2. Multi-Level WAP (Vectorized)
    n_rows = len(df)
    wap_numerator = np.zeros(n_rows, dtype=np.float64)
    wap_denominator = np.zeros(n_rows, dtype=np.float64)

    # Vectorized calculation for all levels at once
    for i in range(1, LOB_DEPTH_FOR_FEATURES + 1):
        bid_price = df[f'bid_price_L{i}'].fillna(0).values
        bid_size = df[f'bid_size_L{i}'].fillna(0).values
        ask_price = df[f'ask_price_L{i}'].fillna(0).values
        ask_size = df[f'ask_size_L{i}'].fillna(0).values

        # Vectorized multiply and accumulate
        wap_numerator += bid_price * bid_size + ask_price * ask_size
        wap_denominator += bid_size + ask_size
    
    # Calculate WAP with robust fallback
    wap_vals = np.where(
        wap_denominator > 0,
        wap_numerator / wap_denominator,
        mid_price_vals
    )

    # Additional validation for WAP
    zero_wap_mask = wap_vals <= 0
    if zero_wap_mask.any():
        pct_invalid = (zero_wap_mask.sum() / n_rows) * 100
        logging.warning(f"Found {zero_wap_mask.sum()} ({pct_invalid:.2f}%) zero/negative WAP values for {symbol}, using mid price")
        wap_vals[zero_wap_mask] = mid_price_vals[zero_wap_mask]

    df['wap'] = wap_vals

    # No clipping - preserve raw distribution for scaling in training script

    # 2. Price vs. Slow EMA (5min)
    # Calculate 5-minute EMA window size (5 minutes at 100ms sampling = 5 * 60 * 10 = 3000 samples)
    ema_window_5min = int(pd.to_timedelta("5min") / sampling_td)

    # Calculate EMA using pandas ewm (exponential weighted moving average)
    # span parameter: for EMA, alpha = 2/(span+1), so span = (2/alpha) - 1
    # Common EMA convention: span = window size for comparable smoothing
    wap_ema_5min = pd.Series(wap_vals).ewm(span=ema_window_5min, adjust=False).mean().values

    # Calculate deviation: (Current WAP / EMA) - 1
    # This gives percentage deviation from the EMA (stationary signal)
    df['price_vs_ema_5min'] = np.where(
        wap_ema_5min > 0,
        (wap_vals / wap_ema_5min) - 1,
        0
    )
    # No clipping - preserve raw distribution for scaling in training script

    # 3. Multi-Level Cumulative OBI (Vectorized)
    total_bid_size = np.zeros(n_rows, dtype=np.float64)
    total_ask_size = np.zeros(n_rows, dtype=np.float64)

    for i in range(1, LOB_DEPTH_FOR_FEATURES + 1):
        total_bid_size += df[f'bid_size_L{i}'].fillna(0).values
        total_ask_size += df[f'ask_size_L{i}'].fillna(0).values

    total_size = total_bid_size + total_ask_size
    df['obi'] = np.where(
        total_size > 0,
        (total_bid_size - total_ask_size) / total_size,
        0
    )
    # OBI is naturally bounded between -1 and 1, no need to clip

    # 4. Taker Imbalance
    if 'taker_imbalance' not in df.columns:
        df['taker_imbalance'] = 0
    # No clipping - preserve raw distribution for scaling in training script

    # 5. Realized Volatility
    # Use the high-fidelity 'mid_price_log_return', which was calculated from every tick
    # and then summed during resampling. This preserves the true variance.
    log_returns_clean = df['mid_price_log_return'].values
    df['realized_volatility'] = fast_rolling_std(log_returns_clean, VOLATILITY_WINDOW_INT)
    # No clipping - preserve raw distribution for scaling in training script

    # # 6. Open Interest (Log Return) - Optimized - DISABLED
    # # Forward-fill to handle patchy updates, which is a more realistic assumption
    # oi_vals = df['openInterest'].ffill().values
    # oi_safe = np.maximum(oi_vals, 1.0)  # Clip lower to 1
    # oi_safe = np.nan_to_num(oi_safe, nan=1.0)

    # # Compute log diff in one vectorized operation
    # oi_log = np.log(oi_safe)
    # oi_log_shifted = np.roll(oi_log, 1)
    # oi_log_shifted[0] = oi_log[0]
    # oi_log_return = oi_log - oi_log_shifted
    # oi_log_return = np.nan_to_num(oi_log_return, nan=0, posinf=0, neginf=0)

    # df['open_interest_log_return'] = oi_log_return

    # # 7. Liquidation Volume - DISABLED
    # if 'liquidation_volume' not in df.columns:
    #     df['liquidation_volume'] = 0

    # # 8. Enhanced Funding Rate - Optimized - DISABLED
    # if 'fundingRate' in df.columns:
    #     funding_vals = df['fundingRate'].values.copy()
    #     # Enhancement 5: Incorporate funding rate change (current vs indicative)
    #     if 'indicativeFundingRate' in df.columns:
    #         # Use the change in funding rate to capture forward-looking pressure
    #         indicative_vals = df['indicativeFundingRate'].values
    #         funding_change = funding_vals - indicative_vals
    #         # Blend current funding with the change signal
    #         # Positive change = funding pressure increasing
    #         funding_vals = funding_vals + (0.5 * funding_change)
    #     df['fundingRate'] = funding_vals
    #     # No clipping - preserve raw distribution for scaling in training script
    # else:
    #     df['fundingRate'] = 0

    # Mid-price log return (for PnL calculation) - Already computed, reuse
    df['mid_price_log_return'] = log_returns_clean
    # No clipping - preserve raw distribution for scaling in training script

    # Store volume24h for potential future use if available
    if 'volume24h' in df.columns:
        df['volume24h_raw'] = df['volume24h'].fillna(0)

    # Map to MAR model's expected feature IDs
    feature_map = {
        'price_vs_ema_5min':        f"{symbol}_f{MAR_ALL_FEATURES['Price vs. Slow EMA (5min)']}",
        'mid_price_log_return':     f"{symbol}_f{MAR_ALL_FEATURES['Mid-Price (Log Return)']}",
        'relative_spread':          f"{symbol}_f{MAR_ALL_FEATURES['Relative Spread']}",
        'obi':                      f"{symbol}_f{MAR_ALL_FEATURES['Multi-Level Cumulative OBI']}",
        'taker_imbalance':          f"{symbol}_f{MAR_ALL_FEATURES['Taker Imbalance']}",
        'realized_volatility':      f"{symbol}_f{MAR_ALL_FEATURES['Realized Volatility (dt)']}",
        'wap':                      f"{symbol}_f{MAR_ALL_FEATURES['Multi-Level WAP']}",
        # 'open_interest_log_return': f"{symbol}_f{MAR_ALL_FEATURES['Open Interest (Log Return)']}",
        # 'liquidation_volume':       f"{symbol}_f{MAR_ALL_FEATURES['Liquidation Volume']}",
        # 'fundingRate':              f"{symbol}_f{MAR_ALL_FEATURES['Funding Rate']}",
    }

    final_features = df[list(feature_map.keys())].rename(columns=feature_map)

    # Add volume24h as a separate column for potential future use
    if 'volume24h_raw' in df.columns:
        final_features[f'{symbol}_volume24h'] = df['volume24h_raw']

    # Add mid_price column (if available from combined_df)
    if 'mid_price' in df.columns:
        final_features[f'{symbol}_mid_price'] = df['mid_price']

    # Final sanity check - no infinities or NaNs
    if final_features.isin([np.inf, -np.inf]).any().any():
        logging.error(f"Infinities detected after feature calculation for {symbol}")
        final_features = final_features.replace([np.inf, -np.inf], 0)
    
    return final_features.fillna(0.0)

def reconstruct_book_snapshots_optimized(orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """Highly optimized orderbook reconstruction using sorted dictionaries and vectorization."""
    if orderbook_df.empty:
        return pd.DataFrame()

    orderbook_df = orderbook_df.sort_index()

    n_rows = len(orderbook_df)

    # Pre-allocate arrays
    timestamps = orderbook_df.index.values
    log_returns = np.zeros(n_rows, dtype=np.float64)
    mid_prices = np.full(n_rows, np.nan, dtype=np.float64)  # NEW: Store mid-prices

    # Allocate 2D arrays for prices and sizes
    bid_prices = np.full((n_rows, LOB_DEPTH_FOR_FEATURES), np.nan, dtype=np.float64)
    bid_sizes = np.full((n_rows, LOB_DEPTH_FOR_FEATURES), np.nan, dtype=np.float64)
    ask_prices = np.full((n_rows, LOB_DEPTH_FOR_FEATURES), np.nan, dtype=np.float64)
    ask_sizes = np.full((n_rows, LOB_DEPTH_FOR_FEATURES), np.nan, dtype=np.float64)

    # Use sorted dictionaries for O(log n) operations
    if HAVE_SORTEDCONTAINERS:
        from sortedcontainers import SortedDict
        bids = SortedDict()  # price -> {id: size}
        asks = SortedDict()
    else:
        # Fallback to regular dict (will need sorting later)
        bids = {}
        asks = {}

    # Extract columns as numpy arrays for faster access
    actions = orderbook_df['action'].values
    ids = orderbook_df['id'].values
    sides = orderbook_df['side'].values
    prices = orderbook_df['price'].values
    sizes = orderbook_df['size'].values

    last_mid_price = -1.0

    iterator = range(n_rows)
    if not PERFORMANCE_MODE:
        iterator = tqdm(iterator, desc="Reconstructing Order Book", total=n_rows)

    for idx in iterator:
        action = actions[idx]
        item_id = ids[idx]
        side = sides[idx]
        price = prices[idx]
        size = sizes[idx]

        is_buy = side == 'Buy'
        book_side = bids if is_buy else asks

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
            for p in book_side:
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
            for p in book_side:
                if item_id in book_side[p]:
                    del book_side[p][item_id]
                    if len(book_side[p]) == 0:
                        del book_side[p]
                    break

        # Get top N levels
        # For bids: highest prices first (reverse)
        # For asks: lowest prices first (normal)

        if HAVE_SORTEDCONTAINERS:
            # Already sorted, just reverse for bids
            bid_levels = list(reversed(list(bids.items())))[:LOB_DEPTH_FOR_FEATURES]
            ask_levels = list(asks.items())[:LOB_DEPTH_FOR_FEATURES]
        else:
            # Need to sort manually
            bid_levels = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:LOB_DEPTH_FOR_FEATURES]
            ask_levels = sorted(asks.items(), key=lambda x: x[0])[:LOB_DEPTH_FOR_FEATURES]

        for i, (p, orders) in enumerate(bid_levels):
            total_size = sum(orders.values())
            bid_prices[idx, i] = p
            bid_sizes[idx, i] = total_size

        for i, (p, orders) in enumerate(ask_levels):
            total_size = sum(orders.values())
            ask_prices[idx, i] = p
            ask_sizes[idx, i] = total_size

        # Calculate log return and store mid-price
        if bid_levels and ask_levels:
            best_bid = bid_levels[0][0]
            best_ask = ask_levels[0][0]
            current_mid_price = (best_bid + best_ask) * 0.5
            mid_prices[idx] = current_mid_price  # NEW: Store mid-price
            if last_mid_price > 0 and current_mid_price > 0:
                log_return = np.log(current_mid_price / last_mid_price)
                log_returns[idx] = np.clip(log_return, -0.1, 0.1)
            last_mid_price = current_mid_price

    # Build dataframe efficiently
    snapshots_data = {
        'mid_price_log_return': log_returns,
        'mid_price': mid_prices  # NEW: Include mid-price column
    }

    for i in range(LOB_DEPTH_FOR_FEATURES):
        snapshots_data[f'bid_price_L{i+1}'] = bid_prices[:, i]
        snapshots_data[f'bid_size_L{i+1}'] = bid_sizes[:, i]
        snapshots_data[f'ask_price_L{i+1}'] = ask_prices[:, i]
        snapshots_data[f'ask_size_L{i+1}'] = ask_sizes[:, i]

    snapshots_df = pd.DataFrame(snapshots_data, index=timestamps)

    # Remove duplicates
    if snapshots_df.index.duplicated().any():
        snapshots_df = snapshots_df[~snapshots_df.index.duplicated(keep='last')]

    return snapshots_df


def process_single_symbol(args):
    """Process a single symbol - designed for parallel execution."""
    run_data_dir, year, month, day, symbol = args
    date_str = f"{year}{month:02d}{day:02d}"

    try:
        logging.info(f"--- Processing {symbol} for date {date_str} ---")
        filters = [('year', '==', year), ('month', '==', month), ('day', '==', day), ('symbol', '==', symbol)]

        trades_df = pd.read_parquet(run_data_dir / "trades", filters=filters,
                                   columns=['timestamp', 'symbol', 'side', 'size', 'foreignNotional', 'tickDirection'])
        orderbook_df = pd.read_parquet(run_data_dir / "orderbook", filters=filters)
        # DISABLED: 'openInterest', 'fundingRate', 'indicativeFundingRate' (features disabled)
        instruments_df = pd.read_parquet(run_data_dir / "instruments", filters=filters,
                                        columns=['timestamp', 'symbol', 'markPrice', 'indexPrice', 'volume24h'])

        # DISABLED: Liquidation data loading (feature disabled)
        liquidations_df = pd.DataFrame()
        # # try:
        # #     liquidations_df = pd.read_parquet(run_data_dir / "liquidations", filters=filters,
        # #                                      columns=['timestamp', 'symbol', 'leavesQty', 'side'])
        # #     if not liquidations_df.empty:
        # #         logging.info(f"Found {len(liquidations_df)} liquidation events for {symbol}")
        # #     else:
        # #         logging.info(f"Liquidation data file found for {symbol}, but it is empty.")
        # # except Exception as e:
        # #     liquidations_df = pd.DataFrame()
        # #     logging.warning(f"Could not load liquidations data for {symbol} on {date_str}. Error: {e}. Assuming zero liquidations.")

        # Convert timestamps once for all dataframes
        # Use timezone-naive timestamps to avoid tz-aware/tz-naive join errors
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='s', utc=True)
            trades_df = trades_df.set_index('timestamp')
            trades_df.index = trades_df.index.tz_localize(None)
        if not orderbook_df.empty:
            orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'], unit='s', utc=True)
            orderbook_df = orderbook_df.set_index('timestamp')
            orderbook_df.index = orderbook_df.index.tz_localize(None)
        if not instruments_df.empty:
            instruments_df['timestamp'] = pd.to_datetime(instruments_df['timestamp'], unit='s', utc=True)
            instruments_df = instruments_df.set_index('timestamp')
            instruments_df.index = instruments_df.index.tz_localize(None)
        # DISABLED: Liquidations timestamp conversion (feature disabled)
        # if not liquidations_df.empty:
        #     liquidations_df['timestamp'] = pd.to_datetime(liquidations_df['timestamp'], unit='s', utc=True)
        #     liquidations_df = liquidations_df.set_index('timestamp')
        #     liquidations_df.index = liquidations_df.index.tz_localize(None)

    except Exception as e:
        logging.warning(f"Could not load data for {symbol} on {date_str}. Error: {e}")
        return None

    if orderbook_df.empty and trades_df.empty and instruments_df.empty:
        logging.warning(f"No data found for {symbol} on {date_str}. Skipping.")
        return None

    book_snapshots_df = reconstruct_book_snapshots_optimized(orderbook_df)

    logging.info(f"Resampling data to common interval for {symbol}...")

    if not trades_df.empty:
        # Enhancement 1: Use foreignNotional (USD value) instead of size
        # Apply tick direction weighting for aggressiveness
        tick_weight = trades_df['tickDirection'].map({
            'PlusTick': 1.5,
            'ZeroPlusTick': 1.5,
            'MinusTick': -1.5,
            'ZeroMinusTick': -1.5
        }).fillna(1.0)  # Default weight for unrecognized tick directions

        # Use foreignNotional if available, otherwise fallback to size
        if 'foreignNotional' in trades_df.columns:
            trade_value = trades_df['foreignNotional'].abs()
        else:
            trade_value = trades_df['size'].abs()

        # Calculate weighted buy/sell volume
        trades_df['buy_volume_weighted'] = np.where(
            trades_df['side'] == 'Buy',
            trade_value * tick_weight,
            0
        )
        trades_df['sell_volume_weighted'] = np.where(
            trades_df['side'] == 'Sell',
            trade_value * tick_weight.abs(),  # Use abs to keep sells positive
            0
        )

        trades_resampled = trades_df[['buy_volume_weighted', 'sell_volume_weighted']].resample(SAMPLING_INTERVAL).sum()
        imbalance = trades_resampled['buy_volume_weighted'] - trades_resampled['sell_volume_weighted']

        # --- NEW: Apply signed log transform to tame the scale ---
        trades_resampled['taker_imbalance'] = np.sign(imbalance) * np.log1p(np.abs(imbalance))

    else:
        trades_resampled = pd.DataFrame(columns=['taker_imbalance'])

    # Liquidation resampling disabled (feature disabled)
    liquidations_resampled = pd.DataFrame(columns=['liquidation_volume'])
    # if liquidations_df.empty:
    #     liquidations_resampled = pd.DataFrame(columns=['liquidation_volume'])
    # else:
    #     # Enhancement 2: Make liquidation volume directional
    #     # side='Sell' means long positions being liquidated (bearish) -> positive
    #     # side='Buy' means short positions being liquidated (bullish) -> negative
    #     # NOTE: Using 'leavesQty' which contains the actual liquidation quantity (orderQty is always 0)
    #     liquidations_df['liquidation_volume'] = np.where(
    #         liquidations_df['side'] == 'Sell',
    #         liquidations_df['leavesQty'],     # Long liquidations: positive
    #         -liquidations_df['leavesQty']      # Short liquidations: negative
    #     )
    #     liquidations_resampled = liquidations_df[['liquidation_volume']].resample(SAMPLING_INTERVAL).sum()

    instruments_resampled = instruments_df.resample(SAMPLING_INTERVAL).ffill() if not instruments_df.empty else pd.DataFrame()

    log_returns_resampled = book_snapshots_df[['mid_price_log_return']].resample(SAMPLING_INTERVAL).sum()

    # Handle mid_price resampling: ffill then resample with 'last' to get the most recent value
    # This matches the behavior in add_mid_price_column.py for handling multiple updates per interval
    mid_price_filled = book_snapshots_df['mid_price'].ffill()
    mid_price_resampled = mid_price_filled.resample(SAMPLING_INTERVAL).last()

    # Drop mid_price_log_return and mid_price from book snapshots before ffill
    book_resampled = book_snapshots_df.drop(columns=['mid_price_log_return', 'mid_price']).resample(SAMPLING_INTERVAL).ffill()

    # 1. Fill event-based data. NaNs mean the event did not happen, so fill with 0.
    if 'taker_imbalance' in trades_resampled.columns:
        trades_resampled['taker_imbalance'] = trades_resampled['taker_imbalance'].fillna(0)
    # Liquidation fillna disabled (feature disabled)
    # if not liquidations_resampled.empty:
    #     liquidations_resampled['liquidation_volume'] = liquidations_resampled['liquidation_volume'].fillna(0)

    # 2. Fill state-based data. NaNs mean the state is unchanged.
    # Use ffill and then bfill to handle NaNs at the very beginning of the series.
    if not instruments_resampled.empty:
        instruments_resampled = instruments_resampled.ffill().bfill()

    # Book snapshots are already ffilled during resampling. We do it again as a safety measure.
    book_resampled = book_resampled.ffill()

    # 3. Combine all sources using join.
    # Start with the most frequent data source (the book) as the base.
    # The log returns are event-like, so NaNs should be 0.
    combined_df = book_resampled.join(log_returns_resampled.fillna(0), how='outer')

    # Join mid_price (forward fill any remaining NaNs from alignment)
    combined_df = combined_df.join(mid_price_resampled.to_frame('mid_price'), how='outer')
    combined_df['mid_price'] = combined_df['mid_price'].ffill()

    if not instruments_resampled.empty:
        # --- FIX: Explicitly select ONLY numeric columns to join ---
        # This prevents Categorical columns from being added to combined_df
        # Removed: 'openInterest', 'fundingRate', 'indicativeFundingRate' (features disabled)
        numeric_instrument_cols = [
            'markPrice', 'indexPrice', 'volume24h'
        ]
        # Filter for columns that actually exist in the dataframe
        cols_to_join = [col for col in numeric_instrument_cols if col in instruments_resampled.columns]
        if cols_to_join:
            combined_df = combined_df.join(instruments_resampled[cols_to_join], how='outer')

    if not trades_resampled.empty:
        combined_df = combined_df.join(trades_resampled[['taker_imbalance']], how='outer')
    # Liquidation join disabled (feature disabled)
    # if not liquidations_resampled.empty:
    #     combined_df = combined_df.join(liquidations_resampled[['liquidation_volume']], how='outer')

    # 4. Final fill to ensure no gaps remain.
    # ffill first to propagate the last known states, then fill any remaining NaNs (at the start) with 0.
    combined_df = combined_df.ffill().fillna(0)

    if combined_df.empty:
        logging.warning(f"No combined data for {symbol} on {date_str}. Skipping.")
        return None

    symbol_base_features = calculate_base_features(combined_df, symbol)
    return symbol_base_features

def process_day(run_data_dir: Path, date_parts: tuple, symbols: list, prev_day_df: pd.DataFrame = None):
    """Process a single day of data with optional previous day for warmup."""
    year, month, day = date_parts
    date_str = f"{year}{month:02d}{day:02d}"

    # Validate before processing
    if not validate_raw_data(run_data_dir, symbols, date_parts):
        logging.error(f"Validation failed for {date_str}. Skipping.")
        return None  # Changed from return to return None

    all_core_features = []

    # Prepare arguments for parallel processing
    if NUM_WORKERS > 1 and len(symbols) > 1:
        logging.info(f"Processing {len(symbols)} symbols in parallel with {NUM_WORKERS} workers...")
        args_list = [(run_data_dir, year, month, day, symbol) for symbol in symbols]

        with Pool(NUM_WORKERS) as pool:
            results = pool.map(process_single_symbol, args_list)

        # Filter out None results
        all_core_features = [r for r in results if r is not None]
    else:
        # Sequential processing
        logging.info(f"Processing {len(symbols)} symbols sequentially...")
        for symbol in symbols:
            result = process_single_symbol((run_data_dir, year, month, day, symbol))
            if result is not None:
                all_core_features.append(result)

    if not all_core_features:
        logging.error(f"No features could be generated for any symbol on {date_str}.")
        return None

    daily_features_df = pd.concat(all_core_features, axis=1).sort_index()
    daily_features_df = daily_features_df.fillna(0.0)

    # **NEW: Prepend previous day data for warmup if available**
    if prev_day_df is not None:
        logging.info(f"Using {len(prev_day_df)} samples from previous day as warmup")
        combined_df = pd.concat([prev_day_df, daily_features_df])
        warmup_samples = len(prev_day_df)
    else:
        combined_df = daily_features_df
        warmup_samples = 0

    final_df = daily_features_df

    # Final data quality check
    inf_count = np.isinf(final_df.values).sum()
    nan_count = np.isnan(final_df.values).sum()
    if inf_count > 0 or nan_count > 0:
        logging.warning(f"Final data contains {inf_count} infinities and {nan_count} NaNs. Cleaning...")
        final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    final_df.index = pd.to_datetime(final_df.index)

    output_path = FEATURE_OUTPUT_DIR / f"features_output_{date_str}.csv"
    final_df.to_csv(output_path, date_format='%Y-%m-%d %H:%M:%S.%f')
    logging.info(f"Successfully saved final features to {output_path} ({len(final_df)} rows)")

    return final_df  # Return the dataframe for next day's warmup

if __name__ == "__main__":
    FEATURE_OUTPUT_DIR.mkdir(exist_ok=True)
    run_dir = BASE_RAW_DATA_DIR / RUN_ID_TO_PROCESS

    if not run_dir.exists():
        logging.error(f"FATAL: Run directory not found at {run_dir}")
        exit()

    logging.info(f"Starting feature generation for run: {RUN_ID_TO_PROCESS}")

    all_day_paths = run_dir.glob("trades/year=*/month=*/day=*")
    unique_dates = sorted(list(set(
        (int(p.parts[-3].split('=')[1]), int(p.parts[-2].split('=')[1]), int(p.parts[-1].split('=')[1]))
        for p in all_day_paths
    )))

    if not unique_dates:
        logging.error("FATAL: No day-partitioned data found in the 'trades' directory.")
        exit()

    logging.info(f"Discovered {len(unique_dates)} days to process: {unique_dates}")

    SYMBOLS_TO_PROCESS = [
        "XBTUSD", "ETHUSD", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"
        # , "LTCUSDT", "LINKUSDT", "DOTUSDT", "SUIUSDT"
    ]

    prev_day_features = None
    for date_tuple in unique_dates:
        result = process_day(run_dir, date_tuple, SYMBOLS_TO_PROCESS, prev_day_features)
        if result is not None:
            # Keep last 120 minutes (or 2x REGIME_WINDOW) for next day's warmup
            warmup_duration = pd.Timedelta(minutes=int(REGIME_WINDOW_STR.replace('min', '')))
            prev_day_features = result[result.index > (result.index[-1] - warmup_duration * 2)]

    logging.info("--- Feature Generation Complete ---")

    # Automatically run regime discovery
    logging.info("\n" + "="*60)
    logging.info("Starting automatic regime discovery...")
    logging.info("="*60)

    import subprocess
    import sys

    try:
        # Run discover_regimes.py as a separate process
        result = subprocess.run(
            [sys.executable, "discover_regimes.py"],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            logging.info("Regime discovery completed successfully!")
            logging.info(result.stdout)
        else:
            logging.error("Regime discovery failed!")
            logging.error(result.stderr)
    except Exception as e:
        logging.error(f"Failed to run regime discovery: {e}", exc_info=True)
        logging.warning("Feature files saved without regime labels")
