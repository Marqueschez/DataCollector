# data_manager.py - Updated with Definitive Data Sanitization Logic

import pandas as pd
import numpy as np
import logging

# --- CONFIGURATION FROM DEFINITIVE FINAL VERSION ---

# Z-score thresholds for outlier detection during data sanitization
# Adjust these values per feature - higher values = less aggressive filtering
DEFAULT_Z_SCORE_THRESHOLD = 15.0

# Feature-level thresholds (applied to all assets by default)
FEATURE_Z_SCORE_THRESHOLDS = {
    "Price vs. Slow EMA (5min)": 15.0,
    "Multi-Level WAP": 4.0,
    "Mid-Price (Log Return)": 60.0,
    "Relative Spread": 30.0,
    "Realized Volatility (dt)": 30.0,
    # Add other features here as needed with custom thresholds
}

# Per-asset overrides for specific features (format: (feature_name, asset_name): threshold)
# These take precedence over FEATURE_Z_SCORE_THRESHOLDS
ASSET_FEATURE_Z_SCORE_THRESHOLDS = {
    ("Multi-Level WAP", "ADAUSDT"): 2.0,
}

# --- SANITIZATION FUNCTIONS FROM DEFINITIVE FINAL VERSION ---

def sanitize_features(df):
    """Final safety check on feature data"""
    # Replace any remaining infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # Check for suspicious zero prices in WAP columns (f6 only, NOT f0!)
    # f0 = "Price vs. Slow EMA (5min)" - this is a DEVIATION metric that can be negative
    # f6 = "Multi-Level WAP" - this is an actual price that should never be zero/negative
    wap_cols = [col for col in df.columns if '_f6' in col]
    for col in wap_cols:
        if (df[col] <= 0).any():
            logging.warning(f"Found zero/negative values in {col}, replacing with median")
            median_val = df[col][df[col] > 0].median() if (df[col] > 0).any() else 1.0
            df.loc[df[col] <= 0, col] = median_val

    # Fill any remaining NaNs
    df = df.fillna(0)
    return df

def handle_known_artifacts(df):
    """
    Detects and corrects extreme statistical outliers across the entire dataset.
    Uses z-score thresholding to identify data artifacts that are likely collection errors.
    Supports per-feature and per-asset z-score thresholds.
    """
    logging.info("Scanning for extreme statistical outliers across entire dataset...")

    # Define features susceptible to data collection artifacts
    susceptible_features = {
        "Price vs. Slow EMA (5min)": 0,
        "Mid-Price (Log Return)": 1,
        "Relative Spread": 2,
        "Realized Volatility (dt)": 5,
        "Multi-Level WAP": 6,
    }

    # Log the thresholds being used
    logging.info("Per-feature z-score thresholds (default):")
    for feature_name in susceptible_features.keys():
        threshold = FEATURE_Z_SCORE_THRESHOLDS.get(feature_name, DEFAULT_Z_SCORE_THRESHOLD)
        logging.info(f"  {feature_name}: {threshold}")

    if ASSET_FEATURE_Z_SCORE_THRESHOLDS:
        logging.info("Per-asset overrides:")
        for (feature_name, asset_name), threshold in ASSET_FEATURE_Z_SCORE_THRESHOLDS.items():
            logging.info(f"  {feature_name} for {asset_name}: {threshold}")

    assets_in_data = sorted(list(set(col.split('_f')[0] for col in df.columns if '_f' in col)))

    MAX_ITERATIONS = 5  # Iterate to catch outliers that were hidden by other outliers

    outlier_count = 0
    cols_corrected = set()

    # Iterative outlier removal
    for iteration in range(MAX_ITERATIONS):
        iteration_outliers = 0

        for asset in assets_in_data:
            for feature_name, feature_id in susceptible_features.items():
                col_name = f"{asset}_f{feature_id}"
                if col_name not in df.columns:
                    continue

                series = df[col_name]
                mean = series.mean()
                std = series.std()

                if std == 0:
                    continue  # Skip if no variation

                z_scores = np.abs((series - mean) / std)
                
                Z_SCORE_THRESHOLD = ASSET_FEATURE_Z_SCORE_THRESHOLDS.get(
                    (feature_name, asset),
                    FEATURE_Z_SCORE_THRESHOLDS.get(feature_name, DEFAULT_Z_SCORE_THRESHOLD)
                )

                extreme_outliers = z_scores > Z_SCORE_THRESHOLD

                if extreme_outliers.any():
                    outlier_timestamps = series[extreme_outliers].index
                    num_outliers = len(outlier_timestamps)

                    if num_outliers > 0:
                        worst_z = z_scores[extreme_outliers].max()
                        if iteration == 0 or num_outliers > 10:
                            logging.info(f"  Iteration {iteration+1}: {col_name}: Removing {num_outliers} outliers (worst z={worst_z:.1f}, threshold={Z_SCORE_THRESHOLD})")

                    df.loc[outlier_timestamps, col_name] = np.nan
                    cols_corrected.add(col_name)
                    outlier_count += num_outliers
                    iteration_outliers += num_outliers

        if iteration_outliers == 0:
            logging.info(f"No more outliers found after {iteration + 1} iteration(s)")
            break
        elif iteration < MAX_ITERATIONS - 1:
            logging.info(f"Iteration {iteration + 1}: Found {iteration_outliers} additional outliers")

    # Second pass: Fill the NaN values we just created
    if cols_corrected:
        cols_to_fill = list(cols_corrected)
        logging.info(f"Filling {outlier_count} outlier values across {len(cols_corrected)} columns...")
        
        for col in cols_to_fill:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')

        remaining_nans = df[cols_to_fill].isna().sum().sum()
        if remaining_nans > 0:
            df[cols_to_fill] = df[cols_to_fill].ffill()
            
            remaining_nans_after_ffill = df[cols_to_fill].isna().sum().sum()
            if remaining_nans_after_ffill > 0:
                logging.warning(f"{remaining_nans_after_ffill} values at start/end of data. Using backward-fill.")
                df[cols_to_fill] = df[cols_to_fill].bfill()

            final_nans = df[cols_to_fill].isna().sum().sum()
            if final_nans > 0:
                logging.error(f"ERROR: Still have {final_nans} NaN values after interpolation and filling!")
                df[cols_to_fill] = df[cols_to_fill].fillna(0)

    logging.info(f"Artifact correction complete. Corrected {outlier_count} extreme outliers using per-feature thresholds.")
    return df

# --- MAIN SANITIZATION PIPELINE ---

def sanitize_dataframe(df, warmup_minutes=60):
    """
    Complete sanitization pipeline for feature data using the definitive workflow.

    This function:
    1. Applies a warmup period to skip potentially corrupt initial data.
    2. Performs a pre-sanitization pass to handle infinities and zero/negative prices.
    3. Detects and neutralizes statistical outliers using a configurable z-score methodology.
    4. Performs a final sanitization pass to fill any remaining NaN values.

    Args:
        df: The raw DataFrame with feature columns.
        warmup_minutes: Number of minutes to skip at the start for warmup (default: 60).

    Returns:
        A fully sanitized DataFrame.
    """
    logging.info("Starting complete data sanitization pipeline...")
    df = df.copy()

    # Step 1: Apply warmup period if specified
    if warmup_minutes > 0:
        warmup_period = pd.Timedelta(minutes=warmup_minutes)
        initial_time = df.index.min()
        original_length = len(df)
        df = df.loc[df.index >= initial_time + warmup_period]
        removed_count = original_length - len(df)
        if removed_count > 0:
            logging.info(f"Applied {warmup_minutes}-minute warmup: removed {removed_count} samples from start")

    # Step 2: First perform general sanitization (handle inf, zero/negative prices)
    logging.info("Applying initial data sanitization...")
    df = sanitize_features(df)

    # Step 3: THEN detect and remove statistical outliers
    df = handle_known_artifacts(df)

    # Step 4: Final sanitization pass to ensure no NaNs remain
    logging.info("Applying final data sanitization and filling remaining NaNs...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    logging.info("Data sanitization pipeline complete.")
    return df