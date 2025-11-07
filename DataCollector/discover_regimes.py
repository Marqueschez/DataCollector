#!/usr/bin/env python3
# discover_regimes.py
"""
Discovers market regimes using HDBSCAN + UMAP on engineered features.
Adds regime labels to feature CSVs for MAR model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from data_manager import sanitize_dataframe
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
FEATURE_BASE_DIR = Path("./mar_feature_data")
FEATURE_OUTPUT_DIR = Path("./mar_feature_data_with_regimes")  # Same directory, will overwrite with regime-enhanced versions
ANALYSIS_OUTPUT_DIR = Path("./analysis")  # Directory for visualizations and analysis outputs
RUN_ID_TO_PROCESS = "run_20250920_000825"

# Regime discovery parameters
AGGREGATION_INTERVAL = "10min"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 10

# Post-processing
TARGET_N_REGIMES = 5  # ← Set final desired regime count
ENABLE_REGIME_MERGING = True  # ← Enable hierarchical merging

# Safety limits
MAX_ALLOWED_REGIMES = 25  # Allow more initially since we'll merge
MAX_NOISE_PCT = 0.3

# Features to use for regime discovery (exclude target and PnL features)
FEATURES_FOR_CLUSTERING = [
    "Multi-Level WAP",
    "Relative Spread",
    "Multi-Level Cumulative OBI",
    "Taker Imbalance",
    "Realized Volatility (dt)"
    # , "Open Interest (Log Return)",
    # "Liquidation Volume",
    # "Funding Rate"
]

def load_all_feature_files(run_id: str) -> pd.DataFrame:
    """Load all features_output_*.csv files for a run."""
    pattern = f"features_output_*.csv"
    all_files = sorted(FEATURE_BASE_DIR.glob(pattern))

    if not all_files:
        raise FileNotFoundError(f"No feature files found matching {pattern} in {FEATURE_BASE_DIR}")

    logging.info(f"Loading {len(all_files)} feature files...")

    dfs = []
    for file in tqdm(all_files, desc="Loading files"):
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        # Remove existing regime column if present (for re-runs)
        if 'regime' in df.columns:
            logging.info(f"Found existing 'regime' column in {file.name}, removing it...")
            df = df.drop(columns=['regime'])

        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0).sort_index()
    logging.info(f"Loaded {len(combined_df)} total samples from {combined_df.index[0]} to {combined_df.index[-1]}")

    return combined_df

def extract_feature_columns(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Extract specific features for all assets."""
    # Get all unique symbols
    all_cols = df.columns.tolist()
    symbols = sorted(list(set(col.split('_f')[0] for col in all_cols if '_f' in col)))
    
    # Map feature names to IDs (from your MAR_ALL_FEATURES dict)
    feature_map = {
        "Price vs. Slow EMA (5min)": 0,
        "Mid-Price (Log Return)": 1,
        "Relative Spread": 2,
        "Multi-Level Cumulative OBI": 3,
        "Taker Imbalance": 4,
        "Realized Volatility (dt)": 5,
        "Multi-Level WAP": 6,
        "Open Interest (Log Return)": 7,
        "Liquidation Volume": 8,
        "Funding Rate": 9,
    }
    
    selected_cols = []
    for symbol in symbols:
        for feat_name in feature_names:
            feat_id = feature_map[feat_name]
            col_name = f"{symbol}_f{feat_id}"
            if col_name in df.columns:
                selected_cols.append(col_name)
    
    return df[selected_cols]

def aggregate_to_segments(df: pd.DataFrame, interval: str = "10min") -> pd.DataFrame:
    """Aggregate 100ms features to longer intervals for regime discovery."""
    logging.info(f"Aggregating features to {interval} segments...")
    
    # For each feature, compute multiple statistics to capture segment characteristics
    agg_funcs = {
        col: ['mean', 'std', 'min', 'max'] 
        for col in df.columns
    }
    
    segments_df = df.resample(interval).agg(agg_funcs)
    
    # Flatten column names: 'XBTUSD_f0_mean', 'XBTUSD_f0_std', etc.
    segments_df.columns = ['_'.join(col).strip() for col in segments_df.columns.values]
    
    # Drop segments with any NaN (incomplete data)
    segments_df = segments_df.dropna()
    
    logging.info(f"Created {len(segments_df)} segments of {interval} each")
    
    return segments_df

def merge_regimes_hierarchically(
    regime_labels: np.ndarray,
    segments_df: pd.DataFrame,
    target_n_regimes: int = 5
) -> np.ndarray:
    """
    Merge similar regimes using hierarchical clustering on regime centroids.
    
    Args:
        regime_labels: Original regime assignments
        segments_df: Feature data for all segments
        target_n_regimes: Desired number of final regimes
    
    Returns:
        New regime labels with merged regimes
    """
    logging.info(f"\nMerging regimes from {len(set(regime_labels)) - (1 if -1 in regime_labels else 0)} to ~{target_n_regimes}...")
    
    # Get unique regimes (excluding noise)
    unique_regimes = sorted([r for r in set(regime_labels) if r != -1])
    
    if len(unique_regimes) <= target_n_regimes:
        logging.info("Already at or below target regime count, no merging needed")
        return regime_labels
    
    # Calculate regime centroids (mean feature values for each regime)
    regime_centroids = []
    regime_ids = []
    
    for regime_id in unique_regimes:
        regime_mask = regime_labels == regime_id
        regime_data = segments_df[regime_mask]
        centroid = regime_data.mean().values
        regime_centroids.append(centroid)
        regime_ids.append(regime_id)
    
    regime_centroids = np.array(regime_centroids)
    
    # Hierarchical clustering on regime centroids
    # Using Ward linkage (minimizes within-cluster variance)
    linkage_matrix = linkage(regime_centroids, method='ward')
    
    # Cut dendrogram to get target number of clusters
    merged_cluster_ids = fcluster(linkage_matrix, target_n_regimes, criterion='maxclust')
    
    # Create mapping from old regime IDs to new regime IDs
    old_to_new = {regime_ids[i]: merged_cluster_ids[i] - 1 for i in range(len(regime_ids))}
    
    # Apply mapping to all labels
    new_labels = regime_labels.copy()
    for old_id, new_id in old_to_new.items():
        new_labels[regime_labels == old_id] = new_id
    
    # Log merging results
    logging.info("Regime merging complete:")
    for old_id, new_id in sorted(old_to_new.items(), key=lambda x: x[1]):
        old_count = (regime_labels == old_id).sum()
        logging.info(f"  Old Regime {old_id} ({old_count} segments) → New Regime {new_id}")
    
    # Show new regime distribution
    new_regime_counts = pd.Series(new_labels[new_labels != -1]).value_counts().sort_index()
    logging.info("\nNew regime distribution:")
    for regime_id, count in new_regime_counts.items():
        pct = count / len(new_labels[new_labels != -1]) * 100
        logging.info(f"  Regime {regime_id}: {count} segments ({pct:.1f}%)")
    
    return new_labels

def discover_regimes(segments_df: pd.DataFrame, target_n_regimes: int = None) -> tuple:
    """Apply UMAP + HDBSCAN to discover regimes with safety checks."""
    logging.info("Applying dimensionality reduction and clustering...")

    # Use provided target_n_regimes or fall back to module-level default
    if target_n_regimes is None:
        target_n_regimes = TARGET_N_REGIMES
    
    # Scale features
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(segments_df)
    
    # UMAP
    logging.info("Running UMAP...")
    umap_model = UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric='euclidean',
        random_state=42
    )
    embedding_2d = umap_model.fit_transform(features_scaled)
    
    # HDBSCAN
    logging.info("Running HDBSCAN...")
    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='euclidean'
    )
    regime_labels = clusterer.fit_predict(embedding_2d)
    
    # === SAFETY CHECKS ===
    unique_regimes = set(regime_labels)
    n_regimes = len(unique_regimes) - (1 if -1 in unique_regimes else 0)
    n_noise = (regime_labels == -1).sum()
    
    logging.info(f"Initial clustering found {n_regimes} regimes")
    
    # === HIERARCHICAL MERGING ===
    if target_n_regimes and n_regimes > target_n_regimes:
        regime_labels = merge_regimes_hierarchically(
            regime_labels,
            segments_df,
            target_n_regimes
        )
        n_regimes = len(set(regime_labels)) - (1 if -1 in regime_labels else 0)
    
    # Check 1: Too many regimes
    MAX_ALLOWED_REGIMES = 20
    if n_regimes > MAX_ALLOWED_REGIMES:
        logging.error(f"HDBSCAN found {n_regimes} regimes (max allowed: {MAX_ALLOWED_REGIMES})")
        logging.error("This suggests parameters are too permissive or data quality issues.")
        logging.error(f"Try increasing HDBSCAN_MIN_CLUSTER_SIZE (current: {HDBSCAN_MIN_CLUSTER_SIZE})")
        raise ValueError(f"Too many regimes discovered: {n_regimes}")
    
    # Check 2: Too much noise
    noise_pct = n_noise / len(regime_labels)
    MAX_NOISE_PCT = 0.3
    if noise_pct > MAX_NOISE_PCT:
        logging.warning(f"High noise percentage: {noise_pct:.1%} (threshold: {MAX_NOISE_PCT:.0%})")
        logging.warning("Consider decreasing HDBSCAN_MIN_CLUSTER_SIZE or HDBSCAN_MIN_SAMPLES")
    
    # Check 3: Minimum regime sizes
    MIN_REGIME_SIZE = 5  # At least 5 segments (50 minutes) per regime
    regime_counts = pd.Series(regime_labels).value_counts()
    small_regimes = regime_counts[regime_counts < MIN_REGIME_SIZE]
    if len(small_regimes) > 0:
        logging.warning(f"Found {len(small_regimes)} regimes with fewer than {MIN_REGIME_SIZE} segments:")
        for regime_id, count in small_regimes.items():
            if regime_id != -1:  # Don't warn about noise
                logging.warning(f"  Regime {regime_id}: {count} segments")
        logging.warning("These will be merged into nearest neighbors")
        
        # Merge small regimes into their nearest cluster
        regime_labels = merge_small_regimes(regime_labels, embedding_2d, MIN_REGIME_SIZE)
        n_regimes = len(set(regime_labels)) - (1 if -1 in regime_labels else 0)
    
    # Log final statistics
    logging.info(f"Discovered {n_regimes} valid regimes")
    logging.info(f"Noise points (label -1): {n_noise} ({noise_pct:.1%})")
    
    for regime_id in sorted(set(regime_labels)):
        count = (regime_labels == regime_id).sum()
        pct = count / len(regime_labels) * 100
        label = "Noise" if regime_id == -1 else f"Regime {regime_id}"
        logging.info(f"  {label}: {count} segments ({pct:.1f}%)")
    
    return regime_labels, embedding_2d, umap_model, clusterer

def print_regime_summary(segments_df: pd.DataFrame, regime_labels: np.ndarray):
    """Print detailed summary of discovered regimes for interpretation."""
    
    segments_with_regimes = segments_df.copy()
    segments_with_regimes['regime'] = regime_labels
    
    print("\n" + "="*80)
    print("REGIME SUMMARY")
    print("="*80)
    
    for regime_id in sorted(set(regime_labels)):
        if regime_id == -1:
            continue
        
        regime_data = segments_with_regimes[segments_with_regimes['regime'] == regime_id]
        count = len(regime_data)
        duration_hours = count * 10 / 60  # 10-minute segments
        
        # Calculate sample count at 100ms frequency
        samples_per_segment = 10 * 60 * (1000 / 100)  # 6000 samples per 10-min segment
        total_samples = int(count * samples_per_segment)
        
        print(f"\nRegime {regime_id}:")
        print(f"  Duration: {count} segments ({duration_hours:.1f} hours, ~{total_samples:,} samples at 100ms)")

def merge_small_regimes(labels: np.ndarray, embedding: np.ndarray, min_size: int) -> np.ndarray:
    """Merge regimes smaller than min_size into their nearest neighbor."""
    from sklearn.metrics import pairwise_distances
    
    labels = labels.copy()
    regime_counts = pd.Series(labels).value_counts()
    
    # Find small regimes (excluding noise)
    small_regimes = [r for r, count in regime_counts.items() 
                     if count < min_size and r != -1]
    
    if not small_regimes:
        return labels
    
    # Calculate regime centroids
    valid_regimes = [r for r in set(labels) if r != -1 and r not in small_regimes]
    centroids = {r: embedding[labels == r].mean(axis=0) for r in valid_regimes}
    
    # Merge each small regime into nearest large regime
    for small_regime in small_regimes:
        small_regime_mask = labels == small_regime
        small_centroid = embedding[small_regime_mask].mean(axis=0)
        
        # Find nearest large regime
        min_dist = np.inf
        nearest_regime = None
        for regime_id, centroid in centroids.items():
            dist = np.linalg.norm(small_centroid - centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_regime = regime_id
        
        # Merge
        labels[small_regime_mask] = nearest_regime
        logging.info(f"Merged small regime {small_regime} into regime {nearest_regime}")
    
    return labels

def visualize_regimes(embedding_2d: np.ndarray, regime_labels: np.ndarray, segments_df: pd.DataFrame):
    """Create visualizations of discovered regimes."""
    logging.info("Creating regime visualizations...")

    # Create analysis directory if it doesn't exist
    ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Regime clusters in 2D embedding space
    scatter = axes[0].scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=regime_labels,
        cmap='tab10',
        alpha=0.6,
        s=20
    )
    axes[0].set_xlabel('UMAP Dimension 1')
    axes[0].set_ylabel('UMAP Dimension 2')
    axes[0].set_title('Market Regimes in Embedding Space')
    plt.colorbar(scatter, ax=axes[0], label='Regime')

    # Plot 2: Regimes over time
    regime_timeline = pd.Series(regime_labels, index=segments_df.index)
    axes[1].scatter(
        regime_timeline.index,
        regime_timeline.values,
        c=regime_timeline.values,
        cmap='tab10',
        alpha=0.6,
        s=10
    )
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Regime')
    axes[1].set_title('Regime Evolution Over Time')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save visualization to analysis directory
    output_path = ANALYSIS_OUTPUT_DIR / "regime_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logging.info(f"Saved visualization to {output_path}")
    plt.close()

def forward_fill_regime_labels(
    full_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    regime_labels: np.ndarray
) -> pd.Series:
    """Forward-fill regime labels from 10-min segments to 100ms frequency."""
    logging.info("Forward-filling regime labels to 100ms frequency...")
    
    # Create series with regime labels at segment timestamps
    regime_series = pd.Series(regime_labels, index=segments_df.index)
    
    # Reindex to full 100ms frequency and forward-fill
    regime_series_full = regime_series.reindex(full_df.index, method='ffill')
    
    # Handle any remaining NaNs at the start (use first regime)
    if regime_series_full.isna().any():
        first_valid_regime = regime_series.iloc[0]
        regime_series_full = regime_series_full.fillna(first_valid_regime)
    
    return regime_series_full.astype(int)

def analyze_regime_characteristics(segments_df: pd.DataFrame, regime_labels: np.ndarray):
    """Enhanced analysis showing all feature contributions."""
    logging.info("\n" + "="*60)
    logging.info("REGIME CHARACTERISTICS ANALYSIS")
    logging.info("="*60)
    
    segments_with_regimes = segments_df.copy()
    segments_with_regimes['regime'] = regime_labels
    
    for regime_id in sorted(set(regime_labels)):
        if regime_id == -1:
            continue
            
        regime_data = segments_with_regimes[segments_with_regimes['regime'] == regime_id]
        
        logging.info(f"\nRegime {regime_id} ({len(regime_data)} segments):")
        
        # Volatility (f5)
        vol_cols = [col for col in segments_df.columns if 'f5_mean' in col]
        if vol_cols:
            vol_mean = regime_data[vol_cols].mean().mean()
            logging.info(f"  Volatility: {vol_mean:.6e}")

        # Spread (f2)
        spread_cols = [col for col in segments_df.columns if 'f2_mean' in col]
        if spread_cols:
            spread_mean = regime_data[spread_cols].mean().mean()
            logging.info(f"  Spread: {spread_mean:.6e}")

        # OBI (f3)
        obi_cols = [col for col in segments_df.columns if 'f3_mean' in col]
        if obi_cols:
            obi_mean = regime_data[obi_cols].mean().mean()
            logging.info(f"  OBI: {obi_mean:.6e}")

        # Taker Imbalance (f4)
        taker_cols = [col for col in segments_df.columns if 'f4_mean' in col]
        if taker_cols:
            taker_mean = regime_data[taker_cols].mean().mean()
            logging.info(f"  Taker Imbalance: {taker_mean:.6e}")

        # # OI Change (f6)
        # oi_cols = [col for col in segments_df.columns if 'f6_mean' in col]
        # if oi_cols:
        #     oi_mean = regime_data[oi_cols].mean().mean()
        #     logging.info(f"  OI Change: {oi_mean:.6e}")

        # # Liquidations (f7)
        # liq_cols = [col for col in segments_df.columns if 'f7_mean' in col]
        # if liq_cols:
        #     liq_sum = regime_data[liq_cols].sum().sum()
        #     logging.info(f"  Liquidations: {liq_sum:.6e}")

        # # Funding (f8)
        # funding_cols = [col for col in segments_df.columns if 'f8_mean' in col]
        # if funding_cols:
        #     funding_mean = regime_data[funding_cols].mean().mean()
        #     logging.info(f"  Funding: {funding_mean:.6e}")

def save_enhanced_features(full_df: pd.DataFrame, regime_labels: pd.Series):
    """Save feature CSVs with regime labels prepended."""
    logging.info("Saving regime-enhanced feature files...")

    # Group by date to save daily files
    full_df['regime'] = regime_labels

    # Move regime column to be first
    cols = ['regime'] + [col for col in full_df.columns if col != 'regime']
    full_df = full_df[cols]

    # Save by date - get UNIQUE dates only
    unique_dates = sorted(set(full_df.index.date))
    for date in unique_dates:
        date_str = date.strftime("%Y%m%d")
        date_mask = full_df.index.date == date
        day_df = full_df[date_mask]

        if len(day_df) > 0:
            output_path = FEATURE_OUTPUT_DIR / f"features_output_{date_str}.csv"
            day_df.to_csv(output_path, date_format='%Y-%m-%d %H:%M:%S.%f')
            logging.info(f"Saved {len(day_df)} samples to {output_path}")

def main():
    """Main regime discovery pipeline."""
    try:
        # 1. Load all feature files
        full_features_df = load_all_feature_files(RUN_ID_TO_PROCESS)

        # 2. Sanitize the data
        logging.info("="*60)
        logging.info("SANITIZING DATA BEFORE REGIME DISCOVERY")
        logging.info("="*60)
        full_features_df = sanitize_dataframe(full_features_df)

        # 3. Extract relevant features for clustering
        clustering_features_df = extract_feature_columns(full_features_df, FEATURES_FOR_CLUSTERING).copy()

        # --- NEW: Replace absolute WAP with a stationary Rolling Percentile Rank ---
        logging.info("Transforming WAP to stationary Rolling Percentile Rank for clustering...")
        # A 1-hour window gives a good local context (1 hours * 60 min/hr * 10 samples/min = 7200)
        ROLLING_WINDOW = 6 * 60 * 10 
        wap_cols = [col for col in clustering_features_df.columns if '_f6' in col]

        # Try to import numba for JIT compilation (massive speedup)
        try:
            from numba import jit

            @jit(nopython=True)
            def fast_rolling_rank(values, window, min_periods):
                """JIT-compiled rolling percentile rank calculation."""
                n = len(values)
                result = np.empty(n)

                for i in range(n):
                    start_idx = max(0, i - window + 1)
                    window_size = i - start_idx + 1

                    if window_size >= min_periods:
                        current_value = values[i]
                        count_less = 0
                        for j in range(start_idx, i + 1):
                            if values[j] < current_value:
                                count_less += 1
                        result[i] = count_less / window_size
                    else:
                        result[i] = np.nan

                return result

            # Use JIT-compiled version (50-100x faster)
            for col in tqdm(wap_cols, desc="Calculating WAP Percentile Ranks (Numba JIT)"):
                values = clustering_features_df[col].values
                rolling_rank = fast_rolling_rank(values, ROLLING_WINDOW, ROLLING_WINDOW // 10)
                clustering_features_df[col] = rolling_rank

        except ImportError:
            logging.warning("Numba not available, falling back to slower NumPy method")
            logging.warning("Install numba with 'pip install numba' for 50-100x speedup")

            # Fallback: Use NumPy (still much faster than pandas apply)
            for col in tqdm(wap_cols, desc="Calculating WAP Percentile Ranks"):
                values = clustering_features_df[col].values
                n = len(values)
                min_periods = ROLLING_WINDOW // 10
                rolling_rank = np.full(n, np.nan)

                for i in range(n):
                    start_idx = max(0, i - ROLLING_WINDOW + 1)
                    window_size = i - start_idx + 1

                    if window_size >= min_periods:
                        window = values[start_idx:i+1]
                        current_value = values[i]
                        rank = np.sum(window < current_value) / window_size
                        rolling_rank[i] = rank

                clustering_features_df[col] = rolling_rank
        
        # Drop initial NaNs created by the rolling window
        clustering_features_df = clustering_features_df.dropna()
        logging.info(f"Stationary WAP feature created. Using {len(clustering_features_df)} rows for clustering.")
        # --- END NEW ---

        # 4. Aggregate to segments (using the now-modified dataframe)
        segments_df = aggregate_to_segments(clustering_features_df, AGGREGATION_INTERVAL)

        # 5. Discover and merge regimes to the target count
        regime_labels, embedding_2d, umap_model, clusterer = discover_regimes(
            segments_df,
            target_n_regimes=TARGET_N_REGIMES
        )

        # 6. Print summary, visualize, and analyze the final, merged regimes
        print_regime_summary(segments_df, regime_labels)
        visualize_regimes(embedding_2d, regime_labels, segments_df)
        analyze_regime_characteristics(segments_df, regime_labels)

        # 7. Forward-fill labels to 100ms frequency
        regime_labels_full = forward_fill_regime_labels(full_features_df, segments_df, regime_labels)

        # 8. Save enhanced features
        save_enhanced_features(full_features_df, regime_labels_full)
        
        logging.info("\n" + "="*60)
        logging.info("REGIME DISCOVERY COMPLETE")
        logging.info("="*60)
        logging.info(f"Enhanced feature files saved to {FEATURE_OUTPUT_DIR}")
        logging.info("Regime labels are in the first column of each CSV")
        
    except Exception as e:
        logging.error(f"Fatal error in regime discovery: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())