# plot_raw_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend for batch processing
plt.switch_backend('Agg')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# --- Configuration (matching create_features.py) ---
BASE_RAW_DATA_DIR = Path("./data/mar_raw_data")
RUN_ID_TO_PROCESS = "run_20250920_000825"
ANALYSIS_OUTPUT_DIR = Path("./analysis")

# Symbols to process (matching create_features.py)
SYMBOLS_TO_PROCESS = [
    "XBTUSD", "ETHUSD", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "SUIUSDT"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define metrics to plot for each data type
METRICS_CONFIG = {
    'trades': {
        'columns': ['price', 'size', 'tickDirection'],
        'numeric_columns': ['price', 'size']
    },
    'orderbook': {
        'columns': ['price', 'size', 'action', 'side'],
        'numeric_columns': ['price', 'size']
    },
    'instruments': {
        'columns': ['openInterest', 'fundingRate', 'indicativeSettlePrice', 'markPrice'],
        'numeric_columns': ['openInterest', 'fundingRate', 'indicativeSettlePrice', 'markPrice']
    },
    'liquidations': {
        'columns': ['orderQty', 'price'],
        'numeric_columns': ['orderQty', 'price']
    }
}

def create_output_dirs():
    """Create output directory structure."""
    ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)
    for data_type in METRICS_CONFIG.keys():
        (ANALYSIS_OUTPUT_DIR / data_type).mkdir(exist_ok=True)
    logging.info(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")

def plot_timeseries(df: pd.DataFrame, metric: str, symbol: str, date_str: str, data_type: str):
    """
    Create timeseries plot for a given metric.

    Args:
        df: DataFrame with timestamp index and metric column
        metric: Name of the metric column to plot
        symbol: Asset symbol
        date_str: Date string (YYYYMMDD)
        data_type: Type of data (trades, orderbook, etc.)
    """
    if metric not in df.columns or df[metric].isna().all():
        logging.warning(f"Metric {metric} not found or all NaN for {symbol} on {date_str}")
        return

    try:
        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot the timeseries
        df[metric].plot(ax=ax, linewidth=0.5, alpha=0.7)

        ax.set_title(f'{symbol} - {metric} (Timeseries) - {date_str}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestamp', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f"Mean: {df[metric].mean():.4f}\n"
        stats_text += f"Std: {df[metric].std():.4f}\n"
        stats_text += f"Min: {df[metric].min():.4f}\n"
        stats_text += f"Max: {df[metric].max():.4f}\n"
        stats_text += f"Count: {len(df[metric]):,}"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9, family='monospace')

        plt.tight_layout()

        # Save figure
        output_path = ANALYSIS_OUTPUT_DIR / data_type / f"{symbol}_{metric}_timeseries_{date_str}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        logging.debug(f"Saved timeseries plot: {output_path}")

    except Exception as e:
        logging.error(f"Error plotting timeseries for {symbol} {metric}: {e}")
        plt.close('all')

def plot_distribution(df: pd.DataFrame, metric: str, symbol: str, date_str: str, data_type: str):
    """
    Create distribution plot (histogram + KDE) for a given metric.

    Args:
        df: DataFrame with metric column
        metric: Name of the metric column to plot
        symbol: Asset symbol
        date_str: Date string (YYYYMMDD)
        data_type: Type of data (trades, orderbook, etc.)
    """
    if metric not in df.columns or df[metric].isna().all():
        logging.warning(f"Metric {metric} not found or all NaN for {symbol} on {date_str}")
        return

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Remove inf and nan values
        clean_data = df[metric].replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean_data) == 0:
            logging.warning(f"No valid data for {symbol} {metric} on {date_str}")
            plt.close(fig)
            return

        # Histogram with KDE
        ax1.hist(clean_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)

        # Add KDE if we have enough data points
        if len(clean_data) > 10:
            try:
                clean_data.plot.kde(ax=ax1, color='red', linewidth=2, label='KDE')
                ax1.legend()
            except Exception as e:
                logging.debug(f"Could not plot KDE for {metric}: {e}")

        ax1.set_title(f'{symbol} - {metric} (Distribution)', fontsize=14, fontweight='bold')
        ax1.set_xlabel(metric, fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(clean_data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_title(f'{symbol} - {metric} (Box Plot)', fontsize=14, fontweight='bold')
        ax2.set_ylabel(metric, fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add statistics text box
        q25, q50, q75 = clean_data.quantile([0.25, 0.5, 0.75])
        stats_text = f"Mean: {clean_data.mean():.4f}\n"
        stats_text += f"Std: {clean_data.std():.4f}\n"
        stats_text += f"Median: {q50:.4f}\n"
        stats_text += f"Q25: {q25:.4f}\n"
        stats_text += f"Q75: {q75:.4f}\n"
        stats_text += f"Min: {clean_data.min():.4f}\n"
        stats_text += f"Max: {clean_data.max():.4f}\n"
        stats_text += f"Count: {len(clean_data):,}\n"
        stats_text += f"Skew: {clean_data.skew():.4f}"

        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9, family='monospace')

        plt.tight_layout()

        # Save figure
        output_path = ANALYSIS_OUTPUT_DIR / data_type / f"{symbol}_{metric}_distribution_{date_str}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        logging.debug(f"Saved distribution plot: {output_path}")

    except Exception as e:
        logging.error(f"Error plotting distribution for {symbol} {metric}: {e}")
        plt.close('all')

def process_data_type(run_data_dir: Path, data_type: str, date_parts: tuple, symbol: str):
    """
    Process a specific data type for a given symbol and date.

    Args:
        run_data_dir: Path to the run directory
        data_type: Type of data (trades, orderbook, instruments, liquidations)
        date_parts: Tuple of (year, month, day)
        symbol: Asset symbol
    """
    year, month, day = date_parts
    date_str = f"{year}{month:02d}{day:02d}"

    try:
        # Load data with filters
        filters = [('year', '==', year), ('month', '==', month), ('day', '==', day), ('symbol', '==', symbol)]
        data_path = run_data_dir / data_type

        if not data_path.exists():
            logging.warning(f"Data path does not exist: {data_path}")
            return

        df = pd.read_parquet(data_path, filters=filters)

        if df.empty:
            logging.info(f"No {data_type} data found for {symbol} on {date_str}")
            return

        # Convert timestamp to datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df = df.set_index('timestamp')

        logging.info(f"Processing {data_type} for {symbol} on {date_str} ({len(df):,} rows)")

        # Get numeric columns for this data type
        numeric_columns = METRICS_CONFIG[data_type]['numeric_columns']

        # Plot each numeric metric
        for metric in numeric_columns:
            if metric in df.columns:
                plot_timeseries(df, metric, symbol, date_str, data_type)
                plot_distribution(df, metric, symbol, date_str, data_type)

        # Handle categorical columns separately
        for col in df.columns:
            if col not in numeric_columns and df[col].dtype in ['object', 'category']:
                # Create count plot for categorical variables
                try:
                    value_counts = df[col].value_counts()

                    # Skip if no data
                    if len(value_counts) == 0:
                        logging.debug(f"Skipping empty categorical column {col} for {symbol}")
                        continue

                    fig, ax = plt.subplots(figsize=(12, 6))
                    value_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
                    ax.set_title(f'{symbol} - {col} (Value Counts) - {date_str}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(col, fontsize=12)
                    ax.set_ylabel('Count', fontsize=12)
                    ax.grid(True, alpha=0.3, axis='y')

                    # Only rotate labels if there are any
                    if len(ax.get_xticklabels()) > 0:
                        plt.xticks(rotation=45, ha='right')

                    plt.tight_layout()

                    output_path = ANALYSIS_OUTPUT_DIR / data_type / f"{symbol}_{col}_counts_{date_str}.png"
                    plt.savefig(output_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)

                    logging.debug(f"Saved categorical plot: {output_path}")
                except Exception as e:
                    logging.error(f"Error plotting categorical data {col}: {e}")
                    plt.close('all')

    except Exception as e:
        logging.error(f"Error processing {data_type} for {symbol} on {date_str}: {e}")

def main():
    """Main processing function."""
    create_output_dirs()

    run_dir = BASE_RAW_DATA_DIR / RUN_ID_TO_PROCESS

    if not run_dir.exists():
        logging.error(f"FATAL: Run directory not found at {run_dir}")
        return

    logging.info(f"Starting data analysis for run: {RUN_ID_TO_PROCESS}")

    # Discover all available dates
    all_day_paths = run_dir.glob("trades/year=*/month=*/day=*")
    unique_dates = sorted(list(set(
        (int(p.parts[-3].split('=')[1]), int(p.parts[-2].split('=')[1]), int(p.parts[-1].split('=')[1]))
        for p in all_day_paths
    )))

    if not unique_dates:
        logging.error("FATAL: No day-partitioned data found in the 'trades' directory.")
        return

    logging.info(f"Discovered {len(unique_dates)} days to process: {unique_dates}")
    logging.info(f"Processing {len(SYMBOLS_TO_PROCESS)} symbols: {SYMBOLS_TO_PROCESS}")

    # Calculate total number of tasks for progress bar
    total_tasks = len(unique_dates) * len(SYMBOLS_TO_PROCESS) * len(METRICS_CONFIG)

    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        for date_tuple in unique_dates:
            year, month, day = date_tuple
            date_str = f"{year}{month:02d}{day:02d}"
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing date: {date_str}")
            logging.info(f"{'='*60}")

            for symbol in SYMBOLS_TO_PROCESS:
                for data_type in METRICS_CONFIG.keys():
                    process_data_type(run_dir, data_type, date_tuple, symbol)
                    pbar.update(1)

    logging.info("\n" + "="*60)
    logging.info("Analysis Complete!")
    logging.info(f"All plots saved to: {ANALYSIS_OUTPUT_DIR.absolute()}")
    logging.info("="*60)

if __name__ == "__main__":
    main()
