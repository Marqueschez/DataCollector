# BitMEX Market Data Collection & Feature Engineering Pipeline

A complete end-to-end system for collecting high-frequency cryptocurrency market data from BitMEX, engineering stationary features for machine learning, and discovering market regimes using unsupervised learning. This pipeline is designed to support quantitative trading research and Multi-Asset Regime (MAR) model development.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Data Collection (BitMEX)](#data-collection-bitmex)
- [Feature Engineering](#feature-engineering)
- [Regime Discovery](#regime-discovery)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Schemas](#data-schemas)
- [Usage Examples](#usage-examples)
- [Performance](#performance)

---

## Overview

This project implements a three-stage pipeline:

1. **Real-time Data Collection**: WebSocket-based collection of trades, orderbook updates, liquidations, and instrument data from BitMEX
2. **Feature Engineering**: Transform raw market data into stationary features suitable for machine learning models
3. **Regime Discovery**: Unsupervised clustering to identify distinct market regimes for regime-adaptive trading strategies

### Key Features

- ✅ **High-frequency data capture** at microsecond precision
- ✅ **Full orderbook reconstruction** from Level-2 delta updates
- ✅ **Stationary feature engineering** with no information leakage
- ✅ **Unsupervised regime discovery** using UMAP + HDBSCAN
- ✅ **Production-ready** with robust error handling and monitoring
- ✅ **Optimized storage** using partitioned Parquet format
- ✅ **Multi-asset support** with configurable symbols

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BitMEX WebSocket Feed                        │
│          (trades, orderbook, liquidations, instruments)         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Stage 1: Data Collection                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  WebSocket  │→ │ Thread-Safe  │→ │  Parquet Storage   │    │
│  │   Manager   │  │   Buffers    │  │  (Partitioned by   │    │
│  │             │  │              │  │   date & type)     │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                  Parquet Files (Raw Data)
        ┌──────────┬──────────┬──────────┬──────────┐
        │  trades/ │orderbook/│liquidat../│instrum../│
        └──────────┴──────────┴──────────┴──────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               Stage 2: Feature Engineering                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │  Orderbook   │→ │  Feature     │→ │  Feature CSVs    │     │
│  │Reconstruction│  │ Calculation  │  │ (100ms samples)  │     │
│  │  (L2 deltas) │  │ (7 features) │  │                  │     │
│  └──────────────┘  └──────────────┘  └──────────────────┘     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                Feature CSVs (7 features per asset)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 3: Regime Discovery                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │ Aggregate to │→ │ UMAP Reduce  │→ │ HDBSCAN Cluster  │     │
│  │  10min bins  │  │ (15D → 2D)   │  │  + Hierarchical  │     │
│  │              │  │              │  │     Merging      │     │
│  └──────────────┘  └──────────────┘  └──────────────────┘     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
              Feature CSVs with Regime Labels
                  (Ready for MAR model training)
```

---

## Data Collection (BitMEX)

### Collected Data Types

The collector captures **four distinct data streams** via WebSocket:

#### 1. **Trades** (`trade` stream)
Real-time trade executions with aggressor identification.

**Captured Fields:**
- `timestamp`: Execution time (UTC, float64)
- `trdMatchID`: Unique trade identifier (string)
- `symbol`: Contract symbol (category: XBTUSD, ETHUSD, etc.)
- `price`: Execution price (float32)
- `size`: Contract quantity (int32)
- `homeNotional`: BTC-denominated value (float64)
- `foreignNotional`: USD-denominated value (float64)
- `side`: Aggressor side - 'Buy' or 'Sell' (category)
- `tickDirection`: Price movement - 'PlusTick', 'ZeroPlusTick', 'MinusTick', 'ZeroMinusTick' (category)
- `grossValue`: Trade value in Satoshis (int64)

**Use Case:** Measure market impact, order flow toxicity, price discovery

#### 2. **Level-2 Orderbook** (`orderBookL2` stream)
Full depth orderbook updates with incremental deltas.

**Captured Fields:**
- `timestamp`: Processing timestamp (float64)
- `id`: Unique level identifier (int64)
- `symbol`: Contract symbol (category)
- `side`: 'Buy' (bid) or 'Sell' (ask) (category)
- `price`: Price level (float32)
- `size`: Aggregate quantity at level (int32)
- `action`: Update type - 'partial' (snapshot), 'insert' (new level), 'update' (size change), 'delete' (level removed) (category)

**Processing:** Full orderbook state is reconstructed from delta updates to calculate:
- Best bid/ask prices
- Mid-price evolution
- Weighted average price (WAP) across depth
- Order book imbalance (OBI)

**Use Case:** Liquidity analysis, spread dynamics, orderbook pressure

#### 3. **Liquidations** (`liquidation` stream)
Forced position closures due to insufficient margin.

**Captured Fields:**
- `timestamp`: Liquidation timestamp (float64)
- `orderID`: Unique order identifier (string)
- `symbol`: Contract symbol (category)
- `side`: 'Sell' = long liquidation (bearish), 'Buy' = short liquidation (bullish) (category)
- `orderQty`: Original order quantity (int32)
- `price`: Liquidation price (float32)
- `leavesQty`: Actual liquidated quantity (int32)

**Use Case:** Market stress detection, cascade risk assessment, forced seller identification

#### 4. **Instruments** (`instrument` stream)
Contract-level metrics and funding data.

**Captured Fields:**
- `timestamp`: Update timestamp (float64)
- `symbol`: Contract symbol (category)
- `openInterest`: Total open contracts (int64)
- `fundingRate`: Current 8-hour funding rate (float64)
- `indicativeFundingRate`: Next period funding rate estimate (float64)
- `fundingTimestamp`: Next funding time (float64)
- `markPrice`: Mark price for margin calculations (float32)
- `indexPrice`: Underlying spot index price (float32)
- `volume24h`: 24-hour trading volume (int64)

**Use Case:** Funding rate arbitrage, open interest momentum, basis spread analysis

### Data Collection Features

#### 🚀 **High Performance**
- **Thread-safe concurrent collection**: Independent buffers per data type
- **Memory efficient**: Configurable buffers (10K trades, 50K orderbook updates, 5K liquidations)
- **Automatic flushing**: Time-based (30s) and size-based (80% full) triggers
- **Low latency**: <1ms from WebSocket to buffer

#### 🔧 **Production Ready**
- **Robust reconnection**: Exponential backoff with jitter (max 60s delay)
- **Graceful shutdown**: Signal handling (SIGINT, SIGTERM) for clean saves
- **Deduplication**: Automatic removal of duplicate records via trade/order IDs
- **Schema validation**: Type enforcement and range checks

#### 💾 **Optimized Storage**
- **Parquet format**: Columnar storage with Snappy compression
- **Partitioned by date**: `year=YYYY/month=MM/day=DD/` structure
- **Hourly file rotation**: Manageable file sizes for parallel processing
- **~100MB/day** compressed storage per symbol

### File Organization

```
data/
└── run_YYYYMMDD_HHMMSS/
    ├── trades/
    │   └── year=2025/month=01/day=20/
    │       └── symbol=XBTUSD/
    │           ├── data_00.parquet  (Hour 00:00-00:59)
    │           ├── data_01.parquet  (Hour 01:00-01:59)
    │           └── ...
    ├── orderbook/
    │   └── year=2025/month=01/day=20/
    │       └── symbol=XBTUSD/
    │           └── data_00.parquet
    ├── liquidations/
    │   └── year=2025/month=01/day=20/
    │       └── symbol=XBTUSD/
    │           └── data.parquet  (All day)
    └── instruments/
        └── year=2025/month=01/day=20/
            └── symbol=XBTUSD/
                └── data.parquet  (All day)
```

---

## Feature Engineering

The `create_features.py` script transforms raw market data into **7 stationary features** per asset at 100ms resolution.

### Design Philosophy: Transform & Scale

**This script:** Transform to stationarity (log returns, ratios, etc.)
**Training script:** Apply StandardScaler (Z-score normalization) fit only on train data

**Rationale:**
- Preserves true magnitude of extreme events (5-sigma moves)
- No information leakage from static bounds
- Allows model to adapt to changing market regimes
- Captures full distribution for robust scaling

### Feature Set (7 Features per Asset)

#### Feature 0: **Price vs. Slow EMA (5min)**
```python
wap_ema_5min = wap.ewm(span=3000).mean()  # 5min at 100ms = 3000 samples
price_vs_ema = (wap / wap_ema_5min) - 1
```
- **Type:** Mean-reverting signal
- **Range:** Typically ±0.5% (±0.005)
- **Use:** Trend identification, mean-reversion opportunities
- **Stationary:** Yes (ratio deviation from moving average)

#### Feature 1: **Mid-Price Log Return**
```python
mid_price = (best_bid + best_ask) / 2
mid_price_log_return = log(mid_price_t / mid_price_t-1)
```
- **Type:** Instantaneous return
- **Range:** Typically ±0.001 (±0.1%)
- **Use:** PnL calculation, momentum
- **Stationary:** Yes (first-difference of log prices)

#### Feature 2: **Relative Spread**
```python
bid_ask_spread = (ask_L1 - bid_L1) / mid_price
basis_spread = abs((markPrice - indexPrice) / indexPrice)
relative_spread = bid_ask_spread + basis_spread
```
- **Type:** Liquidity measure
- **Range:** 0.0001 to 0.01 (1-100 bps)
- **Use:** Liquidity cost estimation, market stress
- **Stationary:** Yes (ratio, bounded near zero)

#### Feature 3: **Multi-Level Cumulative OBI**
```python
total_bid_size = sum(bid_size_L1 to L15)
total_ask_size = sum(ask_size_L1 to L15)
obi = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
```
- **Type:** Orderbook pressure
- **Range:** -1 to +1
- **Use:** Directional bias, orderbook imbalance
- **Stationary:** Yes (bounded ratio)

#### Feature 4: **Taker Imbalance**
```python
buy_volume_weighted = buy_volume * tick_direction_weight
sell_volume_weighted = sell_volume * tick_direction_weight
imbalance = buy_volume_weighted - sell_volume_weighted
taker_imbalance = sign(imbalance) * log1p(abs(imbalance))
```
- **Type:** Order flow toxicity
- **Range:** Varies (signed log scale)
- **Use:** Aggressive buyer/seller identification
- **Stationary:** Yes (signed log transform)
- **Note:** Uses `foreignNotional` (USD value) weighted by tick direction aggressiveness

#### Feature 5: **Realized Volatility (dt)**
```python
log_returns_window = log_returns[-50:]  # 5s window at 100ms
realized_volatility = std(log_returns_window)
```
- **Type:** Short-term volatility
- **Range:** 0.0001 to 0.01 (typical)
- **Use:** Risk sizing, regime detection
- **Stationary:** Yes (rolling statistic)
- **Window:** 5 seconds (50 samples at 100ms)

#### Feature 6: **Multi-Level WAP**
```python
wap_numerator = sum(bid_price_Li * bid_size_Li + ask_price_Li * ask_size_Li) for i in 1..15
wap_denominator = sum(bid_size_Li + ask_size_Li) for i in 1..15
wap = wap_numerator / wap_denominator
```
- **Type:** Volume-weighted mid-price
- **Range:** Similar to mid-price (USD)
- **Use:** True execution price estimation
- **Stationary:** No (use Feature 0 for stationary version)

### Additional Columns

- **`{symbol}_volume24h`**: 24-hour volume from instrument feed (informational)
- **`{symbol}_mid_price`**: True mid-price from reconstructed orderbook

### Orderbook Reconstruction

The orderbook is fully reconstructed from Level-2 delta updates:

1. **Partial snapshot**: Initialize book state
2. **Insert**: Add new price level
3. **Update**: Modify size at existing level
4. **Delete**: Remove price level

```python
# Reconstruct mid-price handling multiple updates per 100ms interval
mid_price_updates = reconstruct_orderbook(orderbook_df)  # Every tick
mid_price_filled = mid_price_updates.ffill()  # Forward fill
mid_price_resampled = mid_price_filled.resample('100ms').last()  # Take LAST in interval
```

**Key Insight:** When multiple orderbook updates occur within a 100ms interval, use the **LAST** (most recent) mid-price as the true end-of-interval value.

### Output Format

**File:** `features_output_YYYYMMDD.csv`

**Columns (per symbol):**
```
timestamp,
XBTUSD_f0, XBTUSD_f1, XBTUSD_f2, XBTUSD_f3, XBTUSD_f4, XBTUSD_f5, XBTUSD_f6, XBTUSD_volume24h, XBTUSD_mid_price,
ETHUSD_f0, ETHUSD_f1, ETHUSD_f2, ETHUSD_f3, ETHUSD_f4, ETHUSD_f5, ETHUSD_f6, ETHUSD_volume24h, ETHUSD_mid_price,
...
```

**Sample Rate:** 100ms (10 Hz)
**Format:** CSV with microsecond timestamps

### Performance

- **Processing speed:** ~10,000 samples/second (single core)
- **Parallel processing:** Configurable worker count (default: CPU count - 1)
- **Memory usage:** <2GB for full day processing
- **Optimization:** Numba JIT compilation for rolling volatility, vectorized operations

---

## Regime Discovery

The `discover_regimes.py` script identifies **distinct market regimes** using unsupervised learning.

### Algorithm: UMAP + HDBSCAN + Hierarchical Merging

```
Feature CSVs (100ms, 7 features × 6 assets)
         ↓
Aggregate to 10-minute segments (mean, std, min, max)
         ↓
Robust scaling (RobustScaler - resistant to outliers)
         ↓
UMAP dimensionality reduction (42D → 2D)
  • n_neighbors=15, min_dist=0.1
         ↓
HDBSCAN clustering
  • min_cluster_size=30, min_samples=10
         ↓
Hierarchical merging of similar regimes
  • Compute regime centroids
  • Hierarchical clustering on centroids
  • Merge to target_n_regimes=5
         ↓
Regime labels (0, 1, 2, 3, 4) added to CSVs
```

### Features Used for Clustering

Only **market microstructure features** (excludes target and PnL):
- Multi-Level WAP (f6)
- Relative Spread (f2)
- Multi-Level Cumulative OBI (f3)
- Taker Imbalance (f4)
- Realized Volatility (f5)

**Excluded:** Price vs. EMA (f0) - target feature, Mid-Price Log Return (f1) - PnL feature

### Why This Approach?

1. **UMAP:** Preserves both local and global structure (better than PCA/t-SNE)
2. **HDBSCAN:** Finds clusters of varying density, handles noise (-1 label)
3. **Hierarchical merging:** Reduces regime count to interpretable number (5 regimes)
4. **10-min aggregation:** Reduces noise, captures regime persistence
5. **RobustScaler:** Resistant to outliers in crypto markets

### Configuration

```python
AGGREGATION_INTERVAL = "10min"       # Segment duration
UMAP_N_NEIGHBORS = 15                # UMAP: local structure
UMAP_MIN_DIST = 0.1                  # UMAP: point spacing
HDBSCAN_MIN_CLUSTER_SIZE = 30        # HDBSCAN: min points per cluster
HDBSCAN_MIN_SAMPLES = 10             # HDBSCAN: core point threshold
TARGET_N_REGIMES = 5                 # Final regime count after merging
```

### Output

**Files:** Same filenames as input, saved to `mar_feature_data_with_regimes/`

**New column:** `regime` (int: 0, 1, 2, 3, 4)

**Visualizations:** Saved to `analysis/` directory:
- `regime_tsne.png`: 2D UMAP projection colored by regime
- `regime_stats.png`: Feature distributions per regime
- `regime_transitions.png`: Temporal evolution of regimes

### Regime Interpretation

Each regime represents a distinct market microstructure state:

- **Regime 0:** Low volatility, tight spreads (normal market)
- **Regime 1:** High volatility, wide spreads (stressed market)
- **Regime 2:** Bid pressure (orderbook imbalance toward buyers)
- **Regime 3:** Ask pressure (orderbook imbalance toward sellers)
- **Regime 4:** High taker imbalance (directional flow)

**Note:** Actual regime semantics depend on the data and should be validated with domain knowledge.

---

## Installation

### Requirements

- Python 3.9+
- 16GB+ RAM (for regime discovery on large datasets)
- 100GB+ disk space (for raw data storage)

### Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0         # Parquet I/O
websocket-client>=1.6.0 # WebSocket
numba>=0.57.0           # JIT optimization
sortedcontainers>=2.4.0 # Fast orderbook reconstruction
scikit-learn>=1.3.0     # Feature scaling
umap-learn>=0.5.3       # Dimensionality reduction
hdbscan>=0.8.29         # Clustering
matplotlib>=3.7.0       # Visualization
seaborn>=0.12.0         # Visualization
tqdm>=4.65.0            # Progress bars
```

---

## Quick Start

### 1. Collect Real-time Data

```bash
# Collect Bitcoin data (default)
python main_collector.py

# Collect multiple symbols
python main_collector.py --symbols XBTUSD,ETHUSD,SOLUSDT

# Custom data directory
python main_collector.py --data-dir /mnt/data/bitmex

# Collect for 24 hours then exit
python main_collector.py --duration 86400

# Trades and liquidations only (no orderbook)
python main_collector.py --no-orderbook --no-instruments
```

**Output:** Parquet files in `data/run_YYYYMMDD_HHMMSS/`

### 2. Engineer Features

Edit `create_features.py` configuration:

```python
BASE_RAW_DATA_DIR = Path("./data/mar_raw_data")
RUN_ID_TO_PROCESS = "run_20250920_000825"  # Your run ID
FEATURE_OUTPUT_DIR = Path("./mar_feature_data")

SYMBOLS_TO_PROCESS = [
    "XBTUSD", "ETHUSD", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"
]
```

Run feature engineering:

```bash
python create_features.py
```

**Output:** Feature CSVs in `mar_feature_data/features_output_YYYYMMDD.csv`

### 3. Discover Regimes

```bash
python discover_regimes.py
```

**Output:**
- Feature CSVs with regime labels in `mar_feature_data_with_regimes/`
- Visualizations in `analysis/`

### 4. Train MAR Model

Load features with regime labels:

```python
import pandas as pd

df = pd.read_csv('mar_feature_data_with_regimes/features_output_20250920.csv',
                 index_col=0, parse_dates=True)

# Separate features and regime labels
regime_labels = df['regime']
features = df.drop(columns=['regime'])

# Apply StandardScaler (fit on train only!)
from sklearn.preprocessing import StandardScaler

train_mask = df.index < '2025-09-25'
test_mask = df.index >= '2025-09-25'

scaler = StandardScaler()
scaler.fit(features[train_mask])

X_train = scaler.transform(features[train_mask])
X_test = scaler.transform(features[test_mask])
y_train = regime_labels[train_mask]
y_test = regime_labels[test_mask]
```

---

## Configuration

### Data Collection Config

**File:** `config/settings.py`

```python
@dataclass
class CollectionConfig:
    # Exchange
    exchange: str = "bitmex"
    symbols: List[str] = ["XBTUSD", "ETHUSD"]

    # Data streams
    collect_trades: bool = True
    collect_orderbook: bool = True
    collect_liquidations: bool = True
    collect_instruments: bool = True

    # Buffers
    buffers: BufferConfig = BufferConfig(
        trade_buffer_size=10_000,
        orderbook_buffer_size=50_000,
        liquidation_buffer_size=5_000,
        flush_interval_seconds=30,
        flush_size_threshold=0.8
    )

    # Storage
    storage: StorageConfig = StorageConfig(
        data_dir=Path("data"),
        file_rotation_hours=1,
        compression="snappy"
    )
```

### Feature Engineering Config

**File:** `create_features.py` (top of file)

```python
RUN_ID_TO_PROCESS = "run_20250920_000825"
SAMPLING_INTERVAL = "100ms"
LOB_DEPTH_FOR_FEATURES = 15  # Orderbook levels to use
VOLATILITY_WINDOW_STR = "5s"  # Rolling volatility window
PERFORMANCE_MODE = True  # Disable progress bars for speed
NUM_WORKERS = max(1, cpu_count() - 1)  # Parallel workers
```

### Regime Discovery Config

**File:** `discover_regimes.py` (top of file)

```python
AGGREGATION_INTERVAL = "10min"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 10
TARGET_N_REGIMES = 5
ENABLE_REGIME_MERGING = True
```

---

## Data Schemas

### Trades Parquet Schema

```
timestamp: float64 (UTC seconds)
trdMatchID: string (unique ID)
symbol: category (e.g., 'XBTUSD')
price: float32
size: int32
homeNotional: float64 (BTC value)
foreignNotional: float64 (USD value)
side: category ('Buy', 'Sell')
tickDirection: category ('PlusTick', 'ZeroPlusTick', 'MinusTick', 'ZeroMinusTick')
grossValue: int64 (Satoshis)
year: int32 (partition)
month: int32 (partition)
day: int32 (partition)
```

### Orderbook Parquet Schema

```
timestamp: float64 (UTC seconds)
id: int64 (unique level ID)
symbol: category
side: category ('Buy', 'Sell')
price: float32
size: int32
action: category ('partial', 'insert', 'update', 'delete')
year: int32 (partition)
month: int32 (partition)
day: int32 (partition)
```

### Liquidations Parquet Schema

```
timestamp: float64 (UTC seconds)
orderID: string (unique ID)
symbol: category
side: category ('Sell'=long liq, 'Buy'=short liq)
orderQty: int32
price: float32
leavesQty: int32 (actual liquidated qty)
year: int32 (partition)
month: int32 (partition)
day: int32 (partition)
```

### Instruments Parquet Schema

```
timestamp: float64 (UTC seconds)
symbol: category
openInterest: int64
fundingRate: float64
indicativeFundingRate: float64
fundingTimestamp: float64
markPrice: float32
indexPrice: float32
volume24h: int64
year: int32 (partition)
month: int32 (partition)
day: int32 (partition)
```

### Feature CSV Schema

```
timestamp: datetime64[ns] (index)
{SYMBOL}_f0: float64 (Price vs. Slow EMA)
{SYMBOL}_f1: float64 (Mid-Price Log Return)
{SYMBOL}_f2: float64 (Relative Spread)
{SYMBOL}_f3: float64 (Multi-Level Cumulative OBI)
{SYMBOL}_f4: float64 (Taker Imbalance)
{SYMBOL}_f5: float64 (Realized Volatility)
{SYMBOL}_f6: float64 (Multi-Level WAP)
{SYMBOL}_volume24h: float64 (24h volume)
{SYMBOL}_mid_price: float64 (reconstructed mid-price)
regime: int32 (0-4, added by discover_regimes.py)
```

---

## Usage Examples

### Load and Analyze Trade Data

```python
import pandas as pd

# Load hourly trades
trades = pd.read_parquet('data/run_20250920_000825/trades/year=2025/month=09/day=20/symbol=XBTUSD/')

trades['timestamp'] = pd.to_datetime(trades['timestamp'], unit='s')
trades = trades.set_index('timestamp')

# Calculate trade statistics
print(f"Total trades: {len(trades):,}")
print(f"Volume (USD): ${trades['foreignNotional'].sum():,.0f}")
print(f"Price range: ${trades['price'].min():.0f} - ${trades['price'].max():.0f}")

# Aggressive buyer/seller analysis
buy_aggression = trades[trades['side'] == 'Buy']['foreignNotional'].sum()
sell_aggression = trades[trades['side'] == 'Sell']['foreignNotional'].sum()
print(f"Buy/Sell ratio: {buy_aggression/sell_aggression:.2f}")
```

### Reconstruct Orderbook Snapshot

```python
# Load orderbook updates
orderbook = pd.read_parquet('data/run_20250920_000825/orderbook/year=2025/month=09/day=20/symbol=XBTUSD/')

# Filter to specific timestamp
target_time = pd.Timestamp('2025-09-20 14:30:00')
snapshot = orderbook[orderbook['timestamp'] <= target_time.timestamp()]

# Get current state
bids = snapshot[(snapshot['side'] == 'Buy') & (snapshot['action'] != 'delete')]
asks = snapshot[(snapshot['side'] == 'Sell') & (snapshot['action'] != 'delete')]

best_bid = bids['price'].max()
best_ask = asks['price'].min()
print(f"Spread: {best_ask - best_bid:.2f} ({(best_ask/best_bid - 1)*10000:.1f} bps)")
```

### Analyze Liquidations

```python
# Load liquidations
liquidations = pd.read_parquet('data/run_20250920_000825/liquidations/year=2025/month=09/day=20/')

liquidations['timestamp'] = pd.to_datetime(liquidations['timestamp'], unit='s')
liquidations = liquidations.set_index('timestamp')

# Long vs short liquidations
long_liqs = liquidations[liquidations['side'] == 'Sell']
short_liqs = liquidations[liquidations['side'] == 'Buy']

print(f"Long liquidations: {len(long_liqs):,} (${long_liqs['leavesQty'].sum():,.0f})")
print(f"Short liquidations: {len(short_liqs):,} (${short_liqs['leavesQty'].sum():,.0f})")

# Hourly liquidation volume
hourly_liqs = liquidations.resample('1H')['leavesQty'].sum()
print(hourly_liqs)
```

### Load Features for ML

```python
# Load feature CSVs with regime labels
df = pd.read_csv('mar_feature_data_with_regimes/features_output_20250920.csv',
                 index_col=0, parse_dates=True)

# Extract features for a specific asset
xbt_features = df[[col for col in df.columns if col.startswith('XBTUSD_f')]]

# Analyze regime distribution
regime_counts = df['regime'].value_counts().sort_index()
print("Regime distribution:")
print(regime_counts)
print(f"\nRegime percentages:")
print((regime_counts / len(df) * 100).round(1))
```

---

## Performance

### Data Collection Benchmarks

**XBTUSD (typical day):**
- Trades: ~500,000 executions
- Orderbook updates: ~5,000,000 deltas
- Liquidations: ~1,000 events
- Storage: ~100MB compressed

**Latency:**
- WebSocket to buffer: <1ms
- Buffer to disk: <100ms (batched)
- End-to-end: <150ms (p95)

**Resource usage:**
- Memory: <1GB
- CPU: <10% (single core)
- Network: ~2 Mbps

### Feature Engineering Benchmarks

**6 assets, 1 day, ~860K samples:**
- Processing time: ~5 minutes (8 cores)
- Memory usage: ~2GB peak
- Output size: ~50MB CSV

**Optimization:**
- Numba JIT: 10x speedup on rolling volatility
- Vectorization: 5x speedup on WAP calculation
- Parallel processing: Linear scaling with cores

### Regime Discovery Benchmarks

**30 days, 6 assets, ~25M samples → 43K segments:**
- UMAP: ~2 minutes
- HDBSCAN: ~30 seconds
- Hierarchical merging: <1 second
- Total: ~3 minutes

---

## Troubleshooting

### Data Collection Issues

**Connection drops:**
```
Solution: Check network stability, firewall settings
The collector auto-reconnects with exponential backoff
```

**Missing orderbook data:**
```
Solution: Verify 'partial' snapshot received on connection
Check logs for "Received partial snapshot for {symbol}"
```

**Duplicate trades:**
```
Solution: Deduplication is automatic via trdMatchID
Check buffer flush frequency if memory usage high
```

### Feature Engineering Issues

**Zero mid-prices:**
```
Solution: Ensure orderbook data exists for the date
Check reconstruct_book_snapshots_optimized() logs
Fallback: Uses median price from valid samples
```

**High memory usage:**
```
Solution: Reduce NUM_WORKERS or process fewer symbols
Process one day at a time instead of batching
```

**Infinities/NaNs in features:**
```
Solution: Script automatically replaces inf→0, NaN→0
Check logs for "Infinities detected" warnings
Verify input data quality with inspect_raw_data.py
```

### Regime Discovery Issues

**Too many regimes (>25):**
```
Solution: Increase HDBSCAN_MIN_CLUSTER_SIZE
Enable ENABLE_REGIME_MERGING = True
Adjust TARGET_N_REGIMES to desired count
```

**High noise percentage (>30%):**
```
Solution: Reduce HDBSCAN_MIN_SAMPLES
Increase AGGREGATION_INTERVAL (e.g., '15min')
Check data quality (missing values, outliers)
```

**Regime imbalance (one regime >80%):**
```
Solution: Adjust UMAP parameters (n_neighbors, min_dist)
Use longer time periods (more diverse market conditions)
Consider manual regime assignment for training
```

---

## Utility Scripts

### `inspect_raw_data.py`
Explore collected Parquet files:
```bash
python inspect_raw_data.py --run-id run_20250920_000825 --symbol XBTUSD
```

### `plot_raw_data.py`
Visualize raw data:
```bash
python plot_raw_data.py --run-id run_20250920_000825 --date 20250920
```

### `add_mid_price_column.py`
**Deprecated** - now integrated into `create_features.py`

Previously used to retroactively add mid-price columns to existing feature CSVs.

---

## Advanced Configuration

### Environment Variables

```bash
# Override default settings
export DATA_DIR=/mnt/data/bitmex
export SYMBOLS=XBTUSD,ETHUSD,SOLUSDT
export LOG_LEVEL=DEBUG

python main_collector.py
```

### Custom Callbacks

Add real-time processing hooks:

```python
from collectors.bitmex_collector import BitMEXCollector

def on_large_trade(trades):
    for trade in trades:
        if abs(trade['foreignNotional']) > 1_000_000:  # $1M+ trade
            print(f"🐋 Whale trade: ${trade['foreignNotional']:,.0f} @ ${trade['price']:,.0f}")

collector = BitMEXCollector(config)
collector.add_data_callback('trades', on_large_trade)
collector.start_collection()
```

### Multi-Day Processing

Process multiple days in sequence:

```python
from pathlib import Path
from create_features import process_day

run_dir = Path("data/run_20250920_000825")
dates = [(2025, 9, 20), (2025, 9, 21), (2025, 9, 22)]

prev_day_features = None
for year, month, day in dates:
    result = process_day(run_dir, (year, month, day), SYMBOLS, prev_day_features)
    if result is not None:
        # Keep last 240 minutes for warmup
        prev_day_features = result.tail(240 * 10 * 60)  # 240min at 100ms
```

---

## Integration with MAR Model

### Expected Workflow

1. **Collect data** for desired time period (e.g., 30 days)
2. **Engineer features** with `create_features.py`
3. **Discover regimes** with `discover_regimes.py`
4. **Split train/test** by date (e.g., 80/20 split)
5. **Fit StandardScaler** on training data only
6. **Train MAR model** with regime-conditioned parameters
7. **Backtest** on test data with regime-aware strategy

### Regime-Adaptive Trading

```python
# Example: Regime-specific strategy parameters
regime_params = {
    0: {'position_size': 1.0, 'stop_loss': 0.01},  # Normal market
    1: {'position_size': 0.5, 'stop_loss': 0.02},  # High volatility
    2: {'position_size': 1.2, 'stop_loss': 0.01},  # Bid pressure
    3: {'position_size': 1.2, 'stop_loss': 0.01},  # Ask pressure
    4: {'position_size': 0.8, 'stop_loss': 0.015}, # High flow
}

current_regime = df['regime'].iloc[-1]
params = regime_params[current_regime]
```

---

## License

This project is designed for academic research and quantitative trading development. Ensure compliance with BitMEX API terms of service and data usage policies.

---

## Contributing

Contributions welcome! Priority areas:

- Additional exchange support (Binance, Bybit, OKX)
- Real-time feature calculation (streaming mode)
- GPU-accelerated regime discovery
- Enhanced regime interpretability (feature importance)
- Backtesting integration

---

## Support

**Issues:**
- Data collection: Check WebSocket connectivity, buffer sizes
- Feature engineering: Verify raw data integrity with `inspect_raw_data.py`
- Regime discovery: Adjust HDBSCAN/UMAP parameters, check segment count

**Contact:**
For research collaboration or technical questions, please open a GitHub issue.

---

## Acknowledgments

This pipeline builds on:
- BitMEX WebSocket API for real-time data
- UMAP (McInnes et al.) for dimensionality reduction
- HDBSCAN (McInnes et al.) for density-based clustering
- Numba for JIT optimization

Designed for Multi-Asset Regime (MAR) model research in quantitative finance.
