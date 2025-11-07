# config/settings.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import os

@dataclass
class WebSocketConfig:
    """WebSocket connection configuration"""
    url: str = "wss://ws.bitmex.com/realtime"
    ping_interval: int = 30
    ping_timeout: int = 10
    reconnect_attempts: int = 5
    reconnect_delay_base: float = 1.0  # Base delay for exponential backoff
    reconnect_delay_max: float = 60.0  # Maximum delay between reconnect attempts

@dataclass 
class BufferConfig:
    """In-memory buffer configuration"""
    trade_buffer_size: int = 10_000
    orderbook_buffer_size: int = 50_000
    liquidation_buffer_size: int = 5_000
    instrument_buffer_size: int = 1_000
    
    # Flush triggers
    flush_interval_seconds: int = 30
    flush_size_threshold: float = 0.8  # Flush when buffer is 80% full

@dataclass
class StorageConfig:
    """Parquet storage configuration"""
    data_dir: Path = field(default_factory=lambda: Path("data"))
    
    # File organization
    file_rotation_hours: int = 1  # Create new file every hour
    partition_by_date: bool = True
    
    # Parquet optimization
    compression: str = "snappy"
    row_group_size: int = 50_000
    page_size: int = 8192
    use_dictionary: bool = True
    write_statistics: bool = True
    
    # Schema validation
    enforce_schema: bool = True
    validate_on_write: bool = True

@dataclass
class QualityConfig:
    """Data quality monitoring configuration"""
    max_latency_ms: int = 100
    min_trades_per_minute: int = 10
    max_price_change_pct: float = 5.0  # Flag suspiciously large price moves
    duplicate_trade_threshold: int = 10  # Alert if too many duplicates
    
    # Monitoring intervals
    metrics_report_interval: int = 60  # Report metrics every 60 seconds
    health_check_interval: int = 10    # Health check every 10 seconds

@dataclass
class CollectionConfig:
    """Main configuration class"""
    # Exchange settings
    exchange: str = "bitmex"
    symbols: List[str] = field(default_factory=lambda: ["XBTUSD"])
    
    # Component configs
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    buffers: BufferConfig = field(default_factory=BufferConfig) 
    storage: StorageConfig = field(default_factory=StorageConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # Data streams to collect
    collect_trades: bool = True
    collect_orderbook: bool = True
    collect_liquidations: bool = True
    collect_instruments: bool = True
    
    # Operational settings
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_health_monitoring: bool = True
    
    @classmethod
    def from_env(cls) -> 'CollectionConfig':
        """Create config from environment variables"""
        config = cls()
        
        # Override from environment
        if os.getenv('DATA_DIR'):
            config.storage.data_dir = Path(os.getenv('DATA_DIR'))
        
        if os.getenv('SYMBOLS'):
            config.symbols = os.getenv('SYMBOLS').split(',')
            
        if os.getenv('LOG_LEVEL'):
            config.log_level = os.getenv('LOG_LEVEL')
            
        return config
    
    def validate(self) -> None:
        """Validate configuration"""
        assert len(self.symbols) > 0, "Must specify at least one symbol"
        assert self.buffers.flush_size_threshold < 1.0, "Flush threshold must be < 1.0"
        
        # Create data directory if it doesn't exist (don't require absolute path)
        try:
            self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create data directory {self.storage.data_dir}: {e}")

# Data type schemas
TRADE_SCHEMA = {
    'timestamp': 'float64',
    'trdMatchID': 'string',
    'symbol': 'category',
    'price': 'float32',
    'size': 'int32', 
    'homeNotional': 'float64',
    'foreignNotional': 'float64',
    'side': 'category',
    'tickDirection': 'category',
    'grossValue': 'int64',
}

ORDERBOOK_SCHEMA = {
    'timestamp': 'float64',
    'id': 'int64',
    'symbol': 'category',
    'side': 'category',
    'price': 'float32',
    'size': 'int32',
    'action': 'category',
}

LIQUIDATION_SCHEMA = {
    'timestamp': 'float64',
    'orderID': 'string',
    'symbol': 'category',
    'side': 'category',
    'orderQty': 'int32',
    'price': 'float32',
    'leavesQty': 'int32',
    'cumQty': 'int32',
    'ordType': 'category',
    'timeInForce': 'category',
}

INSTRUMENT_SCHEMA = {
    'timestamp': 'float64',
    'symbol': 'category',
    'openInterest': 'int64',
    'fundingRate': 'float64',
    'indicativeFundingRate': 'float64',
    'fundingTimestamp': 'float64',
    'markPrice': 'float32',
    'indexPrice': 'float32',
    'settlementPrice': 'float32',
    'volume24h': 'int64',
    'turnover24h': 'float64',
}

# Quality thresholds by data type
QUALITY_THRESHOLDS = {
    'trades': {
        'min_per_minute': 10,
        'max_price_change_pct': 5.0,
        'max_latency_ms': 100,
    },
    'orderbook': {
        'min_updates_per_minute': 100,
        'max_spread_pct': 1.0,
        'max_latency_ms': 50,
    },
    'liquidations': {
        'max_per_minute': 1000,  # Alert if too many liquidations
        'min_size': 1,
    },
    'instruments': {
        'max_funding_rate': 0.01,  # 1% per 8 hours
        'min_update_interval': 1,  # At least every second
    }
}