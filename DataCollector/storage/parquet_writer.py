# storage/parquet_writer.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import threading
import time

from config.settings import StorageConfig, TRADE_SCHEMA, ORDERBOOK_SCHEMA, LIQUIDATION_SCHEMA, INSTRUMENT_SCHEMA

logger = logging.getLogger(__name__)

@dataclass
class WriteStats:
    """Statistics for write operations"""
    records_written: int = 0
    files_created: int = 0
    total_size_bytes: int = 0
    write_duration_ms: float = 0.0
    last_write_time: Optional[datetime] = None

class ParquetWriter:
    """High-performance parquet writer with schema validation and partitioning"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.write_lock = threading.RLock()
        self.stats = {
            'trades': WriteStats(),
            'orderbook': WriteStats(), 
            'liquidations': WriteStats(),
            'instruments': WriteStats(),
        }
        
        # Schema mappings
        self.schemas = {
            'trades': TRADE_SCHEMA,
            'orderbook': ORDERBOOK_SCHEMA,
            'liquidations': LIQUIDATION_SCHEMA,
            'instruments': INSTRUMENT_SCHEMA,
        }
        
        # Ensure data directories exist
        self._create_directories()
        
    def _create_directories(self):
        """Create required directory structure"""
        for data_type in ['trades', 'orderbook', 'liquidations', 'instruments', 'metadata']:
            dir_path = self.config.data_dir / data_type
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _get_file_path(self, data_type: str, symbol: str, timestamp: datetime) -> Path:
        """Generate partitioned file path"""
        if self.config.partition_by_date:
            # Partitioned: data/trades/year=2025/month=01/day=15/XBTUSD_trades_20250115_14.parquet
            year = timestamp.year
            month = f"{timestamp.month:02d}"
            day = f"{timestamp.day:02d}"
            hour = f"{timestamp.hour:02d}"
            
            dir_path = (self.config.data_dir / data_type / 
                       f"year={year}" / f"month={month}" / f"day={day}")
            dir_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"{symbol}_{data_type}_{timestamp.strftime('%Y%m%d')}_{hour}.parquet"
        else:
            # Flat: data/trades/XBTUSD_trades_20250115_14.parquet
            dir_path = self.config.data_dir / data_type
            filename = f"{symbol}_{data_type}_{timestamp.strftime('%Y%m%d_%H')}.parquet"
            
        return dir_path / filename
    
    def _validate_schema(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate and coerce DataFrame to expected schema"""
        if not self.config.enforce_schema:
            return df
            
        expected_schema = self.schemas.get(data_type)
        if not expected_schema:
            logger.warning(f"No schema defined for data type: {data_type}")
            return df
            
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Ensure all expected columns exist
        for col, dtype in expected_schema.items():
            if col not in df.columns:
                # Add missing columns with appropriate default values
                if dtype.startswith('int'):
                    df[col] = 0
                elif dtype.startswith('float'):
                    df[col] = 0.0
                elif dtype == 'string' or dtype == 'category':
                    df[col] = ''
                else:
                    df[col] = None
                    
        # Coerce types
        for col, dtype in expected_schema.items():
            if col in df.columns:
                try:
                    if dtype == 'category':
                        df[col] = df[col].astype('category')
                    elif dtype.startswith('int'):
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
                    elif dtype.startswith('float'):
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(dtype)
                    elif dtype == 'string':
                        df[col] = df[col].astype('string')
                except Exception as e:
                    logger.warning(f"Failed to coerce column {col} to {dtype}: {e}")
                    
        return df
    
    def _get_parquet_kwargs(self) -> Dict[str, Any]:
        """Get optimized parquet write parameters"""
        return {
            'compression': self.config.compression,
            'row_group_size': self.config.row_group_size,
            'use_dictionary': self.config.use_dictionary,
            'write_statistics': self.config.write_statistics,
            'engine': 'pyarrow',
        }
    
    def write_data(self, data: List[Dict[str, Any]], data_type: str, symbol: str) -> bool:
        """Write data to parquet file with deduplication and validation"""
        if not data:
            return True
            
        start_time = time.time()
        
        try:
            with self.write_lock:
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Add datetime column for partitioning
                if 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                else:
                    df['datetime'] = pd.Timestamp.now(tz=timezone.utc)
                
                # Validate schema
                if self.config.validate_on_write:
                    df = self._validate_schema(df, data_type)
                
                # Determine file path based on latest timestamp
                latest_time = datetime.fromtimestamp(df['timestamp'].max(), tz=timezone.utc) if len(df) > 0 and 'timestamp' in df.columns else datetime.now(timezone.utc)
                file_path = self._get_file_path(data_type, symbol, latest_time)
                
                # Handle existing file (append or merge)
                if file_path.exists():
                    df = self._merge_with_existing(df, file_path, data_type)
                
                # Sort by timestamp for optimal compression
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Write to parquet
                df.to_parquet(
                    file_path,
                    index=False,
                    **self._get_parquet_kwargs()
                )
                
                # Update statistics
                write_duration = (time.time() - start_time) * 1000  # ms
                self._update_stats(data_type, len(data), file_path.stat().st_size, write_duration)
                
                logger.debug(f"Wrote {len(data)} {data_type} records to {file_path.name} "
                            f"in {write_duration:.1f}ms")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to write {data_type} data: {e}")
            return False
    
    def _merge_with_existing(self, new_df: pd.DataFrame, file_path: Path, data_type: str) -> pd.DataFrame:
        """Merge new data with existing file, handling duplicates"""
        try:
            existing_df = pd.read_parquet(file_path)
            
            # Combine dataframes
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates based on data type
            if data_type == 'trades' and 'trdMatchID' in combined_df.columns:
                # Deduplicate by trade ID
                combined_df = combined_df.drop_duplicates(subset=['trdMatchID'], keep='last')
            elif data_type == 'orderbook':
                # Deduplicate by id and timestamp (keep latest)
                combined_df = combined_df.drop_duplicates(subset=['id', 'timestamp'], keep='last')
            elif data_type == 'liquidations' and 'orderID' in combined_df.columns:
                # Deduplicate by order ID
                combined_df = combined_df.drop_duplicates(subset=['orderID'], keep='last')
            elif data_type == 'instruments':
                # Deduplicate by symbol and timestamp
                combined_df = combined_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
            else:
                # Fallback: deduplicate by timestamp
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                
            return combined_df
            
        except Exception as e:
            logger.warning(f"Failed to merge with existing file {file_path}: {e}")
            return new_df
    
    def _update_stats(self, data_type: str, record_count: int, file_size: int, duration_ms: float):
        """Update write statistics"""
        stats = self.stats[data_type]
        stats.records_written += record_count
        stats.total_size_bytes += file_size
        stats.write_duration_ms = duration_ms
        stats.last_write_time = datetime.now(timezone.utc)
        
        # Check if this created a new file
        if record_count > 0:
            stats.files_created += 1
    
    def get_stats(self) -> Dict[str, WriteStats]:
        """Get current write statistics"""
        with self.write_lock:
            return self.stats.copy()
    
    def write_metadata(self, metadata: Dict[str, Any], filename: str = None) -> bool:
        """Write metadata/quality metrics"""
        try:
            if filename is None:
                filename = f"collection_metadata_{datetime.now(timezone.utc).strftime('%Y%m%d_%H')}.parquet"
                
            file_path = self.config.data_dir / "metadata" / filename
            
            # Convert to DataFrame if it isn't already
            if isinstance(metadata, dict):
                df = pd.DataFrame([metadata])
            else:
                df = pd.DataFrame(metadata)
                
            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = time.time()
                
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            
            # Write metadata
            df.to_parquet(file_path, index=False, **self._get_parquet_kwargs())
            
            logger.debug(f"Wrote metadata to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write metadata: {e}")
            return False
    
    def optimize_storage(self):
        """Optimize existing parquet files (compress, merge small files, etc.)"""
        logger.info("Starting storage optimization...")
        
        for data_type in ['trades', 'orderbook', 'liquidations', 'instruments']:
            data_dir = self.config.data_dir / data_type
            if not data_dir.exists():
                continue
                
            # Find all parquet files
            parquet_files = list(data_dir.rglob("*.parquet"))
            
            # Group by date and symbol for potential merging
            file_groups = {}
            for file_path in parquet_files:
                # Extract date and symbol from filename
                parts = file_path.stem.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    date = parts[2]
                    key = f"{symbol}_{date}"
                    
                    if key not in file_groups:
                        file_groups[key] = []
                    file_groups[key].append(file_path)
            
            # Merge small files (< 10MB) for the same day
            for group_key, files in file_groups.items():
                if len(files) > 1:
                    small_files = [f for f in files if f.stat().st_size < 10 * 1024 * 1024]  # 10MB
                    if len(small_files) > 3:  # Only merge if we have many small files
                        self._merge_small_files(small_files, data_type)
        
        logger.info("Storage optimization completed")
    
    def _merge_small_files(self, files: List[Path], data_type: str):
        """Merge small files into larger ones for better query performance"""
        try:
            # Read all files
            dfs = []
            for file_path in files:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            
            # Combine and deduplicate
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            if data_type == 'trades' and 'trdMatchID' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['trdMatchID'], keep='last')
            else:
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Write merged file
            merged_path = files[0].parent / f"merged_{files[0].name}"
            combined_df.to_parquet(merged_path, index=False, **self._get_parquet_kwargs())
            
            # Remove original small files
            for file_path in files:
                file_path.unlink()
                
            logger.info(f"Merged {len(files)} small {data_type} files into {merged_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to merge small files: {e}")