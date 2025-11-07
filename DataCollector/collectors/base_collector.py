# collectors/base_collector.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import logging
import time
import threading
from dataclasses import dataclass, field
from collections import deque
import pandas as pd

from config.settings import CollectionConfig
from storage.parquet_writer import ParquetWriter
from utils.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

@dataclass
class DataBuffer:
    """Thread-safe data buffer with configurable size limits"""
    data: deque = field(default_factory=deque)
    max_size: int = 10000
    lock: threading.RLock = field(default_factory=threading.RLock)
    
    def append(self, item: Dict[str, Any]):
        """Add item to buffer"""
        with self.lock:
            self.data.append(item)
            if len(self.data) > self.max_size:
                self.data.popleft()  # Remove oldest item
    
    def extend(self, items: List[Dict[str, Any]]):
        """Add multiple items to buffer"""
        with self.lock:
            self.data.extend(items)
            # Trim to max size
            while len(self.data) > self.max_size:
                self.data.popleft()
    
    def get_and_clear(self) -> List[Dict[str, Any]]:
        """Get all items and clear buffer"""
        with self.lock:
            items = list(self.data)
            self.data.clear()
            return items
    
    def get_copy(self) -> List[Dict[str, Any]]:
        """Get copy of buffer without clearing"""
        with self.lock:
            return list(self.data)
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.data)
    
    def is_near_full(self, threshold: float = 0.8) -> bool:
        """Check if buffer is near capacity"""
        with self.lock:
            return len(self.data) >= (self.max_size * threshold)

@dataclass 
class CollectionMetrics:
    """Track data collection metrics"""
    trades_collected: int = 0
    orderbook_updates: int = 0
    liquidations_collected: int = 0
    instruments_collected: int = 0
    
    total_messages: int = 0
    errors: int = 0
    duplicates_filtered: int = 0
    
    collection_start_time: Optional[float] = None
    last_activity_time: Optional[float] = None
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity_time = time.time()
        if self.collection_start_time is None:
            self.collection_start_time = time.time()
    
    def get_uptime_seconds(self) -> float:
        """Get collection uptime in seconds"""
        if self.collection_start_time is None:
            return 0.0
        return time.time() - self.collection_start_time
    
    def get_rates(self) -> Dict[str, float]:
        """Get collection rates per second"""
        uptime = self.get_uptime_seconds()
        if uptime == 0:
            return {}
            
        return {
            'trades_per_second': self.trades_collected / uptime,
            'orderbook_updates_per_second': self.orderbook_updates / uptime,
            'liquidations_per_second': self.liquidations_collected / uptime,
            'instruments_per_second': self.instruments_collected / uptime, 
            'total_messages_per_second': self.total_messages / uptime,
            'error_rate': self.errors / max(self.total_messages, 1),
        }

class BaseCollector(ABC):
    """Abstract base class for market data collectors"""
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.writer = ParquetWriter(config.storage)
        self.ws_manager: Optional[WebSocketManager] = None
        
        # Data buffers
        self.buffers = {
            'trades': DataBuffer(max_size=config.buffers.trade_buffer_size),
            'orderbook': DataBuffer(max_size=config.buffers.orderbook_buffer_size),
            'liquidations': DataBuffer(max_size=config.buffers.liquidation_buffer_size),
            'instruments': DataBuffer(max_size=config.buffers.instrument_buffer_size),
        }
        
        # Metrics and monitoring
        self.metrics = CollectionMetrics()
        self.is_running = False
        self.flush_thread: Optional[threading.Thread] = None
        self.metrics_thread: Optional[threading.Thread] = None
        
        # Deduplication tracking
        self.seen_trade_ids = set()
        self.seen_order_ids = set()
        
        # Callbacks for custom processing
        self.data_callbacks: Dict[str, List[Callable]] = {
            'trades': [],
            'orderbook': [],
            'liquidations': [],
            'instruments': [],
        }
        
        logger.info(f"Initialized {self.__class__.__name__} for symbols: {config.symbols}")
    
    @abstractmethod
    def get_websocket_url(self) -> str:
        """Get WebSocket URL for the exchange"""
        pass
    
    @abstractmethod
    def _subscribe_to_streams(self) -> None:
        """Handles the logic of subscribing to the required data streams."""
        pass
    
    @abstractmethod
    def process_message(self, message: str) -> None:
        """Process incoming WebSocket message"""
        pass
    
    def add_data_callback(self, data_type: str, callback: Callable[[List[Dict[str, Any]]], None]):
        """Add callback for processed data"""
        if data_type in self.data_callbacks:
            self.data_callbacks[data_type].append(callback)
        else:
            logger.warning(f"Unknown data type for callback: {data_type}")
    
    def start_collection(self) -> bool:
        """Start data collection"""
        if self.is_running:
            logger.warning("Collection already running")
            return True
            
        logger.info("Starting data collection...")
        
        try:
            # Initialize WebSocket manager
            self.ws_manager = WebSocketManager(self.config.websocket)
            self.ws_manager.set_callbacks(
                on_message=self._on_websocket_message,
                on_connect=self._on_websocket_connect,
                on_disconnect=self._on_websocket_disconnect,
                on_error=self._on_websocket_error
            )
            
            # Start WebSocket connection
            if not self.ws_manager.connect():
                logger.error("Failed to establish WebSocket connection")
                return False
            
            # Start background threads
            self.is_running = True
            self._start_flush_thread()
            
            if self.config.enable_metrics:
                self._start_metrics_thread()
            
            logger.info("Data collection started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start collection: {e}")
            return False
    
    def stop_collection(self):
        """Stop data collection and save remaining data"""
        if not self.is_running:
            return
            
        logger.info("Stopping data collection...")
        self.is_running = False
        
        # Disconnect WebSocket
        if self.ws_manager:
            self.ws_manager.disconnect()
        
        # Wait for threads to finish
        if self.flush_thread and self.flush_thread.is_alive():
            self.flush_thread.join(timeout=10.0)
            
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5.0)
        
        # Final flush of all buffers
        self._flush_all_buffers()
        
        # Write final metrics
        if self.config.enable_metrics:
            self._write_metrics()
        
        logger.info("Data collection stopped")
    
    def _on_websocket_connect(self):
        """Handle WebSocket connection"""
        logger.info("WebSocket connected - delegating subscription to subclass")
        self._subscribe_to_streams()
    
    def _on_websocket_disconnect(self, code: int, message: str):
        """Handle WebSocket disconnection"""
        logger.warning(f"WebSocket disconnected: {code} - {message}")
    
    def _on_websocket_error(self, ws, error):
        """Handle WebSocket error"""
        logger.error(f"WebSocket error: {error}")
        self.metrics.errors += 1
    
    def _on_websocket_message(self, ws, message: str):
        """Handle incoming WebSocket message"""
        try:
            self.metrics.total_messages += 1
            self.metrics.update_activity()
            
            # Process message in subclass
            self.process_message(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics.errors += 1
    
    def _start_flush_thread(self):
        """Start background thread for flushing buffers to disk"""
        def flush_worker():
            while self.is_running:
                try:
                    time.sleep(self.config.buffers.flush_interval_seconds)
                    
                    if not self.is_running:
                        break
                        
                    self._check_and_flush_buffers()
                    
                except Exception as e:
                    logger.error(f"Error in flush worker: {e}")
        
        self.flush_thread = threading.Thread(target=flush_worker, daemon=True, name="FlushWorker")
        self.flush_thread.start()
    
    def _start_metrics_thread(self):
        """Start background thread for metrics reporting"""
        def metrics_worker():
            while self.is_running:
                try:
                    time.sleep(self.config.quality.metrics_report_interval)
                    
                    if not self.is_running:
                        break
                        
                    self._report_metrics()
                    
                except Exception as e:
                    logger.error(f"Error in metrics worker: {e}")
        
        self.metrics_thread = threading.Thread(target=metrics_worker, daemon=True, name="MetricsWorker")
        self.metrics_thread.start()
    
    def _check_and_flush_buffers(self):
        """Check buffer levels and flush if needed"""
        for data_type, buffer in self.buffers.items():
            # Check if buffer should be flushed
            should_flush = (
                buffer.is_near_full(self.config.buffers.flush_size_threshold) or
                buffer.size() > 0  # Periodic flush even if not full
            )
            
            if should_flush:
                self._flush_buffer(data_type, buffer)
    
    def _flush_buffer(self, data_type: str, buffer: DataBuffer):
        """Flush a specific buffer to disk"""
        data = buffer.get_and_clear()
        if not data:
            return
            
        # Call any registered callbacks first
        for callback in self.data_callbacks[data_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in {data_type} callback: {e}")
        
        # Write to parquet files for each symbol
        symbols_in_data = set()
        if data:
            for record in data:
                if 'symbol' in record:
                    symbols_in_data.add(record['symbol'])
        
        # If no symbol info, use configured symbols
        if not symbols_in_data:
            symbols_in_data = set(self.config.symbols)
        
        # Write data for each symbol separately
        for symbol in symbols_in_data:
            symbol_data = [r for r in data if r.get('symbol', symbol) == symbol]
            if symbol_data:
                success = self.writer.write_data(symbol_data, data_type, symbol)
                if success:
                    logger.debug(f"Flushed {len(symbol_data)} {data_type} records for {symbol}")
                else:
                    logger.error(f"Failed to flush {data_type} data for {symbol}")
    
    def _flush_all_buffers(self):
        """Flush all buffers to disk"""
        logger.info("Flushing all remaining data...")
        for data_type, buffer in self.buffers.items():
            if buffer.size() > 0:
                self._flush_buffer(data_type, buffer)
                logger.info(f"Flushed final {data_type} buffer: {buffer.size()} records")
    
    def _report_metrics(self):
        """Report current collection metrics"""
        rates = self.metrics.get_rates()
        buffer_sizes = {k: v.size() for k, v in self.buffers.items()}
        
        # Log metrics
        logger.info(f"Collection metrics: "
                   f"trades/s={rates.get('trades_per_second', 0):.1f}, "
                   f"orderbook/s={rates.get('orderbook_updates_per_second', 0):.1f}, "
                   f"liquidations/s={rates.get('liquidations_per_second', 0):.1f}, "
                   f"errors={self.metrics.errors}, "
                   f"buffers={buffer_sizes}")
        
        # Check for issues
        if rates.get('error_rate', 0) > 0.01:  # >1% error rate
            logger.warning(f"High error rate: {rates['error_rate']:.2%}")
        
        # Check buffer levels
        for data_type, size in buffer_sizes.items():
            buffer = self.buffers[data_type]
            if buffer.is_near_full(0.9):  # >90% full
                logger.warning(f"{data_type} buffer near full: {size}/{buffer.max_size}")
    
    def _write_metrics(self):
        """Write metrics to metadata file"""
        try:
            rates = self.metrics.get_rates()
            ws_health = self.ws_manager.get_health_info() if self.ws_manager else {}
            buffer_sizes = {k: v.size() for k, v in self.buffers.items()}
            
            metrics_data = {
                'timestamp': time.time(),
                'collection_metrics': {
                    'trades_collected': self.metrics.trades_collected,
                    'orderbook_updates': self.metrics.orderbook_updates,
                    'liquidations_collected': self.metrics.liquidations_collected,
                    'instruments_collected': self.metrics.instruments_collected,
                    'total_messages': self.metrics.total_messages,
                    'errors': self.metrics.errors,
                    'duplicates_filtered': self.metrics.duplicates_filtered,
                    'uptime_seconds': self.metrics.get_uptime_seconds(),
                },
                'rates': rates,
                'websocket_health': ws_health,
                'buffer_sizes': buffer_sizes,
                'symbols': self.config.symbols,
                'exchange': self.config.exchange,
            }
            
            self.writer.write_metadata(metrics_data)
            
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current collection status"""
        rates = self.metrics.get_rates()
        buffer_sizes = {k: v.size() for k, v in self.buffers.items()}
        
        status = {
            'is_running': self.is_running,
            'symbols': self.config.symbols,
            'exchange': self.config.exchange,
            'uptime_seconds': self.metrics.get_uptime_seconds(),
            'metrics': {
                'trades_collected': self.metrics.trades_collected,
                'orderbook_updates': self.metrics.orderbook_updates,
                'liquidations_collected': self.metrics.liquidations_collected,
                'instruments_collected': self.metrics.instruments_collected,
                'total_messages': self.metrics.total_messages,
                'errors': self.metrics.errors,
                'duplicates_filtered': self.metrics.duplicates_filtered,
            },
            'rates': rates,
            'buffer_sizes': buffer_sizes,
        }
        
        if self.ws_manager:
            status['websocket'] = self.ws_manager.get_health_info()
        
        return status
    
    # Utility methods for subclasses
    def _add_trade(self, trade_data: Dict[str, Any]):
        """Add trade data to buffer with deduplication"""
        # Deduplication by trade ID
        trade_id = trade_data.get('trdMatchID')
        if trade_id and trade_id in self.seen_trade_ids:
            self.metrics.duplicates_filtered += 1
            return
            
        if trade_id:
            self.seen_trade_ids.add(trade_id)
            # Prevent memory leak - keep only recent trade IDs
            if len(self.seen_trade_ids) > 100000:
                # Remove oldest 10% of IDs (crude but effective)
                ids_to_remove = list(self.seen_trade_ids)[:10000]
                for old_id in ids_to_remove:
                    self.seen_trade_ids.discard(old_id)
        
        self.buffers['trades'].append(trade_data)
        self.metrics.trades_collected += 1
    
    def _add_orderbook_update(self, orderbook_data: Dict[str, Any]):
        """Add orderbook update to buffer"""
        self.buffers['orderbook'].append(orderbook_data)
        self.metrics.orderbook_updates += 1
    
    def _add_liquidation(self, liquidation_data: Dict[str, Any]):
        """Add liquidation data to buffer with deduplication"""
        # Deduplication by order ID
        order_id = liquidation_data.get('orderID')
        if order_id and order_id in self.seen_order_ids:
            self.metrics.duplicates_filtered += 1
            return
            
        if order_id:
            self.seen_order_ids.add(order_id)
            # Prevent memory leak
            if len(self.seen_order_ids) > 50000:
                ids_to_remove = list(self.seen_order_ids)[:5000]
                for old_id in ids_to_remove:
                    self.seen_order_ids.discard(old_id)
        
        self.buffers['liquidations'].append(liquidation_data)
        self.metrics.liquidations_collected += 1
    
    def _add_instrument_data(self, instrument_data: Dict[str, Any]):
        """Add instrument data to buffer"""
        self.buffers['instruments'].append(instrument_data)
        self.metrics.instruments_collected += 1