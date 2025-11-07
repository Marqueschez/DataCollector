#!/usr/bin/env python3
# main_collector.py
"""
Thermodynamic Market Data Collector
Main entry point for collecting high-frequency market data for thermodynamic analysis.
"""

import argparse
import logging
import signal
import sys
import time
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

# Local imports
from config.settings import CollectionConfig
from collectors.bitmex_collector import BitMEXCollector
from storage.parquet_writer import ParquetWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_collector.log')
    ]
)

logger = logging.getLogger(__name__)

class DataCollectionManager:
    """Manages the data collection process"""
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.collector: Optional[BitMEXCollector] = None
        self.is_running = False
        self.start_time: Optional[float] = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum} - initiating graceful shutdown...")
        self.stop()
    
    def start(self) -> bool:
        """Start data collection"""
        if self.is_running:
            logger.warning("Collection already running")
            return True
            
        logger.info("=" * 60)
        logger.info("THERMODYNAMIC MARKET DATA COLLECTOR - STARTING")
        logger.info("=" * 60)
        logger.info(f"Exchange: {self.config.exchange}")
        logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"Data directory: {self.config.storage.data_dir.absolute()}")
        logger.info(f"Streams: trades={self.config.collect_trades}, "
                   f"orderbook={self.config.collect_orderbook}, "
                   f"liquidations={self.config.collect_liquidations}, "
                   f"instruments={self.config.collect_instruments}")
        
        try:
            # Validate configuration
            self.config.validate()
            
            # Initialize collector based on exchange
            if self.config.exchange.lower() == 'bitmex':
                self.collector = BitMEXCollector(self.config)
            else:
                raise ValueError(f"Unsupported exchange: {self.config.exchange}")
            
            # Start collection
            success = self.collector.start_collection()
            if not success:
                logger.error("Failed to start data collection")
                return False
            
            self.is_running = True
            self.start_time = time.time()
            
            logger.info("Data collection started successfully!")
            logger.info("Press Ctrl+C to stop collection and save data")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start collection: {e}")
            return False
    
    def stop(self):
        """Stop data collection gracefully"""
        if not self.is_running:
            return
            
        logger.info("Stopping data collection...")
        self.is_running = False
        
        if self.collector:
            self.collector.stop_collection()
        
        # Calculate collection duration
        if self.start_time:
            duration = time.time() - self.start_time
            hours = duration / 3600
            logger.info(f"Collection ran for {hours:.1f} hours")
        
        # Final status report
        if self.collector:
            final_status = self.collector.get_status()
            logger.info("Final collection statistics:")
            logger.info(f"  Trades: {final_status['metrics']['trades_collected']:,}")
            logger.info(f"  Orderbook updates: {final_status['metrics']['orderbook_updates']:,}")
            logger.info(f"  Liquidations: {final_status['metrics']['liquidations_collected']:,}")
            logger.info(f"  Instruments: {final_status['metrics']['instruments_collected']:,}")
            logger.info(f"  Total messages: {final_status['metrics']['total_messages']:,}")
            logger.info(f"  Errors: {final_status['metrics']['errors']:,}")
            logger.info(f"  Duplicates filtered: {final_status['metrics']['duplicates_filtered']:,}")
        
        logger.info("Data collection stopped successfully!")
    
    def run_monitoring_loop(self):
        """Run main monitoring loop"""
        try:
            last_report_time = time.time()
            report_interval = 300  # Report every 5 minutes
            
            while self.is_running:
                time.sleep(10)  # Check every 10 seconds
                
                if not self.is_running:
                    break
                
                current_time = time.time()
                
                # Periodic status reports
                if current_time - last_report_time >= report_interval:
                    self._print_status_report()
                    last_report_time = current_time
                
                # Check collector health
                if self.collector and not self.collector.ws_manager.is_connected():
                    logger.warning("WebSocket disconnected - monitoring reconnection...")
                
        except KeyboardInterrupt:
            logger.info("Monitoring loop interrupted")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    def _print_status_report(self):
        """Print periodic status report in a clear and correct format."""
        if not self.collector:
            return
            
        status = self.collector.get_status()
        metrics = status.get('metrics', {})
        rates = status.get('rates', {})
        buffer_sizes = status.get('buffer_sizes', {})
        
        logger.info("=" * 50)
        logger.info("COLLECTION STATUS REPORT")
        logger.info("=" * 50)
        logger.info(f"Uptime: {status.get('uptime_seconds', 0)/3600:.1f} hours")
        logger.info(f"WebSocket: {status.get('websocket', {}).get('state', 'unknown')}")
        
        logger.info("Data collected:")
        # Explicitly print each metric to avoid errors
        trades_count = metrics.get('trades_collected', 0)
        trades_rate = rates.get('trades_per_second', 0)
        logger.info(f"  Trades: {trades_count:,} ({trades_rate:.1f}/sec)")

        orderbook_count = metrics.get('orderbook_updates', 0)
        orderbook_rate = rates.get('orderbook_updates_per_second', 0)
        logger.info(f"  Orderbook Updates: {orderbook_count:,} ({orderbook_rate:.1f}/sec)")

        liquidations_count = metrics.get('liquidations_collected', 0)
        liquidations_rate = rates.get('liquidations_per_second', 0)
        logger.info(f"  Liquidations: {liquidations_count:,} ({liquidations_rate:.1f}/sec)")

        instruments_count = metrics.get('instruments_collected', 0)
        instruments_rate = rates.get('instruments_per_second', 0)
        logger.info(f"  Instruments: {instruments_count:,} ({instruments_rate:.1f}/sec)")
        
        logger.info("Buffer status:")
        for buffer_type, size in buffer_sizes.items():
            logger.info(f"  {buffer_type.title()}: {size:,} records")
        
        errors_count = metrics.get('errors', 0)
        if errors_count > 0:
            error_rate = rates.get('error_rate', 0)
            logger.warning(f"Errors: {errors_count:,} ({error_rate:.2%} rate)")


def create_config_from_args(args) -> CollectionConfig:
    """Create configuration from a JSON file and override with command line arguments."""
    
    # Start with the default configuration object
    config = CollectionConfig.from_env()

    # If a config file is provided, load it and update the config object
    if args.config_file and Path(args.config_file).exists():
        logger.info(f"Loading configuration from {args.config_file}...")
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Update config with values from the file
        if 'exchange' in config_dict:
            config.exchange = config_dict['exchange']
        if 'symbols' in config_dict:
            config.symbols = config_dict['symbols']
        if 'storage' in config_dict and 'data_dir' in config_dict['storage']:
            config.storage.data_dir = Path(config_dict['storage']['data_dir'])
        
        # Update data stream toggles from file
        config.collect_trades = config_dict.get('collect_trades', config.collect_trades)
        config.collect_orderbook = config_dict.get('collect_orderbook', config.collect_orderbook)
        config.collect_liquidations = config_dict.get('collect_liquidations', config.collect_liquidations)
        config.collect_instruments = config_dict.get('collect_instruments', config.collect_instruments)

        # Update performance settings from file
        if 'buffers' in config_dict:
            if 'trade_buffer_size' in config_dict['buffers']:
                 config.buffers.trade_buffer_size = config_dict['buffers']['trade_buffer_size']
            if 'orderbook_buffer_size' in config_dict['buffers']:
                 config.buffers.orderbook_buffer_size = config_dict['buffers']['orderbook_buffer_size']
            if 'flush_interval_seconds' in config_dict['buffers']:
                 config.buffers.flush_interval_seconds = config_dict['buffers']['flush_interval_seconds']
        
        if 'log_level' in config_dict:
            config.log_level = config_dict['log_level'].upper()

    # Now, override with any provided command line arguments (they have the highest priority)
    if args.symbols is not None:
        config.symbols = args.symbols.split(',')
    elif not config.symbols: # If no symbols from file or command line, use a default
        config.symbols = ['XBTUSD']    
    
    if args.data_dir:
        config.storage.data_dir = Path(args.data_dir)
    
    if args.exchange:
        config.exchange = args.exchange
    
    # Data stream toggles
    if args.no_trades:
        config.collect_trades = False
    if args.no_orderbook:
        config.collect_orderbook = False
    if args.no_liquidations:
        config.collect_liquidations = False
    if args.no_instruments:
        config.collect_instruments = False
    
    # Performance settings
    if args.buffer_size:
        config.buffers.trade_buffer_size = args.buffer_size
        config.buffers.orderbook_buffer_size = args.buffer_size * 5
    
    if args.flush_interval:
        config.buffers.flush_interval_seconds = args.flush_interval
    
    # Logging level
    if args.log_level:
        config.log_level = args.log_level.upper()
    
    logging.getLogger().setLevel(getattr(logging, config.log_level))
    
    return config

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Thermodynamic Market Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all data for Bitcoin
  python main_collector.py --symbols XBTUSD
  
  # Collect trades and liquidations only for multiple symbols
  python main_collector.py --symbols XBTUSD,ETHUSD --no-orderbook --no-instruments
  
  # Use custom data directory with higher buffer sizes
  python main_collector.py --data-dir /data/crypto --buffer-size 50000
  
  # Load configuration from file
  python main_collector.py --config config/production.json
        """
    )
    
    # Configuration options
    parser.add_argument('--config-file', '-c', help='Path to JSON configuration file')
    parser.add_argument('--exchange', default='bitmex', choices=['bitmex'], 
                       help='Exchange to collect from (default: bitmex)')
    parser.add_argument('--symbols', '-s', default=None,
                       help='Comma-separated list of symbols (e.g., "XBTUSD,ETHUSD")')
    parser.add_argument('--data-dir', '-d', help='Data directory path')
    
    # Data stream options
    parser.add_argument('--no-trades', action='store_true', help='Disable trade collection')
    parser.add_argument('--no-orderbook', action='store_true', help='Disable orderbook collection')
    parser.add_argument('--no-liquidations', action='store_true', help='Disable liquidation collection')
    parser.add_argument('--no-instruments', action='store_true', help='Disable instrument collection')
    
    # Performance options
    parser.add_argument('--buffer-size', type=int, help='Buffer size for trades (others scaled accordingly)')
    parser.add_argument('--flush-interval', type=int, help='Buffer flush interval in seconds')
    
    # Operational options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='Logging level')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration and exit')
    parser.add_argument('--status-only', action='store_true', help='Show status of existing collection and exit')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = create_config_from_args(args)

        # Create a unique name for this run, e.g., "run_20250918_221530"
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # The actual data will be stored one level deeper, inside this unique folder
        run_data_dir = config.storage.data_dir / run_id
        config.storage.data_dir = run_data_dir
        logger.info(f"Data for this run will be saved to: {run_data_dir.absolute()}")
        
        if args.dry_run:
            logger.info("Configuration validation successful!")
            logger.info(f"Exchange: {config.exchange}")
            logger.info(f"Symbols: {config.symbols}")
            logger.info(f"Data directory: {config.storage.data_dir}")
            logger.info("Dry run completed - no data collection performed")
            return 0
        
        if args.status_only:
            # Show status of existing data files
            writer = ParquetWriter(config.storage)
            # Implementation would show existing file stats
            logger.info("Status check not yet implemented")
            return 0
        
        # Start collection
        manager = DataCollectionManager(config)
        
        if not manager.start():
            logger.error("Failed to start data collection")
            return 1
        
        # Run monitoring loop
        manager.run_monitoring_loop()
        
        # Cleanup
        manager.stop()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())