#!/usr/bin/env python3
# debug_test.py
"""
Simple debug script to test the data collector components step by step
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports work correctly"""
    logger.info("Testing imports...")
    
    try:
        from config.settings import CollectionConfig
        logger.info("Config import successful")
    except Exception as e:
        logger.error(f"Config import failed: {e}")
        return False
    
    try:
        from utils.websocket_manager import WebSocketManager, WebSocketConfig
        logger.info("WebSocket manager import successful")
    except Exception as e:
        logger.error(f"WebSocket manager import failed: {e}")
        return False
    
    try:
        from storage.parquet_writer import ParquetWriter
        logger.info("Parquet writer import successful")
    except Exception as e:
        logger.error(f"❌ Parquet writer import failed: {e}")
        return False
    
    try:
        from collectors.base_collector import BaseCollector
        logger.info("Base collector import successful")
    except Exception as e:
        logger.error(f"Base collector import failed: {e}")
        return False
    
    try:
        from collectors.bitmex_collector import BitMEXCollector
        logger.info("BitMEX collector import successful")
    except Exception as e:
        logger.error(f"BitMEX collector import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration creation"""
    logger.info("Testing configuration...")
    
    try:
        from config.settings import CollectionConfig
        
        config = CollectionConfig()
        config.symbols = ['XBTUSD']
        config.exchange = 'bitmex'
        config.storage.data_dir = Path('test_data')
        
        logger.info(f"  Created config with symbols: {config.symbols}")
        logger.info(f"  Data directory will be: {config.storage.data_dir}")
        
        # Test validation
        config.validate()
        logger.info("Configuration creation and validation successful")
        logger.info(f"  Symbols: {config.symbols}")
        logger.info(f"  Exchange: {config.exchange}")
        logger.info(f"  Data dir: {config.storage.data_dir}")
        
        return True, config
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        import traceback
        logger.error("Full traceback:")
        traceback.print_exc()
        return False, None

def test_websocket_manager():
    """Test WebSocket manager creation"""
    logger.info("Testing WebSocket manager...")
    
    try:
        from utils.websocket_manager import WebSocketManager, WebSocketConfig
        
        ws_config = WebSocketConfig()
        ws_config.url = "wss://ws.bitmex.com/realtime"
        
        ws_manager = WebSocketManager(ws_config)
        logger.info("WebSocket manager creation successful")
        logger.info(f"  URL: {ws_config.url}")
        
        # FIX: Changed get_state() to is_connected() which exists on the manager
        logger.info(f"  Is Connected: {ws_manager.is_connected()}")
        
        return True, ws_manager
        
    except Exception as e:
        logger.error(f"WebSocket manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    
def test_collector_creation():
    """Test collector creation without starting"""
    logger.info("Testing collector creation...")
    
    try:
        success, config = test_config()
        if not success:
            return False
        
        from collectors.bitmex_collector import BitMEXCollector
        
        collector = BitMEXCollector(config)
        logger.info("Collector creation successful")
        logger.info(f"  Exchange: {collector.config.exchange}")
        logger.info(f"  Symbols: {collector.config.symbols}")
        
        return True
        
    except Exception as e:
        logger.error(f"Collector creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_connection():
    """Test actual WebSocket connection (optional)"""
    logger.info("Testing WebSocket connection...")
    
    try:
        success, ws_manager = test_websocket_manager()
        if not success:
            return False
        
        # Test connection without callbacks
        logger.info("Attempting to connect to BitMEX...")
        connected = ws_manager.connect()
        
        if connected:
            logger.info("WebSocket connection initiated")
            
            # Wait a bit to see if connection establishes
            import time
            time.sleep(3)
            
            if ws_manager.is_connected():
                logger.info("WebSocket connected successfully")
                ws_manager.disconnect()
                return True
            else:
                logger.warning("WebSocket connection initiated but not confirmed")
                return True
        else:
            logger.error("Failed to initiate WebSocket connection")
            return False
            
    except Exception as e:
        logger.error(f"WebSocket connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("THERMODYNAMIC DATA COLLECTOR - DEBUG TESTS")
    logger.info("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        logger.error("Import tests failed - check your Python path and dependencies")
        return 1
    
    # Test 2: Configuration
    if not test_config()[0]:
        logger.error("Configuration tests failed")
        return 1
    
    # Test 3: WebSocket Manager
    if not test_websocket_manager()[0]:
        logger.error("WebSocket manager tests failed")
        return 1
    
    # Test 4: Collector Creation
    if not test_collector_creation():
        logger.error("Collector creation tests failed")
        return 1
    
    # Test 5: WebSocket Connection (optional)
    logger.info("All basic tests passed! Testing actual connection...")
    if test_websocket_connection():
        logger.info("All tests passed including connection test!")
    else:
        logger.warning("Basic tests passed but connection test failed")
        logger.warning("This might be due to network issues or BitMEX being down")
    
    logger.info("=" * 60)
    logger.info("DEBUG TESTS COMPLETED")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())