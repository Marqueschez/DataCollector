# collectors/bitmex_collector.py
import json
import time
from typing import Dict, Any, List
import logging
import pandas as pd

from utils.timestamp_utils import normalize_timestamp, validate_timestamp_range


from collectors.base_collector import BaseCollector
from config.settings import CollectionConfig

logger = logging.getLogger(__name__)

class BitMEXCollector(BaseCollector):
    """BitMEX-specific data collector implementation"""
    
    def __init__(self, config: CollectionConfig):
        super().__init__(config)
        
        # BitMEX-specific state
        self.orderbook_state = {}  # Track L2 orderbook state by symbol
        self.last_instrument_update = {}  # Track last update time per symbol
        
        logger.info(f"Initialized BitMEX collector for symbols: {config.symbols}")
    
    def get_websocket_url(self) -> str:
        """Get BitMEX WebSocket URL"""
        return self.config.websocket.url
    
    def _subscribe_to_streams(self) -> None:
        """
        Subscribes to all required data streams for the configured symbols,
        batching requests to stay within BitMEX's API limits.
        """
        if not self.ws_manager or not self.ws_manager.is_connected():
            logger.error("Cannot subscribe, WebSocket is not connected.")
            return

        all_topics = []
        for symbol in self.config.symbols:
            if self.config.collect_trades:
                all_topics.append(f"trade:{symbol}")
            if self.config.collect_orderbook:
                all_topics.append(f"orderBookL2:{symbol}")
            if self.config.collect_liquidations:
                all_topics.append(f"liquidation:{symbol}")
            if self.config.collect_instruments:
                all_topics.append(f"instrument:{symbol}")

        # BitMEX has a limit of 20 subscription arguments per request.
        # We'll use a chunk size of 15 to be safe.
        chunk_size = 15
        
        if not all_topics:
            logger.warning("No topics to subscribe to based on current configuration.")
            return

        for i in range(0, len(all_topics), chunk_size):
            chunk = all_topics[i:i + chunk_size]
            sub_message = {
                "op": "subscribe",
                "args": chunk
            }
            logger.info(f"Sending subscription request for {len(chunk)} topics (Chunk {i//chunk_size + 1})...")
            self.ws_manager.send_message(sub_message)
    
    def process_message(self, message: str) -> None:
        """Process incoming BitMEX WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if 'table' not in data:
                # Non-table messages (confirmations, errors, etc.)
                self._handle_non_table_message(data)
                return
            
            table = data['table']
            action = data.get('action', '')
            table_data = data.get('data', [])
            
            if table == 'trade':
                self._process_trades(table_data)
            elif table == 'orderBookL2':
                self._process_orderbook_l2(table_data, action)
            elif table == 'liquidation':
                self._process_liquidations(table_data)
            elif table == 'instrument':
                self._process_instruments(table_data)
            else:
                logger.debug(f"Unhandled table type: {table}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {e}")
            self.metrics.errors += 1
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics.errors += 1
    
    def _handle_non_table_message(self, data: Dict[str, Any]):
        """Handle non-table messages (confirmations, info, etc.)"""
        if 'info' in data:
            logger.info(f"BitMEX info: {data['info']}")
        elif 'subscribe' in data:
            logger.info(f"Subscription confirmed: {data}")
        elif 'error' in data:
            logger.error(f"BitMEX error: {data['error']}")
            self.metrics.errors += 1
        elif 'success' in data:
            logger.debug(f"BitMEX success: {data}")
    

    # Replace timestamp parsing in _process_trades method:
    def _process_trades(self, trades: List[Dict[str, Any]]):
        """Process trade data from BitMEX"""
        for trade in trades:
            try:
                # Use utility function for robust timestamp parsing
                timestamp_raw = trade.get('timestamp')
                timestamp = normalize_timestamp(timestamp_raw) if timestamp_raw else time.time()
                
                # Validate timestamp is reasonable
                if not validate_timestamp_range(timestamp):
                    logger.warning(f"Suspicious timestamp in trade: {timestamp}")
                    timestamp = time.time()  # Use current time as fallback
                
                # Extract trade data
                trade_data = {
                    'timestamp': timestamp,
                    'trdMatchID': trade.get('trdMatchID'),
                    'symbol': trade.get('symbol'),
                    'price': float(trade.get('price', 0)),
                    'size': int(trade.get('size', 0)),
                    'homeNotional': float(trade.get('homeNotional', 0)),
                    'foreignNotional': float(trade.get('foreignNotional', 0)),
                    'side': trade.get('side'),
                    'tickDirection': trade.get('tickDirection'),
                    'grossValue': int(trade.get('grossValue', 0)),
                }
                
                # Add to buffer
                self._add_trade(trade_data)

                # Log occasionally for monitoring
                if self.metrics.trades_collected % 100 == 0:
                    symbol = trade_data['symbol']
                    price = trade_data['price']
                    size_usd = abs(trade_data.get('foreignNotional', trade_data['size']))
                    side = trade_data['side']
                    logger.debug(f"Trade {self.metrics.trades_collected}: {symbol} ${price:,.0f} | ${size_usd:,.0f} | {side}")
                    
            except Exception as e:
                logger.error(f"Error processing trade: {e}")
                self.metrics.errors += 1
    
    def _process_orderbook_l2(self, orderbook_data: List[Dict[str, Any]], action: str):
        """Process Level 2 orderbook data from BitMEX"""
        try:
            current_time = time.time()
            
            for level in orderbook_data:
                level_id = level.get('id')
                symbol = level.get('symbol')
                
                if not level_id or not symbol:
                    continue
                
                # Initialize symbol state if needed
                if symbol not in self.orderbook_state:
                    self.orderbook_state[symbol] = {}
                
                # Create orderbook update record
                orderbook_update = {
                    'timestamp': current_time,
                    'id': level_id,
                    'symbol': symbol,
                    'side': level.get('side'),
                    'price': float(level.get('price', 0)) if level.get('price') else None,
                    'size': int(level.get('size', 0)) if level.get('size') is not None else None,
                    'action': action,
                }
                
                # Update internal state for partial/insert/update/delete
                if action == 'partial':
                    # Full snapshot - reset state for this symbol
                    self.orderbook_state[symbol] = {}
                    if orderbook_update['price'] and orderbook_update['size'] is not None:
                        self.orderbook_state[symbol][level_id] = {
                            'side': orderbook_update['side'],
                            'price': orderbook_update['price'],
                            'size': orderbook_update['size']
                        }
                elif action == 'insert':
                    # New level
                    if orderbook_update['price'] and orderbook_update['size'] is not None:
                        self.orderbook_state[symbol][level_id] = {
                            'side': orderbook_update['side'],
                            'price': orderbook_update['price'],
                            'size': orderbook_update['size']
                        }
                elif action == 'update':
                    # Update existing level
                    if level_id in self.orderbook_state[symbol]:
                        existing = self.orderbook_state[symbol][level_id]
                        if orderbook_update['size'] is not None:
                            if orderbook_update['size'] == 0:
                                # Size 0 means delete
                                del self.orderbook_state[symbol][level_id]
                                orderbook_update['action'] = 'delete'
                            else:
                                existing['size'] = orderbook_update['size']
                        if orderbook_update['price']:
                            existing['price'] = orderbook_update['price']
                        if orderbook_update['side']:
                            existing['side'] = orderbook_update['side']
                elif action == 'delete':
                    # Remove level
                    self.orderbook_state[symbol].pop(level_id, None)
                
                # Add to buffer
                self._add_orderbook_update(orderbook_update)
                
        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")
            self.metrics.errors += 1
    
    def _process_liquidations(self, liquidations: List[Dict[str, Any]]):
        """Process liquidation data from BitMEX"""
        for liquidation in liquidations:
            try:
                # Robust timestamp parsing
                timestamp_raw = liquidation.get('timestamp')
                timestamp = normalize_timestamp(timestamp_raw) if timestamp_raw else time.time()
                
                # Validate timestamp is reasonable
                if not validate_timestamp_range(timestamp):
                    logger.warning(f"Suspicious timestamp in liquidation: {timestamp}")
                    timestamp = time.time() # Use current time as fallback
                
                # Extract liquidation data
                # Note: BitMEX liquidations use order update format:
                # - cumQty: executed quantity (what was actually liquidated)
                # - leavesQty: remaining unfilled quantity
                # - orderQty: original order size (may be 0 for partial fills)
                liquidation_data = {
                    'timestamp': timestamp,
                    'orderID': liquidation.get('orderID'),
                    'symbol': liquidation.get('symbol'),
                    'side': liquidation.get('side'),  # 'Buy' = short liquidation, 'Sell' = long liquidation
                    'orderQty': int(liquidation.get('cumQty', 0)),  # FIX: Use cumQty (executed) not orderQty
                    'price': float(liquidation.get('price', 0)),
                    'leavesQty': int(liquidation.get('leavesQty', 0)),
                    'cumQty': int(liquidation.get('cumQty', 0)),
                    'ordType': liquidation.get('ordType'),
                    'timeInForce': liquidation.get('timeInForce'),
                }

                # Add to buffer
                self._add_liquidation(liquidation_data)

                # Log liquidations as they're important events
                symbol = liquidation_data['symbol']
                side = liquidation_data['side']
                # Use cumQty directly as it represents executed quantity
                executed_qty = liquidation_data['cumQty']
                price = liquidation_data['price']

                # Only log if there was actual execution
                if executed_qty > 0:
                    logger.info(f"LIQUIDATION: {symbol} {side} {executed_qty:,} contracts @ ${price:,.0f}")
                
            except Exception as e:
                logger.error(f"Error processing liquidation: {e}")
                self.metrics.errors += 1
    
    def _process_instruments(self, instruments: List[Dict[str, Any]]):
        """Process instrument data (OI, funding, mark price, etc.) from BitMEX"""
        for instrument in instruments:
            try:
                # Parse timestamp
                if 'timestamp' in instrument:
                    timestamp = pd.to_datetime(instrument['timestamp'], utc=True).timestamp()
                else:
                    timestamp = time.time()
                
                symbol = instrument.get('symbol')
                
                # Rate limiting - only process if significant time has passed or important fields updated
                if symbol in self.last_instrument_update:
                    time_since_last = timestamp - self.last_instrument_update[symbol]
                    if time_since_last < 1.0:  # Less than 1 second since last update
                        # Only process if important fields are present
                        important_fields = ['openInterest', 'fundingRate', 'indicativeFundingRate']
                        if not any(field in instrument for field in important_fields):
                            continue
                
                self.last_instrument_update[symbol] = timestamp
                
                # Extract instrument data
                instrument_data = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'openInterest': int(instrument.get('openInterest', 0)) if instrument.get('openInterest') is not None else None,
                    'fundingRate': float(instrument.get('fundingRate', 0)) if instrument.get('fundingRate') is not None else None,
                    'indicativeFundingRate': float(instrument.get('indicativeFundingRate', 0)) if instrument.get('indicativeFundingRate') is not None else None,
                    'fundingTimestamp': pd.to_datetime(instrument['fundingTimestamp'], utc=True).timestamp() if instrument.get('fundingTimestamp') else None,
                    'markPrice': float(instrument.get('markPrice', 0)) if instrument.get('markPrice') is not None else None,
                    'indexPrice': float(instrument.get('indexPrice', 0)) if instrument.get('indexPrice') is not None else None,
                    'settlementPrice': float(instrument.get('settlementPrice', 0)) if instrument.get('settlementPrice') is not None else None,
                    'volume24h': int(instrument.get('volume24h', 0)) if instrument.get('volume24h') is not None else None,
                    'turnover24h': float(instrument.get('turnover24h', 0)) if instrument.get('turnover24h') is not None else None,
                }
                
                # Add to buffer
                self._add_instrument_data(instrument_data)
                
                # Log important changes
                if instrument.get('openInterest') is not None:
                    oi = instrument_data['openInterest']
                    if self.metrics.instruments_collected % 50 == 0:  # Log every 50th update
                        logger.debug(f"Instrument: {symbol} OI={oi:,}")
                
                # Log funding rate changes
                if instrument.get('fundingRate') is not None:
                    funding = instrument_data['fundingRate']
                    if abs(funding) > 0.001:  # Funding > 0.1% per 8 hours
                        funding_annual = funding * 365 * 3  # Approximate annual rate
                        logger.info(f"💰 FUNDING: {symbol} {funding*100:.4f}% (8h) | {funding_annual:.1f}% annual")
                
            except Exception as e:
                logger.error(f"Error processing instrument: {e}")
                self.metrics.errors += 1
    
    def get_orderbook_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get current orderbook snapshot for a symbol"""
        if symbol not in self.orderbook_state:
            return {'bids': {}, 'asks': {}}
        
        bids = {}
        asks = {}
        
        for level_id, level_data in self.orderbook_state[symbol].items():
            price = level_data['price']
            size = level_data['size']
            side = level_data['side']
            
            if side == 'Buy':
                bids[price] = bids.get(price, 0) + size
            else:  # Sell
                asks[price] = asks.get(price, 0) + size
        
        return {
            'symbol': symbol,
            'timestamp': time.time(),
            'bids': bids,
            'asks': asks,
            'bid_levels': len(bids),
            'ask_levels': len(asks),
        }
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get summary of current market state"""
        summary = {
            'timestamp': time.time(),
            'symbols': {},
            'collection_status': self.get_status(),
        }
        
        for symbol in self.config.symbols:
            orderbook = self.get_orderbook_snapshot(symbol)
            
            symbol_summary = {
                'orderbook_levels': {
                    'bids': orderbook['bid_levels'],
                    'asks': orderbook['ask_levels'],
                },
            }
            
            # Add best bid/ask if available
            if orderbook['bids']:
                symbol_summary['best_bid'] = max(orderbook['bids'].keys())
            if orderbook['asks']:
                symbol_summary['best_ask'] = min(orderbook['asks'].keys())
            
            # Calculate spread
            if 'best_bid' in symbol_summary and 'best_ask' in symbol_summary:
                bid = symbol_summary['best_bid']
                ask = symbol_summary['best_ask']
                spread = ask - bid
                mid = (bid + ask) / 2
                spread_bps = (spread / mid) * 10000 if mid > 0 else 0
                
                symbol_summary['spread'] = {
                    'absolute': spread,
                    'bps': spread_bps,
                    'mid_price': mid,
                }
            
            summary['symbols'][symbol] = symbol_summary
        
        return summary
    
    def export_sample_data(self, symbol: str, hours: int = 1) -> Dict[str, Any]:
        """Export sample of collected data for analysis"""
        cutoff_time = time.time() - (hours * 3600)
        
        sample_data = {
            'symbol': symbol,
            'export_time': time.time(),
            'sample_hours': hours,
            'trades': [],
            'liquidations': [],
            'orderbook_snapshots': [],
        }
        
        # Get sample trades
        trades = self.buffers['trades'].get_copy()
        sample_data['trades'] = [
            t for t in trades 
            if t.get('symbol') == symbol and t.get('timestamp', 0) > cutoff_time
        ][-100:]  # Last 100 trades
        
        # Get sample liquidations
        liquidations = self.buffers['liquidations'].get_copy()
        sample_data['liquidations'] = [
            l for l in liquidations 
            if l.get('symbol') == symbol and l.get('timestamp', 0) > cutoff_time
        ]
        
        # Get current orderbook snapshot
        sample_data['orderbook_snapshots'] = [self.get_orderbook_snapshot(symbol)]
        
        return sample_data