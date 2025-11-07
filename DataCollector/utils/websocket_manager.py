# utils/websocket_manager.py
import websocket
import json
import time
import threading
from typing import Callable, Optional, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum
import random

from config.settings import WebSocketConfig

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

@dataclass
class ConnectionStats:
    """WebSocket connection statistics"""
    total_connections: int = 0
    total_reconnections: int = 0
    total_messages: int = 0
    total_errors: int = 0
    last_message_time: Optional[float] = None
    last_error_time: Optional[float] = None
    connection_uptime_seconds: float = 0.0
    avg_latency_ms: float = 0.0

class WebSocketManager:
    """Robust WebSocket manager with automatic reconnection and health monitoring"""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.ws: Optional[websocket.WebSocketApp] = None
        self.state = ConnectionState.DISCONNECTED
        self.stats = ConnectionStats()
        
        # Callbacks
        self.on_message_callback: Optional[Callable] = None
        self.on_connect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        
        # Connection management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.reconnect_attempts
        self.connection_start_time: Optional[float] = None
        self.last_ping_time: Optional[float] = None
        self.should_stop = False
        
        # Thread management
        self.ws_thread: Optional[threading.Thread] = None
        self.health_thread: Optional[threading.Thread] = None
        self.stats_lock = threading.RLock()
        
        # Health monitoring
        self.last_message_times: list = []  # For latency calculation
        self.health_check_interval = 10  # seconds
        
    def set_callbacks(self, 
                     on_message: Optional[Callable] = None,
                     on_connect: Optional[Callable] = None, 
                     on_disconnect: Optional[Callable] = None,
                     on_error: Optional[Callable] = None):
        """Set callback functions for WebSocket events"""
        self.on_message_callback = on_message
        self.on_connect_callback = on_connect
        self.on_disconnect_callback = on_disconnect
        self.on_error_callback = on_error
    
    def connect(self) -> bool:
        """Initiate WebSocket connection"""
        if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            logger.warning("Already connected or connecting")
            return True
            
        self.state = ConnectionState.CONNECTING
        self.should_stop = False
        
        try:
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.config.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start connection in separate thread
            self.ws_thread = threading.Thread(
                target=self._run_websocket,
                daemon=True,
                name="WebSocketThread"
            )
            self.ws_thread.start()
            
            # Start health monitoring
            if not self.health_thread or not self.health_thread.is_alive():
                self.health_thread = threading.Thread(
                    target=self._health_monitor,
                    daemon=True,
                    name="HealthMonitorThread"
                )
                self.health_thread.start()
            
            logger.info(f"Connecting to {self.config.url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create WebSocket connection: {e}")
            self.state = ConnectionState.FAILED
            return False
    
    def disconnect(self):
        """Gracefully disconnect WebSocket"""
        logger.info("Disconnecting WebSocket...")
        self.should_stop = True
        
        if self.ws:
            self.ws.close()
            
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5.0)
            
        self.state = ConnectionState.DISCONNECTED
        logger.info("WebSocket disconnected")
    
    def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message through WebSocket"""
        if self.state != ConnectionState.CONNECTED or not self.ws:
            logger.warning("Cannot send message: not connected")
            return False
            
        try:
            json_message = json.dumps(message)
            self.ws.send(json_message)
            logger.debug(f"Sent message: {json_message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def subscribe(self, streams: list) -> bool:
        """Subscribe to data streams"""
        subscription_message = {
            "op": "subscribe", 
            "args": streams
        }
        return self.send_message(subscription_message)
    
    def _run_websocket(self):
        """Run WebSocket connection with ping/pong"""
        try:
            self.ws.run_forever(
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                ping_payload="keepalive"
            )
        except Exception as e:
            logger.error(f"WebSocket run_forever error: {e}")
            if not self.should_stop:
                self._schedule_reconnect()
    
    def _on_open(self, ws):
        """Handle WebSocket open event"""
        logger.info("WebSocket connection opened")
        self.state = ConnectionState.CONNECTED
        self.connection_start_time = time.time()
        self.reconnect_attempts = 0
        
        with self.stats_lock:
            self.stats.total_connections += 1
            
        if self.on_connect_callback:
            try:
                self.on_connect_callback()
            except Exception as e:
                logger.error(f"Error in connect callback: {e}")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message"""
        current_time = time.time()
        
        with self.stats_lock:
            self.stats.total_messages += 1
            self.stats.last_message_time = current_time
            
            # Update latency tracking
            self.last_message_times.append(current_time)
            if len(self.last_message_times) > 100:  # Keep last 100 messages
                self.last_message_times.pop(0)
        
        if self.on_message_callback:
            try:
                self.on_message_callback(ws, message)
            except Exception as e:
                logger.error(f"Error in message callback: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error"""
        logger.error(f"WebSocket error: {error}")
        
        with self.stats_lock:
            self.stats.total_errors += 1
            self.stats.last_error_time = time.time()
            
        if self.on_error_callback:
            try:
                self.on_error_callback(ws, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close event"""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        if self.connection_start_time:
            uptime = time.time() - self.connection_start_time
            with self.stats_lock:
                self.stats.connection_uptime_seconds += uptime
        
        if not self.should_stop:
            self.state = ConnectionState.DISCONNECTED
            self._schedule_reconnect()
        else:
            self.state = ConnectionState.DISCONNECTED
            
        if self.on_disconnect_callback:
            try:
                self.on_disconnect_callback(close_status_code, close_msg)
            except Exception as e:
                logger.error(f"Error in disconnect callback: {e}")
    
    def _schedule_reconnect(self):
        """Schedule reconnection with exponential backoff"""
        if self.should_stop or self.reconnect_attempts >= self.max_reconnect_attempts:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                self.state = ConnectionState.FAILED
            return
            
        self.reconnect_attempts += 1
        
        # Exponential backoff with jitter
        delay = min(
            self.config.reconnect_delay_base * (2 ** (self.reconnect_attempts - 1)),
            self.config.reconnect_delay_max
        )
        # Add jitter to prevent thundering herd
        delay += random.uniform(0, delay * 0.1)
        
        logger.info(f"Scheduling reconnect attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} "
                   f"in {delay:.1f} seconds")
        
        self.state = ConnectionState.RECONNECTING
        
        with self.stats_lock:
            self.stats.total_reconnections += 1
        
        # Schedule reconnection
        reconnect_timer = threading.Timer(delay, self._attempt_reconnect)
        reconnect_timer.daemon = True
        reconnect_timer.start()

    def _attempt_reconnect(self):
        """Attempt to reconnect WebSocket"""
        if self.should_stop or self.reconnect_attempts >= self.max_reconnect_attempts:
            return
            
        logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        self.connect()

    def _health_monitor(self):
        """Monitor WebSocket health and connection quality"""
        while not self.should_stop:
            try:
                time.sleep(self.health_check_interval)
                
                if not self.should_stop:
                    self._check_connection_health()
                    
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    def _check_connection_health(self):
        """Check connection health and quality"""
        current_time = time.time()
        
        with self.stats_lock:
            # Check message frequency
            if (self.stats.last_message_time and 
                current_time - self.stats.last_message_time > 60):  # No messages for 60s
                logger.warning("No messages received for 60 seconds")
            
            # Update connection uptime
            if self.connection_start_time and self.state == ConnectionState.CONNECTED:
                self.stats.connection_uptime_seconds = current_time - self.connection_start_time

    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.state == ConnectionState.CONNECTED

    def get_health_info(self) -> Dict[str, Any]:
        """Get comprehensive health information"""
        with self.stats_lock:
            health_quality = "good"
            
            # Determine health quality
            error_rate = self.stats.total_errors / max(self.stats.total_messages, 1)
            if error_rate > 0.05:  # >5% error rate
                health_quality = "poor"
            elif error_rate > 0.01:  # >1% error rate
                health_quality = "degraded"
            
            # Check message recency
            if (self.stats.last_message_time and 
                time.time() - self.stats.last_message_time > 30):
                health_quality = "poor"
            
            return {
                'state': self.state.value,
                'quality': health_quality,
                'stats': {
                    'total_connections': self.stats.total_connections,
                    'total_reconnections': self.stats.total_reconnections,
                    'total_messages': self.stats.total_messages,
                    'total_errors': self.stats.total_errors,
                    'connection_uptime_seconds': self.stats.connection_uptime_seconds,
                    'last_message_time': self.stats.last_message_time,
                    'error_rate': error_rate,
                },
                'reconnect_attempts': self.reconnect_attempts,
                'max_reconnect_attempts': self.max_reconnect_attempts,
            }