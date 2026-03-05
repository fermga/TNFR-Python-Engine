import os
import sys
import json
import gzip
import heapq
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add paths to find GMX modules
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'future-research/crypto-lab/src'))

# Try importing GMX modules
try:
    from gmx_self_optimization import GMXSelfOptimizationEngine, TradingExperience
    from gmx_tnfr_analyzer import GMXTNFRAnalyzer, GMXMarketData, TNFRStructuralMetrics
    from gmx_trading_engine import GMXTradingSignal
except ImportError as e:
    print(f"Error importing GMX modules: {e}")
    print("Please ensure you are running this script from the workspace root.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GMXOptimizerTrainer")

@dataclass
class HistoricalEvent:
    timestamp: float
    symbol: str
    data: Dict[str, Any]
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

class GMXOptimizerTrainer:
    def __init__(self, data_dir: str = "data/history"):
        self.data_dir = data_dir
        self.brain = GMXSelfOptimizationEngine()
        self.analyzer = GMXTNFRAnalyzer()
        self.event_queue = []
        self.active_signals: Dict[str, List[GMXTradingSignal]] = {}
        
    def load_history(self):
        """Load all .json.gz files and merge into a chronological event queue."""
        logger.info(f"Loading history from {self.data_dir}...")
        count = 0
        
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} not found!")
            return

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json.gz"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                        # Expecting data to be a list of snapshots or a dict with 'history'
                        if isinstance(data, list):
                            for entry in data:
                                self._add_to_queue(entry)
                                count += 1
                        elif isinstance(data, dict) and 'history' in data:
                             for entry in data['history']:
                                self._add_to_queue(entry)
                                count += 1
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        heapq.heapify(self.event_queue)
        logger.info(f"Loaded {count} historical events.")

    def _add_to_queue(self, entry: Dict[str, Any]):
        """Parse entry and add to priority queue."""
        ts = entry.get('timestamp')
        symbol = entry.get('symbol', 'UNKNOWN')
        
        if ts:
            heapq.heappush(self.event_queue, HistoricalEvent(ts, symbol, entry))

    def _determine_capital_state(self, metrics: TNFRStructuralMetrics) -> str:
        """
        Determines the optimal 'State of Capital' (Metamorphic Trading).
        Copied from GMXTradingEngine.
        """
        # 1. Check for GASEOUS State (Explosion/Bifurcation)
        if metrics.structural_potential > 1.618 or metrics.bifurcation_probability > 0.6:
            return "GASEOUS"
            
        # 2. Check for SOLID State (Stability/Coherence)
        if metrics.coherence_length > 50.0 and metrics.phase_gradient < 0.15:
            return "SOLID"
            
        # 3. Default to LIQUID State (Transition/Flow)
        return "LIQUID"

    def _generate_signals(self, market_data: Dict[str, GMXMarketData], 
                         metrics: TNFRStructuralMetrics) -> List[GMXTradingSignal]:
        """
        Generate signals based on metrics.
        Simplified version of GMXTradingEngine._generate_trading_signals.
        """
        signals = []
        
        for symbol, data in market_data.items():
            capital_state = self._determine_capital_state(metrics)
            
            # Simplified signal logic
            if capital_state == "SOLID":
                # LP Signal
                signals.append(GMXTradingSignal(
                    symbol=symbol,
                    action="MINT",
                    confidence=0.9,
                    signal_type="LIQUIDITY_PROVISION",
                    structural_basis="SOLID STATE",
                    primary_metric="coherence_length",
                    timestamp=datetime.fromtimestamp(data.timestamp),
                    entry_price=data.mark_price
                ))
            elif capital_state == "GASEOUS":
                # Directional Signal
                signals.append(GMXTradingSignal(
                    symbol=symbol,
                    action="LONG", # Placeholder direction
                    confidence=0.8,
                    signal_type="DIRECTIONAL_MOMENTUM",
                    structural_basis="GASEOUS STATE",
                    primary_metric="structural_potential",
                    timestamp=datetime.fromtimestamp(data.timestamp),
                    entry_price=data.mark_price
                ))
            
        return signals

    def train_epoch(self):
        """Replay history and train the brain."""
        logger.info("Starting training epoch...")
        
        processed = 0
        
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            
            # Convert raw data to GMXMarketData
            market_data_obj = GMXMarketData(
                symbol=event.symbol,
                timestamp=event.timestamp,
                mark_price=event.data.get('mark_price', 0.0),
                index_price=event.data.get('index_price', 0.0),
                long_open_interest=event.data.get('long_open_interest', 0.0),
                short_open_interest=event.data.get('short_open_interest', 0.0),
                funding_rate=event.data.get('funding_rate', 0.0),
                velocity=event.data.get('velocity', 0.0),
                volatility=event.data.get('volatility', 0.0)
            )
            
            market_data = {event.symbol: market_data_obj}
            
            # 1. Update Analyzer
            self.analyzer.update_market_data(market_data)
            
            # 2. Get Metrics
            metrics = self.analyzer.calculate_structural_metrics(market_data)
            
            # 3. Generate Signals (Virtual)
            signals = self._generate_signals(market_data, metrics)
            
            # 4. Process Active Signals (Check for outcomes)
            self._check_outcomes(market_data)
            
            # 5. Register new signals
            for signal in signals:
                if signal.symbol not in self.active_signals:
                    self.active_signals[signal.symbol] = []
                self.active_signals[signal.symbol].append(signal)
            
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"Processed {processed} events...")

        # Save the trained brain
        self.brain.save_state("gmx_brain_state_trained.pkl")
        logger.info("Training complete. Brain state saved.")

    def _check_outcomes(self, market_data: Dict[str, GMXMarketData]):
        """Check if active signals have realized PnL."""
        for symbol, signals in self.active_signals.items():
            if symbol not in market_data:
                continue
                
            current_price = market_data[symbol].mark_price
            timestamp = market_data[symbol].timestamp
            
            active_list = []
            for signal in signals:
                # Simple logic: Hold for fixed time or price delta
                # For training, let's assume a 1-hour hold or 1% move
                
                signal_time = signal.timestamp.timestamp()
                time_delta = timestamp - signal_time
                
                # Outcome determination (Simplified)
                outcome = 0.0
                is_done = False
                
                if time_delta > 3600: # 1 hour
                    # Close
                    entry = signal.entry_price if signal.entry_price else current_price # Fallback
                    if entry > 0:
                        if signal.action == "LONG" or signal.action == "MINT":
                            outcome = (current_price - entry) / entry
                        else:
                            outcome = (entry - current_price) / entry
                    is_done = True
                
                if is_done:
                    # Record Experience
                    # State vector should ideally come from metrics, but we simplify here
                    exp = TradingExperience(
                        state_vector=[0.5, 0.5, 0.5], 
                        action_id=1 if signal.action == "LONG" else 0,
                        reward=outcome,
                        next_state_vector=[0.5, 0.5, 0.5], 
                        metadata={"symbol": symbol, "type": signal.signal_type}
                    )
                    self.brain.record_experience(exp)
                else:
                    active_list.append(signal)
            
            self.active_signals[symbol] = active_list

if __name__ == "__main__":
    trainer = GMXOptimizerTrainer()
    trainer.load_history()
    trainer.train_epoch()
