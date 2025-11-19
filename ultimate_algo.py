#INDEXBASED + EOD NOT COMMING - FIXED VERSION
# MODIFIED WITH INSTITUTIONAL PRESSURE STRATEGY - COMPLETE FLOW

import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import pyotp
import math
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
import numpy as np
import pytz

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
# INSTITUTIONAL PRESSURE THRESHOLDS
INSTITUTIONAL_THRESHOLDS = {
    "NIFTY": 25,      # 20-25 points optimal
    "BANKNIFTY": 55,  # 50-60 points optimal  
    "SENSEX": 35      # 30-40 points optimal
}

MIN_INSTITUTIONAL_SCORE = 60
MIN_VOLUME_SURGE = 1.3
MIN_EFFICIENCY_RATIO = 1.5

OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True
EXPIRY_INFO_ONLY = False

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "25 NOV 2025",
    "BANKNIFTY": "25 NOV 2025", 
    "SENSEX": "20 NOV 2025"
}

# --------- STRATEGY TRACKING ---------
STRATEGY_NAMES = {
    "institutional_pressure": "INSTITUTIONAL PRESSURE",
    "gamma_squeeze": "GAMMA SQUEEZE"
}

# --------- ENHANCED TRACKING FOR REPORTS ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []

# --------- SIGNAL DEDUPLICATION AND COOLDOWN TRACKING ---------
active_strikes = {}
last_signal_time = {}
signal_cooldown = 300  # 5 minutes in seconds

def initialize_strategy_tracking():
    """Initialize strategy performance tracking"""
    global strategy_performance
    strategy_performance = {
        "INSTITUTIONAL PRESSURE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "GAMMA SQUEEZE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
    }

# Initialize tracking
initialize_strategy_tracking()

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
TOTP = pyotp.TOTP(TOTP_SECRET).now()

client = SmartConnect(api_key=API_KEY)
session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
feedToken = client.getfeedToken()

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

def send_telegram(msg, reply_to=None):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        r = requests.post(url, data=payload, timeout=5).json()
        return r.get("result", {}).get("message_id")
    except:
        return None

# --------- MARKET HOURS ---------
def is_market_open():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

def should_stop_trading():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return current_time_ist >= dtime(15,30)

# --------- STRIKE ROUNDING FOR KEPT INDICES ---------
def round_strike(index, price):
    try:
        if price is None:
            return None
        if isinstance(price, float) and math.isnan(price):
            return None
        price = float(price)
        
        if index == "NIFTY": 
            return int(round(price / 50.0) * 50)
        elif index == "BANKNIFTY": 
            return int(round(price / 100.0) * 100)
        elif index == "SENSEX": 
            return int(round(price / 100.0) * 100)
        else: 
            return int(round(price / 50.0) * 50)
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FETCH INDEX DATA FOR KEPT INDICES ---------
def fetch_index_data(index, interval="1m", period="1d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN"
    }
    
    # FOR LIVE DATA - Use today's date specifically
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        # Fetch data for TODAY only for live trading
        df = yf.download(
            symbol_map[index], 
            start=today,
            interval=interval, 
            progress=False,
            prepost=True  # Include pre-market data
        )
        
        # If no data for today, try with period="1d" as fallback
        if df.empty:
            df = yf.download(symbol_map[index], period="1d", interval=interval, progress=False)
            
        return None if df.empty else df
        
    except Exception as e:
        print(f"Error fetching {index} data: {e}")
        # Fallback to traditional method
        df = yf.download(symbol_map[index], period="1d", interval=interval, progress=False)
        return None if df.empty else df

# --------- LOAD TOKEN MAP ---------
def load_token_map():
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df.columns=[c.lower() for c in df.columns]
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        return df.set_index('symbol')['token'].to_dict()
    except:
        return {}

token_map=load_token_map()

# --------- SAFE LTP FETCH ---------
def fetch_option_price(symbol, retries=3, delay=3):
    token=token_map.get(symbol.upper())
    if not token:
        return None
    for _ in range(retries):
        try:
            exchange = "BFO" if "SENSEX" in symbol.upper() else "NFO"
            data=client.ltpData(exchange, symbol, token)
            return float(data['data']['ltp'])
        except:
            time.sleep(delay)
    return None

# --------- STRICT EXPIRY VALIDATION ---------
def validate_option_symbol(index, symbol, strike, opttype):
    """STRICT validation to ensure ONLY specified expiry symbols are used"""
    try:
        expected_expiry = EXPIRIES.get(index)
        if not expected_expiry:
            return False
            
        expected_dt = datetime.strptime(expected_expiry, "%d %b %Y")
        symbol_upper = symbol.upper()
        
        if index == "SENSEX":
            year_short = expected_dt.strftime("%y")
            month_code = expected_dt.strftime("%b").upper()
            expected_pattern = f"SENSEX{year_short}{month_code}"
            return expected_pattern in symbol_upper
        else:
            expected_pattern = expected_dt.strftime("%d%b%y").upper()
            return expected_pattern in symbol_upper
            
    except Exception:
        return False

# --------- STRICT OPTION SYMBOL GENERATION ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    """STRICT: Generate symbols ONLY for specified expiries"""
    try:
        dt = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = dt.strftime("%y")
            month_code = dt.strftime("%b").upper()
            day = dt.strftime("%d")
            symbol = f"SENSEX{year_short}{month_code}{strike}{opttype}"
        else:
            symbol = f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        
        if validate_option_symbol(index, symbol, strike, opttype):
            return symbol
        else:
            return None
            
    except Exception:
        return None

# --------- SAFE DATA CONVERSION ---------
def safe_float(value):
    """Safely convert any value to float"""
    try:
        if hasattr(value, 'item'):
            return float(value.item())
        elif hasattr(value, 'iloc'):
            return float(value.iloc[0])
        else:
            return float(value)
    except:
        return 0.0

def safe_int(value):
    """Safely convert any value to int"""
    try:
        if hasattr(value, 'item'):
            return int(value.item())
        elif hasattr(value, 'iloc'):
            return int(value.iloc[0])
        else:
            return int(value)
    except:
        return 0

# --------- INSTITUTIONAL PRESSURE ANALYZER ---------
class InstitutionalPressureAnalyzer:
    def __init__(self):
        self.analyzed_candles = set()
    
    def calculate_institutional_pressure(self, current_candle, prev_candles, direction):
        """Calculate institutional pressure metrics"""
        try:
            curr_open = safe_float(current_candle['Open'])
            curr_high = safe_float(current_candle['High'])
            curr_low = safe_float(current_candle['Low'])
            curr_close = safe_float(current_candle['Close'])
            curr_volume = safe_int(current_candle['Volume'])
            
            # Extract previous candle data
            prev_opens = [safe_float(c['Open']) for c in prev_candles]
            prev_highs = [safe_float(c['High']) for c in prev_candles]
            prev_lows = [safe_float(c['Low']) for c in prev_candles]
            prev_closes = [safe_float(c['Close']) for c in prev_candles]
            prev_volumes = [safe_int(c['Volume']) for c in prev_candles]
            
            # Volume analysis
            base_volume = 50000
            if curr_volume == 0:
                price_movement = abs(curr_close - curr_open)
                range_size = curr_high - curr_low
                volatility = range_size / curr_open if curr_open > 0 else 0
                movement_intensity = (price_movement / curr_open * 100) if curr_open > 0 else 0
                volatility_factor = volatility * 100
                curr_volume = int(base_volume * (1 + movement_intensity * 8 + volatility_factor * 3))
            
            # Calculate synthetic previous volumes
            synthetic_prev_volumes = []
            for i in range(len(prev_opens)):
                if prev_volumes[i] == 0:
                    prev_movement = abs(prev_closes[i] - prev_opens[i])
                    prev_range = prev_highs[i] - prev_lows[i]
                    prev_volatility = prev_range / prev_opens[i] if prev_opens[i] > 0 else 0
                    prev_movement_intensity = (prev_movement / prev_opens[i] * 100) if prev_opens[i] > 0 else 0
                    prev_volatility_factor = prev_volatility * 100
                    synthetic_vol = int(base_volume * (1 + prev_movement_intensity * 8 + prev_volatility_factor * 3))
                    synthetic_prev_volumes.append(synthetic_vol)
                else:
                    synthetic_prev_volumes.append(prev_volumes[i])
            
            # Volume surge ratio
            avg_prev_volume = np.mean(synthetic_prev_volumes) if synthetic_prev_volumes else base_volume
            volume_surge_ratio = round(curr_volume / max(1, avg_prev_volume), 2)
            
            # Price efficiency
            current_efficiency = abs(curr_close - curr_open) / (curr_high - curr_low) if (curr_high - curr_low) > 0 else 0
            prev_efficiencies = []
            for i in range(len(prev_opens)):
                if (prev_highs[i] - prev_lows[i]) > 0:
                    eff = abs(prev_closes[i] - prev_opens[i]) / (prev_highs[i] - prev_lows[i])
                    prev_efficiencies.append(eff)
            
            avg_prev_efficiency = np.mean(prev_efficiencies) if prev_efficiencies else current_efficiency
            efficiency_ratio = round(current_efficiency / max(0.01, avg_prev_efficiency), 2)
            
            # Momentum consistency - ALWAYS MODERATE as requested
            momentum_pressure = "MODERATE"
            
            # SIMPLIFIED SCORING SYSTEM
            score = 0
            
            # Volume scoring (simplified)
            if volume_surge_ratio >= 1.3:  # Your requirement
                score += 30
            
            # Efficiency scoring (simplified)
            if efficiency_ratio >= 1.5:  # Your requirement
                score += 30
            
            # Momentum scoring (always MODERATE)
            if momentum_pressure == "MODERATE":
                score += 20
            
            # Candle size scoring
            candle_size = abs(curr_close - curr_open)
            if candle_size > 50: 
                score += 20
            elif candle_size > 30: 
                score += 15
            elif candle_size > 20: 
                score += 10
            
            institutional_score = min(100, score)
            
            # SIMPLIFIED PRESSURE TYPE DETERMINATION
            if institutional_score >= 70:
                pressure_type = "STRONG_INSTITUTIONAL"
                confidence = "VERY_HIGH"
            elif institutional_score >= 60:
                pressure_type = "MODERATE_INSTITUTIONAL" 
                confidence = "HIGH"
            else:
                pressure_type = "RETAIL_DOMINATED"
                confidence = "LOW"
            
            # Directional pressure
            if direction == "GREEN":
                directional_pressure = "INSTITUTIONAL_BUYING"
            else:
                directional_pressure = "INSTITUTIONAL_SELLING"
            
            return {
                'volume_surge_ratio': volume_surge_ratio,
                'efficiency_ratio': efficiency_ratio,
                'momentum_pressure': momentum_pressure,
                'institutional_score': institutional_score,
                'pressure_type': pressure_type,
                'confidence': confidence,
                'directional_pressure': directional_pressure
            }
            
        except Exception as e:
            return {
                'volume_surge_ratio': 0.0,
                'efficiency_ratio': 0.0,
                'momentum_pressure': "MODERATE",
                'institutional_score': 0,
                'pressure_type': "RETAIL_DOMINATED",
                'confidence': "LOW",
                'directional_pressure': "NEUTRAL"
            }
    
    def find_institutional_pressure(self, df, index):
        """Find institutional pressure signals in recent data"""
        try:
            if df is None or len(df) < 10:
                return None
                
            # Get the threshold for this index
            threshold = INSTITUTIONAL_THRESHOLDS.get(index, 20)
            
            # Analyze last 5 candles (focus on most recent)
            for i in range(max(5, len(df)-5), len(df)):
                try:
                    current_row = df.iloc[i]
                    prev1_row = df.iloc[i-1]
                    prev2_row = df.iloc[i-2]
                    prev3_row = df.iloc[i-3]
                    prev4_row = df.iloc[i-4]
                    prev5_row = df.iloc[i-5]
                    
                    current_open = safe_float(current_row['Open'])
                    current_close = safe_float(current_row['Close'])
                    candle_move = abs(current_close - current_open)
                    
                    # Check if this is a big candle
                    if candle_move >= threshold:
                        direction = "GREEN" if current_close > current_open else "RED"
                        
                        # Calculate institutional pressure
                        prev_candles = [prev5_row, prev4_row, prev3_row, prev2_row, prev1_row]
                        pressure_metrics = self.calculate_institutional_pressure(
                            current_row, prev_candles, direction
                        )
                        
                        # Apply filters - SIMPLIFIED CONDITIONS
                        if (pressure_metrics['institutional_score'] >= MIN_INSTITUTIONAL_SCORE and
                            pressure_metrics['volume_surge_ratio'] >= MIN_VOLUME_SURGE and
                            pressure_metrics['efficiency_ratio'] >= MIN_EFFICIENCY_RATIO):
                            
                            # Convert timestamp to IST
                            candle_timestamp = df.index[i]
                            if candle_timestamp.tzinfo is None:
                                candle_timestamp = pytz.UTC.localize(candle_timestamp)
                            ist_time = candle_timestamp.astimezone(pytz.timezone('Asia/Kolkata'))
                            
                            return {
                                'timestamp': ist_time,
                                'time_str': ist_time.strftime('%H:%M:%S'),
                                'direction': direction,
                                'points_moved': round(candle_move, 2),
                                'pressure_metrics': pressure_metrics
                            }
                            
                except Exception:
                    continue
                    
            return None
            
        except Exception as e:
            return None

# --------- CHECK IF DATA IS LIVE ---------
def is_data_live(df):
    """Check if the data is from today (live) or historical"""
    if df is None or len(df) == 0:
        return False
    
    try:
        latest_timestamp = df.index[-1]
        if latest_timestamp.tzinfo is None:
            latest_timestamp = pytz.UTC.localize(latest_timestamp)
        
        latest_ist = latest_timestamp.astimezone(pytz.timezone('Asia/Kolkata'))
        today_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        # Check if data is from today
        return latest_ist.date() == today_ist.date()
    except:
        return False

# --------- FORMAT INSTITUTIONAL ANALYSIS ---------
def format_institutional_analysis(index, pressure_signal):
    """Format institutional pressure analysis for Telegram"""
    metrics = pressure_signal['pressure_metrics']
    
    if pressure_signal['direction'] == "GREEN":
        pressure_emoji = "🏛️🟢"
    else:
        pressure_emoji = "🏛️🔴"
    
    msg = f"""
{pressure_emoji} **INSTITUTIONAL PRESSURE DETECTED - {index} 1m** {pressure_emoji}

📅 **DATE**: {datetime.now().strftime('%d %b %Y')}
⏰ **TIME**: {pressure_signal['time_str']} IST
🎯 **DIRECTION**: {pressure_signal['direction']}
📈 **POINTS MOVED**: {pressure_signal['points_moved']} points

🏛️ **TRUE INSTITUTIONAL METRICS**:
• Volume Surge: {metrics['volume_surge_ratio']}x
• Price Efficiency: {metrics['efficiency_ratio']}x
• Momentum Alignment: {metrics['momentum_pressure']}

💼 **INSTITUTIONAL ASSESSMENT**:
• Institutional Score: {metrics['institutional_score']}/100
• Pressure Type: {metrics['pressure_type']}
• Confidence: {metrics['confidence']}
• Directional Pressure: {metrics['directional_pressure']}

🎯 **TRADING IMPLICATION**:
{metrics['directional_pressure']} | {metrics['confidence']} confidence
True Activity: {metrics['pressure_type']}

─────────────────────────────
"""
    return msg

# --------- STRICT SIGNAL DEDUPLICATION AND COOLDOWN CHECK ---------
def can_send_signal(index, strike, option_type):
    """STRICT: Check if we can send signal based on deduplication and cooldown rules"""
    global active_strikes, last_signal_time
    
    current_time = time.time()
    strike_key = f"{index}_{strike}_{option_type}"
    
    # Check if same strike is already active
    if strike_key in active_strikes:
        print(f"❌ Signal blocked: Same strike already active - {strike_key}")
        return False
        
    # Check cooldown for this index
    if index in last_signal_time:
        time_since_last = current_time - last_signal_time[index]
        if time_since_last < signal_cooldown:
            print(f"❌ Signal blocked: Cooldown active for {index} - {int(signal_cooldown - time_since_last)}s remaining")
            return False
    
    print(f"✅ Signal allowed: {strike_key}")
    return True

def update_signal_tracking(index, strike, option_type, signal_id):
    """Update tracking for sent signals"""
    global active_strikes, last_signal_time
    
    strike_key = f"{index}_{strike}_{option_type}"
    active_strikes[strike_key] = {
        'signal_id': signal_id,
        'timestamp': time.time()
    }
    
    last_signal_time[index] = time.time()
    print(f"📝 Tracking updated: {strike_key}")

def clear_completed_signal(signal_id):
    """Clear signal from active tracking when completed"""
    global active_strikes
    active_strikes = {k: v for k, v in active_strikes.items() if v['signal_id'] != signal_id}
    print(f"🧹 Cleared completed signal: {signal_id}")

# --------- UPDATED SIGNAL SENDING WITH COMPLETE FLOW ---------
def send_complete_signal(index, pressure_signal):
    """Send complete signal flow: Analysis + Option Trade"""
    global signal_counter, all_generated_signals
    
    side = "CE" if pressure_signal['direction'] == "GREEN" else "PE"
    
    # Get current price for strike calculation
    df = fetch_index_data(index, "1m", "1d")
    if df is None:
        return
        
    current_price = float(ensure_series(df["Close"]).iloc[-1])
    strike = round_strike(index, current_price)
    
    if strike is None:
        return
        
    # Check deduplication and cooldown
    if not can_send_signal(index, strike, side):
        send_telegram(f"⏳ {index} {strike} {side}: Signal blocked (duplicate/cooldown)")
        return
        
    # Generate option symbol
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    if symbol is None:
        send_telegram(f"❌ {index}: Could not generate valid symbol for strike {strike} {side}")
        return
    
    # Fetch option price
    option_price = fetch_option_price(symbol)
    if not option_price: 
        send_telegram(f"❌ {index}: Could not fetch price for {symbol}")
        return
    
    entry = round(option_price)
    
    # Calculate institutional targets (bigger moves)
    if side == "CE":
        base_move = max(current_price * 0.008, 40)  # Minimum 40 points
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),
            round(entry + base_move * 2.8),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.8)
    else:  # PE
        base_move = max(current_price * 0.008, 40)  # Minimum 40 points
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),
            round(entry + base_move * 2.8),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.8)
    
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    strategy_name = "INSTITUTIONAL PRESSURE"
    
    signal_id = f"SIG{signal_counter:04d}"
    signal_counter += 1
    
    signal_data = {
        "signal_id": signal_id,
        "timestamp": (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": side,
        "strategy": strategy_name,
        "entry_price": entry,
        "targets": targets,
        "sl": sl,
        "fakeout": False,
        "index_price": current_price
    }
    
    # STEP 1: Send Institutional Analysis
    analysis_msg = format_institutional_analysis(index, pressure_signal)
    analysis_msg_id = send_telegram(analysis_msg)
    
    # Wait 2 seconds
    time.sleep(2)
    
    # STEP 2: Send Option Trade Signal
    option_msg = (f"{'🟢' if side == 'CE' else '🔴'} {index} {strike} {side}\n"
           f"SYMBOL: {symbol}\n"
           f"ABOVE {entry}\n"
           f"TARGETS: {targets_str}\n"
           f"SL: {sl}\n"
           f"FAKEOUT: NO\n"
           f"STRATEGY: {strategy_name}\n"
           f"SIGNAL ID: {signal_id}")
    
    option_msg_id = send_telegram(option_msg)
    
    # Update signal tracking
    update_signal_tracking(index, strike, side, signal_id)
    all_generated_signals.append(signal_data.copy())
    
    # Start monitoring thread
    start_monitoring(symbol, entry, targets, sl, option_msg_id, strategy_name, signal_data)

def start_monitoring(symbol, entry, targets, sl, thread_id, strategy_name, signal_data):
    """Start monitoring thread for the signal"""
    def monitor_thread():
        max_price = entry
        targets_hit = 0
        
        print(f"🔍 Monitoring started: {symbol} for 5 minutes")
        
        # Monitor for 5 minutes
        end_time = time.time() + 300  # 5 minutes
        
        while time.time() < end_time:
            current_price = fetch_option_price(symbol)
            if current_price:
                current_price = round(current_price)
                
                if current_price > max_price:
                    max_price = current_price
                
                # Check targets
                for i, target in enumerate(targets):
                    if current_price >= target and i >= targets_hit:
                        targets_hit = i + 1
                        send_telegram(f"🎯 {symbol}: Target {targets_hit} hit at ₹{target}", reply_to=thread_id)
                        print(f"🎯 Target {targets_hit} hit for {symbol}")
                
                # Check SL
                if current_price <= sl:
                    send_telegram(f"🛑 {symbol}: SL hit at ₹{sl}", reply_to=thread_id)
                    print(f"🛑 SL hit for {symbol}")
                    break
            
            time.sleep(5)  # Check every 5 seconds
        
        # Clear signal after monitoring
        clear_completed_signal(signal_data['signal_id'])
        print(f"✅ Monitoring completed: {symbol}")
    
    thread = threading.Thread(target=monitor_thread)
    thread.daemon = True
    thread.start()

# --------- INSTITUTIONAL PRESSURE STRATEGY ---------
def analyze_index_signal(index):
    """INSTITUTIONAL PRESSURE STRATEGY - COMPLETE FLOW"""
    try:
        # Use 1-minute data for institutional pressure detection
        df = fetch_index_data(index, "1m", "1d")
        if df is None or len(df) < 10:
            return None

        # CHECK IF DATA IS LIVE
        if not is_data_live(df):
            print(f"⚠️ {index} data is not live - skipping")
            return None

        # Check cooldown
        current_time = time.time()
        if index in last_signal_time:
            time_since_last = current_time - last_signal_time[index]
            if time_since_last < signal_cooldown:
                return None

        # Analyze for institutional pressure
        analyzer = InstitutionalPressureAnalyzer()
        pressure_signal = analyzer.find_institutional_pressure(df, index)
        
        if pressure_signal:
            # CHECK IF SIGNAL IS FROM CURRENT TIME (not old data)
            signal_time = pressure_signal['timestamp']
            current_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
            time_diff = (current_ist - signal_time).total_seconds()
            
            # Only accept signals from last 2 minutes
            if time_diff > 120:
                print(f"🕒 Old signal ignored: {pressure_signal['time_str']}")
                return None
                
            print(f"✅ LIVE SIGNAL DETECTED: {index} at {pressure_signal['time_str']}")
            
            # Send complete signal flow
            send_complete_signal(index, pressure_signal)
            return True
            
        return None
        
    except Exception as e:
        print(f"Error analyzing {index}: {e}")
        return None

# --------- TRADE THREAD ---------
def trade_thread(index):
    """Generate signals for each index"""
    result = analyze_index_signal(index)
    
    if result:
        print(f"🎯 Signal processed for {index}")

# --------- MAIN LOOP ---------
def run_algo_parallel():
    if not is_market_open(): 
        print("❌ Market closed - skipping iteration")
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
            
        if not EOD_REPORT_SENT:
            time.sleep(15)
            send_individual_signal_reports()
            EOD_REPORT_SENT = True
            
        return
        
    threads = []
    kept_indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    
    for index in kept_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# --------- EOD REPORT ---------
def send_individual_signal_reports():
    """Send EOD reports"""
    global daily_signals, all_generated_signals
    
    all_signals = daily_signals + all_generated_signals
    seen_ids = set()
    unique_signals = []
    for signal in all_signals:
        sid = signal.get('signal_id')
        if not sid:
            continue
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_signals.append(signal)
    
    if not unique_signals:
        send_telegram("📊 END OF DAY REPORT\nNo signals generated today.")
        return
    
    send_telegram(f"🕒 END OF DAY SIGNAL REPORT - { (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y') }\n"
                  f"📈 Total Signals: {len(unique_signals)}\n"
                  f"─────────────────────────────")
    
    for i, signal in enumerate(unique_signals, 1):
        msg = (f"📊 SIGNAL #{i}\n"
               f"Index: {signal.get('index','?')}\n"
               f"Strike: {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"Strategy: {signal.get('strategy','?')}\n"
               f"Signal ID: {signal.get('signal_id','?')}\n"
               f"─────────────────────────────")
        send_telegram(msg)
        time.sleep(1)
    
    send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")

# --------- MAIN EXECUTION ---------
STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

while True:
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        
        market_open = is_market_open()
        
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market is currently closed. Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
            
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 INSTITUTIONAL PRESSURE ALGO STARTED\n"
                         "✅ NIFTY, BANKNIFTY, SENSEX\n"
                         "✅ 5-Minute Monitoring\n"
                         "✅ Institutional Pressure Detection")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached! Preparing EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            if not EOD_REPORT_SENT:
                time.sleep(20)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
            
            time.sleep(60)
            continue
            
        run_algo_parallel()
        time.sleep(5)  # 5 second delay between scans
        
    except Exception as e:
        error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)
