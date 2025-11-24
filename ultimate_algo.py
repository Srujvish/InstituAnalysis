#INDEXBASED + INSTITUTIONAL ENHANCEMENTS - ULTIMATE VERSION PREFERRRED STATERGYIES ONLY ENCHANCED TAGRTET + ADDED MAIN REVERSKAS FROM INDEX 

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

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

EXPIRY_ACTIONABLE = True
EXPIRY_INFO_ONLY = False
EXPIRY_RELAX_FACTOR = 0.7
GAMMA_VOL_SPIKE_THRESHOLD = 2.0
DELTA_OI_RATIO = 2.0
MOMENTUM_VOL_AMPLIFIER = 1.5

# INSTITUTIONAL ENTRY FILTERS
MIN_VOLUME_RATIO = 1.5  # 1.5x average volume required
MIN_CANDLE_BODY_RATIO = 0.6  # 60% of range should be body
STRONG_MOVE_THRESHOLD = 0.03  # 3% move for reversal detection
ATR_TARGET_MULTIPLIER = [1.2, 1.3, 1.4, 1.5]  # Realistic targets
ATR_SL_MULTIPLIER = 0.8  # Tighter SL

# STRONGER CONFIRMATION THRESHOLDS
VCP_CONTRACTION_RATIO = 0.6
FAULTY_BASE_BREAK_THRESHOLD = 0.25
WYCKOFF_VOLUME_SPRING = 2.2
LIQUIDITY_SWEEP_DISTANCE = 0.005
PEAK_REJECTION_WICK_RATIO = 0.8
FVG_GAP_THRESHOLD = 0.0025
VOLUME_GAP_IMBALANCE = 2.5
OTE_RETRACEMENT_LEVELS = [0.618, 0.786]
DEMAND_SUPPLY_ZONE_LOOKBACK = 20

# INSTITUTIONAL P&L PARAMETERS
INDEX_PNL_PARAMS = {
    "NIFTY": {"max_profit_points": 10, "lots": 2, "quantity": 150},
    "BANKNIFTY": {"max_profit_points": 12, "lots": 2, "quantity": 70},
    "SENSEX": {"max_profit_points": 15, "lots": 2, "quantity": 40},
    "MIDCPNIFTY": {"max_profit_points": 8, "lots": 1, "quantity": 140}
}

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "25 NOV 2025",
    "BANKNIFTY": "25 NOV 2025", 
    "SENSEX": "27 NOV 2025",
    "MIDCPNIFTY": "25 NOV 2025"
}

# --------- ENHANCED STRATEGY TRACKING ---------
STRATEGY_NAMES = {
    "institutional_price_action": "INSTITUTIONAL PRICE ACTION",
    "liquidity_sweeps": "LIQUIDITY SWEEP",
    "ote_retracement": "OTE RETRACEMENT",
    "demand_supply_zones": "DEMAND SUPPLY ZONES",
    "liquidity_zone": "LIQUIDITY ZONE",
    "institutional_reversal": "INSTITUTIONAL REVERSAL",
    "retail_sl_hunt": "RETAIL SL HUNT"
}

# --------- ENHANCED TRACKING FOR REPORTS ---------
all_generated_signals = []  # Track ALL signals for EOD reporting
strategy_performance = {}
signal_counter = 0
daily_signals = []

# --------- NEW: SIGNAL DEDUPLICATION AND COOLDOWN TRACKING ---------
active_strikes = {}  # Track active strikes to prevent duplicates
last_signal_time = {}  # Track last signal time per index
signal_cooldown = 1200  # 20 minutes in seconds

def initialize_strategy_tracking():
    """Initialize strategy performance tracking"""
    global strategy_performance
    strategy_performance = {
        "INSTITUTIONAL PRICE ACTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "LIQUIDITY SWEEP": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "OTE RETRACEMENT": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "DEMAND SUPPLY ZONES": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "LIQUIDITY ZONE": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "INSTITUTIONAL REVERSAL": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "RETAIL SL HUNT": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
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
        payload = {"chat_id": CHAT_ID, "text": msg}
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
        elif index == "MIDCPNIFTY": 
            return int(round(price / 25.0) * 25)
        else: 
            return int(round(price / 50.0) * 50)
    except Exception:
        return None

# --------- ENSURE SERIES ---------
def ensure_series(data):
    return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()

# --------- FETCH INDEX DATA FOR KEPT INDICES ---------
def fetch_index_data(index, interval="5m", period="2d"):
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
    }
    df = yf.download(symbol_map[index], period=period, interval=interval, auto_adjust=True, progress=False)
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

# 🚨 FIXED: STRICT EXPIRY VALIDATION FUNCTIONS 🚨
def validate_option_symbol(index, symbol, strike, opttype):
    """STRICT validation to ensure ONLY specified expiry symbols are used"""
    try:
        # Get the expected expiry for this index
        expected_expiry = EXPIRIES.get(index)
        if not expected_expiry:
            return False
            
        # Parse expected expiry date
        expected_dt = datetime.strptime(expected_expiry, "%d %b %Y")
        
        # STRICT CHECK: For SENSEX: SENSEX25NOV25000CE format
        if index == "SENSEX":
            year_short = expected_dt.strftime("%y")  # 25
            month_code = expected_dt.strftime("%b").upper()  # NOV
            day = expected_dt.strftime("%d")  # 25
            expected_pattern = f"SENSEX{day}{month_code}{year_short}"
            symbol_upper = symbol.upper()
            
            # Check if symbol contains EXACTLY this pattern
            if expected_pattern in symbol_upper:
                return True
            else:
                print(f"❌ SENSEX expiry mismatch: Expected {expected_pattern}, Got {symbol_upper}")
                return False
        else:
            # STRICT CHECK: For NIFTY/BANKNIFTY/MIDCPNIFTY: NIFTY25NOV2521500CE format
            expected_pattern = expected_dt.strftime("%d%b%y").upper()  # 25NOV25
            symbol_upper = symbol.upper()
            
            # Check if symbol contains EXACTLY this pattern
            if expected_pattern in symbol_upper:
                return True
            else:
                print(f"❌ {index} expiry mismatch: Expected {expected_pattern}, Got {symbol_upper}")
                return False
                
    except Exception as e:
        print(f"Symbol validation error: {e}")
        return False

# 🚨 FIXED: GET OPTION SYMBOL WITH STRICT EXPIRY VALIDATION 🚨
def get_option_symbol(index, expiry_str, strike, opttype):
    """STRICTLY generates symbols ONLY with specified expiries"""
    try:
        dt = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = dt.strftime("%y")  # 25
            month_code = dt.strftime("%b").upper()  # NOV
            day = dt.strftime("%d")  # 25
            symbol = f"SENSEX{day}{month_code}{year_short}{strike}{opttype}"
        elif index == "MIDCPNIFTY":
            symbol = f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        else:
            symbol = f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        
        # STRICT VALIDATION: Validate the generated symbol
        if validate_option_symbol(index, symbol, strike, opttype):
            print(f"✅ Valid symbol generated: {symbol}")
            return symbol
        else:
            print(f"❌ Generated symbol validation FAILED: {symbol}")
            return None
            
    except Exception as e:
        print(f"Error generating symbol: {e}")
        return None

# --------- DETECT LIQUIDITY ZONE ---------
def detect_liquidity_zone(df, lookback=20):
    high_series = ensure_series(df['High']).dropna()
    low_series = ensure_series(df['Low']).dropna()
    try:
        if len(high_series) <= lookback:
            high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
        else:
            high_pool = float(high_series.rolling(lookback).max().iloc[-2])
    except Exception:
        high_pool = float(high_series.max()) if len(high_series)>0 else float('nan')
    try:
        if len(low_series) <= lookback:
            low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')
        else:
            low_pool = float(low_series.rolling(lookback).min().iloc[-2])
    except Exception:
        low_pool = float(low_series.min()) if len(low_series)>0 else float('nan')

    if math.isnan(high_pool) and len(high_series)>0:
        high_pool = float(high_series.max())
    if math.isnan(low_pool) and len(low_series)>0:
        low_pool = float(low_series.min())

    return round(high_pool,0), round(low_pool,0)

# --------- INSTITUTIONAL LIQUIDITY HUNT ---------
def institutional_liquidity_hunt(index, df):
    prev_high = None
    prev_low = None
    try:
        prev_high_val = ensure_series(df['High']).iloc[-2]
        prev_low_val = ensure_series(df['Low']).iloc[-2]
        prev_high = float(prev_high_val) if not (isinstance(prev_high_val,float) and math.isnan(prev_high_val)) else None
        prev_low = float(prev_low_val) if not (isinstance(prev_low_val,float) and math.isnan(prev_low_val)) else None
    except Exception:
        prev_high = None
        prev_low = None

    high_zone, low_zone = detect_liquidity_zone(df, lookback=15)

    last_close_val = None
    try:
        lc = ensure_series(df['Close']).iloc[-1]
        if isinstance(lc, float) and math.isnan(lc):
            last_close_val = None
        else:
            last_close_val = float(lc)
    except Exception:
        last_close_val = None

    if last_close_val is None:
        highest_ce_oi_strike = None
        highest_pe_oi_strike = None
    else:
        highest_ce_oi_strike = round_strike(index, last_close_val + 50)
        highest_pe_oi_strike = round_strike(index, last_close_val - 50)

    bull_liquidity = []
    if prev_low is not None: bull_liquidity.append(prev_low)
    if low_zone is not None: bull_liquidity.append(low_zone)
    if highest_pe_oi_strike is not None: bull_liquidity.append(highest_pe_oi_strike)

    bear_liquidity = []
    if prev_high is not None: bear_liquidity.append(prev_high)
    if high_zone is not None: bear_liquidity.append(high_zone)
    if highest_ce_oi_strike is not None: bear_liquidity.append(highest_ce_oi_strike)

    return bull_liquidity, bear_liquidity

def liquidity_zone_entry_check(price, bull_liq, bear_liq):
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None

    for zone in bull_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:
                return "CE"
        except:
            continue
    for zone in bear_liq:
        if zone is None: continue
        try:
            if abs(price - zone) <= 5:
                return "PE"
        except:
            continue

    valid_bear = [z for z in bear_liq if z is not None]
    valid_bull = [z for z in bull_liq if z is not None]
    if valid_bear and valid_bull:
        try:
            if price > max(valid_bear) or price < min(valid_bull):
                return "BOTH"
        except:
            return None
    return None

# 🚨 NEW: INSTITUTIONAL ENTRY FILTERS 🚨
def institutional_entry_filters(df, signal_type):
    """
    INSTITUTIONAL GRADE ENTRY FILTERS
    - Volume confirmation
    - Price momentum filter
    - Liquidity sweep confirmation
    - Institutional order flow
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return False
            
        # 1. VOLUME CONFIRMATION (1.5x average volume)
        vol_avg = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        vol_ratio = current_vol / (vol_avg if vol_avg > 0 else 1)
        
        if vol_ratio < MIN_VOLUME_RATIO:
            return False
            
        # 2. PRICE MOMENTUM FILTER (Strong candle body)
        current_body = abs(close.iloc[-1] - close.iloc[-2])
        current_range = high.iloc[-1] - low.iloc[-1]
        body_ratio = current_body / (current_range if current_range > 0 else 1)
        
        if body_ratio < MIN_CANDLE_BODY_RATIO:
            return False
            
        # 3. LIQUIDITY SWEEP CONFIRMATION
        recent_high = high.iloc[-10:-1].max()
        recent_low = low.iloc[-10:-1].min()
        
        if signal_type == "CE":
            # For CE: Check if we swept lows before moving up
            sweep_low = low.iloc[-3:].min()
            if sweep_low <= recent_low * 0.998:
                return True
        elif signal_type == "PE":
            # For PE: Check if we swept highs before moving down
            sweep_high = high.iloc[-3:].max()
            if sweep_high >= recent_high * 1.002:
                return True
                
        # 4. INSTITUTIONAL ORDER FLOW (Big money participation)
        # Check for consecutive strong moves in signal direction
        if signal_type == "CE":
            if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
                return False
        elif signal_type == "PE":
            if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
                return False
                
        return True
        
    except Exception:
        return False

# 🚨 NEW: RETAIL SL HUNT DETECTION 🚨
def detect_retail_sl_hunt(index, df):
    """
    DETECT RETAIL STOP LOSS HUNTS
    - Find where retail SL clusters are
    - Wait for liquidity grab at these levels
    - Enter after retail stops are taken
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 15:
            return None
            
        # 1. IDENTIFY RETAIL SL CLUSTERS (Common round numbers)
        current_price = close.iloc[-1]
        rounded_levels = []
        
        # Common retail SL levels (round numbers)
        for i in range(-100, 101, 50):
            level = round(current_price + i, -2)  # Round to nearest 100
            rounded_levels.append(level)
            
        # 2. CHECK FOR LIQUIDITY SWEEPS AT THESE LEVELS
        recent_high = high.iloc[-5:].max()
        recent_low = low.iloc[-5:].min()
        
        for level in rounded_levels:
            # Check if level was recently swept
            if (recent_high >= level * 1.002 and 
                close.iloc[-1] < level * 0.998 and
                volume.iloc[-1] > volume.iloc[-10:].mean() * 1.8):
                # Bearish SL hunt detected - Good for PE
                return "PE"
                
            if (recent_low <= level * 0.998 and 
                close.iloc[-1] > level * 1.002 and
                volume.iloc[-1] > volume.iloc[-10:].mean() * 1.8):
                # Bullish SL hunt detected - Good for CE
                return "CE"
                
        return None
        
    except Exception:
        return None

# 🚨 NEW: INSTITUTIONAL REVERSAL STRATEGY 🚨
def institutional_reversal_strategy(index, df):
    """
    PERFECT REVERSAL ENTRIES AFTER STRONG MOVES
    - After strong up moves: Perfect PE entries from top
    - After strong down moves: Perfect CE entries from bottom
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 20:
            return None
            
        # CALCULATE RECENT MOVE STRENGTH
        price_5_bars_ago = close.iloc[-6]
        price_now = close.iloc[-1]
        move_percentage = (price_now - price_5_bars_ago) / price_5_bars_ago
        
        # 1. AFTER STRONG UP MOVE (Look for PE entry from top)
        if move_percentage > STRONG_MOVE_THRESHOLD:
            # Check for exhaustion signals
            current_high = high.iloc[-1]
            prev_high = high.iloc[-2]
            current_low = low.iloc[-1]
            current_close = close.iloc[-1]
            
            # Exhaustion criteria:
            # - New high but weak close
            # - Long upper wick
            # - Volume spike
            upper_wick = current_high - max(current_close, close.iloc[-2])
            body_size = abs(current_close - close.iloc[-2])
            
            if (current_high > prev_high and
                current_close < (current_high + current_low) / 2 and
                upper_wick > body_size * 1.2 and
                volume.iloc[-1] > volume.iloc[-10:].mean() * 1.5):
                return "PE"  # Perfect PE entry from top
                
        # 2. AFTER STRONG DOWN MOVE (Look for CE entry from bottom)
        elif move_percentage < -STRONG_MOVE_THRESHOLD:
            # Check for accumulation signals
            current_low = low.iloc[-1]
            prev_low = low.iloc[-2]
            current_high = high.iloc[-1]
            current_close = close.iloc[-1]
            
            # Accumulation criteria:
            # - New low but strong close
            # - Long lower wick
            # - Volume spike
            lower_wick = min(current_close, close.iloc[-2]) - current_low
            body_size = abs(current_close - close.iloc[-2])
            
            if (current_low < prev_low and
                current_close > (current_high + current_low) / 2 and
                lower_wick > body_size * 1.2 and
                volume.iloc[-1] > volume.iloc[-10:].mean() * 1.5):
                return "CE"  # Perfect CE entry from bottom
                
        return None
        
    except Exception:
        return None

# 🚨 NEW: INSTITUTIONAL TARGET & SL CALCULATION 🚨
def calculate_institutional_targets_sl(index, entry_price, signal_type, df):
    """
    INSTITUTIONAL GRADE TARGETS & SL
    - Realistic, achievable targets (20-50% moves)
    - Smart SL placement below key technical levels
    - Better risk-reward (1:1.5 instead of 1:3+)
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        # USE ATR FOR REALISTIC TARGET SIZING
        atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        
        # INSTITUTIONAL TARGETS (Realistic 20-50% moves)
        base_move = atr * 0.8  # Conservative base move
        
        if signal_type == "CE":
            targets = [
                round(entry_price + base_move * ATR_TARGET_MULTIPLIER[0]),
                round(entry_price + base_move * ATR_TARGET_MULTIPLIER[1]),
                round(entry_price + base_move * ATR_TARGET_MULTIPLIER[2]),
                round(entry_price + base_move * ATR_TARGET_MULTIPLIER[3])
            ]
            # SMART SL: Below recent swing low or liquidity zone
            recent_low = low.iloc[-5:].min()
            sl = round(min(entry_price - base_move * ATR_SL_MULTIPLIER, recent_low * 0.998))
            
        else:  # PE
            targets = [
                round(entry_price + base_move * ATR_TARGET_MULTIPLIER[0]),
                round(entry_price + base_move * ATR_TARGET_MULTIPLIER[1]),
                round(entry_price + base_move * ATR_TARGET_MULTIPLIER[2]),
                round(entry_price + base_move * ATR_TARGET_MULTIPLIER[3])
            ]
            # SMART SL: Above recent swing high or liquidity zone
            recent_high = high.iloc[-5:].max()
            sl = round(max(entry_price - base_move * ATR_SL_MULTIPLIER, recent_high * 1.002))
        
        # ENSURE MINIMUM TARGET/SL DIFFERENCE
        min_diff = entry_price * 0.05  # At least 5% difference
        for i in range(len(targets)):
            if abs(targets[i] - entry_price) < min_diff:
                targets[i] = round(entry_price + min_diff * (1 if signal_type == "CE" else -1) * (i + 1))
                
        if abs(sl - entry_price) < min_diff:
            sl = round(entry_price - min_diff * (1 if signal_type == "CE" else -1))
            
        return targets, sl
        
    except Exception:
        # FALLBACK: Conservative targets
        if signal_type == "CE":
            return [
                round(entry_price * 1.15),
                round(entry_price * 1.25),
                round(entry_price * 1.35),
                round(entry_price * 1.45)
            ], round(entry_price * 0.92)
        else:
            return [
                round(entry_price * 1.15),
                round(entry_price * 1.25),
                round(entry_price * 1.35),
                round(entry_price * 1.45)
            ], round(entry_price * 0.92)

# 🚨 NEW: INSTITUTIONAL P&L CALCULATION 🚨
def calculate_institutional_pnl(signal_data):
    """
    Calculate institutional P&L based on index-specific parameters
    """
    try:
        index = signal_data.get('index')
        entry_price = signal_data.get('entry_price')
        sl_price = signal_data.get('sl')
        max_price_reached = signal_data.get('max_price_reached', entry_price)
        entry_status = signal_data.get('entry_status', 'PENDING')
        
        if index not in INDEX_PNL_PARAMS or entry_status != 'ENTERED':
            return 0, 0, 0
            
        params = INDEX_PNL_PARAMS[index]
        max_profit_points = params['max_profit_points']
        quantity = params['quantity']
        
        # Calculate price movement from entry
        price_movement = max_price_reached - entry_price
        
        # PROFIT CALCULATION: If price moved beyond max profit points, cap at max profit
        if price_movement >= max_profit_points:
            profit = max_profit_points * quantity
            loss = 0
        # LOSS CALCULATION: If SL was hit
        elif max_price_reached <= sl_price:
            loss_points = entry_price - sl_price
            profit = 0
            loss = loss_points * quantity
        # PARTIAL PROFIT: If price moved but didn't reach max profit
        elif price_movement > 0:
            profit = price_movement * quantity
            loss = 0
        # NO MOVEMENT
        else:
            profit = 0
            loss = 0
            
        investment = entry_price * quantity
        
        return investment, profit, loss
        
    except Exception as e:
        print(f"P&L calculation error: {e}")
        return 0, 0, 0

# 🚨 ENHANCED: INSTITUTIONAL PRICE ACTION LAYER 🚨
def institutional_price_action_signal(df):
    """
    Pure price action based institutional signals
    Focuses on breakouts, rejections, and momentum
    """
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        # Recent price range
        recent_high = high.iloc[-10:-1].max()
        recent_low = low.iloc[-10:-1].min()
        current_close = close.iloc[-1]
        
        # Volume analysis
        vol_avg = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # 🚨 INSTITUTIONAL BREAKOUT DETECTION
        if (current_close > recent_high and 
            current_vol > vol_avg * 1.8 and
            current_close > close.iloc[-2] and
            close.iloc[-2] > close.iloc[-3]):
            return "CE"
            
        # 🚨 INSTITUTIONAL BREAKDOWN DETECTION  
        if (current_close < recent_low and
            current_vol > vol_avg * 1.8 and
            current_close < close.iloc[-2] and
            close.iloc[-2] < close.iloc[-3]):
            return "PE"
            
        # 🚨 STRONG REJECTION PATTERNS
        current_body = abs(close.iloc[-1] - close.iloc[-2])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], close.iloc[-2])
        lower_wick = min(close.iloc[-1], close.iloc[-2]) - low.iloc[-1]
        
        # Strong rejection at highs
        if (upper_wick > current_body * 1.5 and
            current_vol > vol_avg * 1.5 and
            close.iloc[-1] < close.iloc[-2]):
            return "PE"
            
        # Strong rejection at lows
        if (lower_wick > current_body * 1.5 and
            current_vol > vol_avg * 1.5 and
            close.iloc[-1] > close.iloc[-2]):
            return "CE"
            
    except Exception:
        return None
    return None

# 🚨 ENHANCED: INSTITUTIONAL MOMENTUM CONFIRMATION 🚨
def institutional_momentum_confirmation(index, df, proposed_signal):
    """
    Final institutional confirmation layer
    """
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 5:
            return False
            
        # Apply institutional entry filters
        if not institutional_entry_filters(df, proposed_signal):
            return False
            
        # Price momentum confirmation
        if proposed_signal == "CE":
            # For CE: require upward momentum
            if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
                return False
            # Strong bullish candle
            if (high.iloc[-1] - low.iloc[-1]) < (high.iloc[-2] - low.iloc[-2]) * 0.7:
                return False
                
        elif proposed_signal == "PE":
            # For PE: require downward momentum
            if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
                return False
            # Strong bearish candle
            if (high.iloc[-1] - low.iloc[-1]) < (high.iloc[-2] - low.iloc[-2]) * 0.7:
                return False
                
        return True
        
    except Exception:
        return False

# 🚨 LAYER 1: LIQUIDITY SWEEPS 🚨
def detect_liquidity_sweeps(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
            
        recent_highs = high.iloc[-10:-2]
        recent_lows = low.iloc[-10:-2]
        
        liquidity_high = recent_highs.max()
        liquidity_low = recent_lows.min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        # STRICTER LIQUIDITY SWEEP CONDITIONS
        if (current_high > liquidity_high * (1 + LIQUIDITY_SWEEP_DISTANCE) and
            current_close < liquidity_high * 0.998 and
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.6):
            return "PE"
            
        if (current_low < liquidity_low * (1 - LIQUIDITY_SWEEP_DISTANCE) and
            current_close > liquidity_low * 1.002 and
            volume.iloc[-1] > volume.iloc[-10:-1].mean() * 1.6):
            return "CE"
    except Exception:
        return None
    return None

# 🚨 LAYER 2: OTE RETRACEMENT 🚨
def detect_ote_retracement(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        
        if len(close) < 15:
            return None
            
        swing_high = high.iloc[-15:-5].max()
        swing_low = low.iloc[-15:-5].min()
        swing_range = swing_high - swing_low
        
        current_price = close.iloc[-1]
        
        for level in OTE_RETRACEMENT_LEVELS:
            ote_level = swing_high - (swing_range * level)
            
            if (abs(current_price - ote_level) / ote_level < 0.0015 and  # Tighter tolerance
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3]):
                return "CE"
                
            ote_level = swing_low + (swing_range * level)
            if (abs(current_price - ote_level) / ote_level < 0.0015 and  # Tighter tolerance
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3]):
                return "PE"
    except Exception:
        return None
    return None

# 🚨 LAYER 3: DEMAND AND SUPPLY ZONES 🚨
def detect_demand_supply_zones(df):
    try:
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < DEMAND_SUPPLY_ZONE_LOOKBACK + 5:
            return None
            
        lookback = DEMAND_SUPPLY_ZONE_LOOKBACK
        
        demand_lows = low.rolling(3, center=True).min().dropna()
        significant_demand = demand_lows[demand_lows == demand_lows.rolling(5).min()]
        
        supply_highs = high.rolling(3, center=True).max().dropna()
        significant_supply = supply_highs[supply_highs == supply_highs.rolling(5).max()]
        
        current_price = close.iloc[-1]
        
        # STRICTER ZONE CONDITIONS
        for zone in significant_demand.iloc[-5:]:
            if (abs(current_price - zone) / zone < 0.002 and  # Tighter tolerance
                close.iloc[-1] > close.iloc[-2] and
                close.iloc[-1] > close.iloc[-3] and  # Additional confirmation
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.4):
                return "CE"
                
        for zone in significant_supply.iloc[-5:]:
            if (abs(current_price - zone) / zone < 0.002 and  # Tighter tolerance
                close.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] < close.iloc[-3] and  # Additional confirmation
                volume.iloc[-1] > volume.iloc[-5:].mean() * 1.4):
                return "PE"
    except Exception:
        return None
    return None

# --------- NEW: SIGNAL DEDUPLICATION AND COOLDOWN CHECK ---------
def can_send_signal(index, strike, option_type):
    """Check if we can send signal based on deduplication and cooldown rules"""
    global active_strikes, last_signal_time
    
    current_time = time.time()
    strike_key = f"{index}_{strike}_{option_type}"
    
    # Check if same strike is already active
    if strike_key in active_strikes:
        return False
        
    # Check cooldown for this index
    if index in last_signal_time:
        time_since_last = current_time - last_signal_time[index]
        if time_since_last < signal_cooldown:
            return False
    
    return True

def update_signal_tracking(index, strike, option_type, signal_id):
    """Update tracking for sent signals"""
    global active_strikes, last_signal_time
    
    strike_key = f"{index}_{strike}_{option_type}"
    active_strikes[strike_key] = {
        'signal_id': signal_id,
        'timestamp': time.time(),
        'targets_hit': 0
    }
    
    last_signal_time[index] = time.time()

def update_signal_progress(signal_id, targets_hit):
    """Update progress of active signal"""
    for strike_key, data in active_strikes.items():
        if data['signal_id'] == signal_id:
            active_strikes[strike_key]['targets_hit'] = targets_hit
            break

def clear_completed_signal(signal_id):
    """Clear signal from active tracking when completed"""
    global active_strikes
    active_strikes = {k: v for k, v in active_strikes.items() if v['signal_id'] != signal_id}

# --------- UPDATED STRATEGY CHECK WITH INSTITUTIONAL LAYERS ---------
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    last_close = float(close5.iloc[-1])
    prev_close = float(close5.iloc[-2])

    # 🚨 NEW: TIME-BASED FILTER - Avoid late day unreliable signals
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time = ist_now.time()
        # Avoid signals in last 45 minutes (low reliability)
        if current_time >= dtime(14, 45):
            return None
    except:
        pass

    # 🚨 NEW: INSTITUTIONAL REVERSAL STRATEGY (HIGHEST PRIORITY)
    reversal_signal = institutional_reversal_strategy(index, df5)
    if reversal_signal:
        if institutional_momentum_confirmation(index, df5, reversal_signal):
            return reversal_signal, df5, False, "institutional_reversal"

    # 🚨 NEW: RETAIL SL HUNT DETECTION (HIGH PRIORITY)
    sl_hunt_signal = detect_retail_sl_hunt(index, df5)
    if sl_hunt_signal:
        if institutional_momentum_confirmation(index, df5, sl_hunt_signal):
            return sl_hunt_signal, df5, False, "retail_sl_hunt"

    # 🚨 NEW: INSTITUTIONAL PRICE ACTION (HIGH PRIORITY) 🚨
    institutional_pa_signal = institutional_price_action_signal(df5)
    if institutional_pa_signal:
        if institutional_momentum_confirmation(index, df5, institutional_pa_signal):
            return institutional_pa_signal, df5, False, "institutional_price_action"

    # 🚨 LAYER 1: LIQUIDITY SWEEPS 🚨
    sweep_sig = detect_liquidity_sweeps(df5)
    if sweep_sig:
        if institutional_momentum_confirmation(index, df5, sweep_sig):
            return sweep_sig, df5, True, "liquidity_sweeps"

    # 🚨 LAYER 2: OTE RETRACEMENT 🚨
    ote_sig = detect_ote_retracement(df5)
    if ote_sig:
        if institutional_momentum_confirmation(index, df5, ote_sig):
            return ote_sig, df5, False, "ote_retracement"

    # 🚨 LAYER 3: DEMAND & SUPPLY ZONES 🚨
    ds_sig = detect_demand_supply_zones(df5)
    if ds_sig:
        if institutional_momentum_confirmation(index, df5, ds_sig):
            return ds_sig, df5, False, "demand_supply_zones"

    # Final fallback: Liquidity-based entry
    bull_liq, bear_liq = institutional_liquidity_hunt(index, df5)
    liquidity_side = liquidity_zone_entry_check(last_close, bull_liq, bear_liq)
    if liquidity_side:
        return liquidity_side, df5, False, "liquidity_zone"

    return None

# --------- INSTITUTIONAL FLOW CHECKS ---------
def institutional_flow_signal(index, df5):
    try:
        last_close = float(ensure_series(df5["Close"]).iloc[-1])
        prev_close = float(ensure_series(df5["Close"]).iloc[-2])
    except:
        return None

    vol5 = ensure_series(df5["Volume"])
    vol_latest = float(vol5.iloc[-1])
    vol_avg = float(vol5.rolling(20).mean().iloc[-1]) if len(vol5) >= 20 else float(vol5.mean())

    # STRICTER FLOW CONDITIONS
    if vol_latest > vol_avg*2.0 and abs(last_close-prev_close)/prev_close>0.005:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg*1.5:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg*1.5:
        return "PE"
    
    high_zone, low_zone = detect_liquidity_zone(df5, lookback=15)
    try:
        if last_close>=high_zone: return "PE"
        elif last_close<=low_zone: return "CE"
    except:
        return None
    return None

# --------- OI + DELTA FLOW DETECTION ---------
def oi_delta_flow_signal(index):
    try:
        url=f"https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        df_index=df[df['symbol'].str.contains(index)]
        if 'oi' not in df_index.columns:
            return None
        df_index['oi'] = pd.to_numeric(df_index['oi'], errors='coerce').fillna(0)
        df_index['oi_change'] = df_index['oi'].diff().fillna(0)
        ce_sum = df_index[df_index['symbol'].str.endswith("CE")]['oi_change'].sum()
        pe_sum = df_index[df_index['symbol'].str.endswith("PE")]['oi_change'].sum()
        # STRICTER OI CONDITIONS
        if ce_sum>pe_sum*DELTA_OI_RATIO: return "CE"
        if pe_sum>ce_sum*DELTA_OI_RATIO: return "PE"
        if ce_sum>0 and pe_sum>0: return "BOTH"
    except:
        return None

# --------- SIMPLIFIED CONFIRMATION ---------
def institutional_confirmation_layer(index, df5, base_signal):
    try:
        close = ensure_series(df5['Close'])
        last_close = float(close.iloc[-1])
        
        high_zone, low_zone = detect_liquidity_zone(df5, lookback=20)
        if base_signal == 'CE' and last_close >= high_zone:
            return False
        if base_signal == 'PE' and last_close <= low_zone:
            return False

        return True
    except Exception:
        return False

def institutional_flow_confirm(index, base_signal, df5):
    flow = institutional_flow_signal(index, df5)
    oi_flow = oi_delta_flow_signal(index)

    if flow and flow != 'BOTH' and flow != base_signal:
        return False
    if oi_flow and oi_flow != 'BOTH' and oi_flow != base_signal:
        return False

    if not institutional_confirmation_layer(index, df5, base_signal):
        return False

    return True

# --------- FIXED: ENHANCED TRADE MONITORING AND TRACKING ---------
active_trades = {}

def calculate_pnl(entry, max_price, targets, targets_hit, sl):
    try:
        if targets is None or len(targets) == 0:
            diff = max_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        
        if not isinstance(targets_hit, (list, tuple)):
            targets_hit = list(targets_hit) if targets_hit is not None else [False]*len(targets)
        if len(targets_hit) < len(targets):
            targets_hit = list(targets_hit) + [False] * (len(targets) - len(targets_hit))
        
        achieved_prices = [target for i, target in enumerate(targets) if targets_hit[i]]
        if achieved_prices:
            exit_price = achieved_prices[-1]
            diff = exit_price - entry
            if diff > 0:
                return f"+{diff:.2f}"
            elif diff < 0:
                return f"-{abs(diff):.2f}"
            else:
                return "0"
        else:
            if max_price <= sl:
                diff = sl - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
            else:
                diff = max_price - entry
                if diff > 0:
                    return f"+{diff:.2f}"
                elif diff < 0:
                    return f"-{abs(diff):.2f}"
                else:
                    return "0"
    except Exception:
        return "0"

def monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data):
    def monitoring_thread():
        global daily_signals
        
        last_high = entry
        weakness_sent = False
        in_trade = False
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        last_activity_time = time.time()
        signal_id = signal_data.get('signal_id')
        
        while True:
            current_time = time.time()
            
            # Check for inactivity (20 minutes)
            if not in_trade and (current_time - last_activity_time) > 1200:  # 20 minutes
                send_telegram(f"⏰ {symbol}: No activity for 20 minutes. Allowing new signals.", reply_to=thread_id)
                clear_completed_signal(signal_id)
                break
                
            if should_stop_trading():
                try:
                    final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                except Exception:
                    final_pnl = "0"
                signal_data.update({
                    "entry_status": "NOT_ENTERED" if not entry_price_achieved else "ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price_reached,
                    "zero_targets": sum(targets_hit) == 0,
                    "no_new_highs": max_price_reached <= entry,
                    "final_pnl": final_pnl
                })
                daily_signals.append(signal_data)
                clear_completed_signal(signal_id)
                break
                
            price = fetch_option_price(symbol)
            if price:
                last_activity_time = current_time
                price = round(price)
                
                if price > max_price_reached:
                    max_price_reached = price
                
                if not in_trade:
                    if price >= entry:
                        send_telegram(f"✅ ENTRY TRIGGERED at {price}", reply_to=thread_id)
                        in_trade = True
                        entry_price_achieved = True
                        last_high = price
                        signal_data["entry_status"] = "ENTERED"
                else:
                    if price > last_high:
                        send_telegram(f"🚀 {symbol} making new high → {price}", reply_to=thread_id)
                        last_high = price
                    elif not weakness_sent and price < sl * 1.05:
                        send_telegram(f"⚡ {symbol} showing weakness near SL {sl}", reply_to=thread_id)
                        weakness_sent = True
                    
                    # Update signal progress
                    current_targets_hit = sum(targets_hit)
                    for i, target in enumerate(targets):
                        if price >= target and not targets_hit[i]:
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                            targets_hit[i] = True
                            current_targets_hit = sum(targets_hit)
                            update_signal_progress(signal_id, current_targets_hit)
                    
                    # SL hit - allow immediate new signal
                    if price <= sl:
                        send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade. ALLOWING NEW SIGNAL.", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": sum(targets_hit),
                            "max_price_reached": max_price_reached,
                            "zero_targets": sum(targets_hit) == 0,
                            "no_new_highs": max_price_reached <= entry,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)  # Clear for new signal
                        break
                        
                    # 2nd target hit - allow new signals but continue monitoring
                    if current_targets_hit >= 2:
                        update_signal_progress(signal_id, current_targets_hit)
                        # Continue monitoring but new signals allowed
                    
                    # All targets hit - complete trade
                    if all(targets_hit):
                        send_telegram(f"🏆 {symbol}: ALL TARGETS HIT! Trade completed successfully!", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": len(targets),
                            "max_price_reached": max_price_reached,
                            "zero_targets": False,
                            "no_new_highs": False,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# 🚨 ENHANCED: INSTITUTIONAL P&L EOD REPORT 🚨
def send_individual_signal_reports():
    """Send each signal in separate detailed messages after market hours"""
    global daily_signals, all_generated_signals
    
    # 🚨 CRITICAL FIX: Combine both signal sources
    all_signals = daily_signals + all_generated_signals
    
    # Remove duplicates based on signal_id
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
    
    # Send header message
    send_telegram(f"🕒 END OF DAY SIGNAL REPORT - { (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y') }\n"
                  f"📈 Total Signals: {len(unique_signals)}\n"
                  f"─────────────────────────────")
    
    total_investment = 0
    total_profit = 0
    total_loss = 0
    
    # Send each signal in separate message
    for i, signal in enumerate(unique_signals, 1):
        targets_hit_list = []
        if signal.get('targets_hit', 0) > 0:
            for j in range(signal.get('targets_hit', 0)):
                if j < len(signal.get('targets', [])):
                    targets_hit_list.append(str(signal['targets'][j]))
        
        targets_for_disp = signal.get('targets', [])
        while len(targets_for_disp) < 4:
            targets_for_disp.append('-')
        
        # Calculate institutional P&L
        investment, profit, loss = calculate_institutional_pnl(signal)
        total_investment += investment
        total_profit += profit
        total_loss += loss
        
        msg = (f"📊 SIGNAL #{i} - {signal.get('index','?')} {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"─────────────────────────────\n"
               f"📅 Date: {(datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y')}\n"
               f"🕒 Time: {signal.get('timestamp','?')}\n"
               f"📈 Index: {signal.get('index','?')}\n"
               f"🎯 Strike: {signal.get('strike','?')}\n"
               f"🔰 Type: {signal.get('option_type','?')}\n"
               f"🏷️ Strategy: {signal.get('strategy','?')}\n\n"
               
               f"💰 ENTRY: ₹{signal.get('entry_price','?')}\n"
               f"🎯 TARGETS: {targets_for_disp[0]} // {targets_for_disp[1]} // {targets_for_disp[2]} // {targets_for_disp[3]}\n"
               f"🛑 STOP LOSS: ₹{signal.get('sl','?')}\n\n"
               
               f"📊 PERFORMANCE:\n"
               f"• Entry Status: {signal.get('entry_status', 'PENDING')}\n"
               f"• Targets Hit: {signal.get('targets_hit', 0)}/4\n")
        
        if targets_hit_list:
            msg += f"• Targets Achieved: {', '.join(targets_hit_list)}\n"
        
        msg += (f"• Max Price Reached: ₹{signal.get('max_price_reached', signal.get('entry_price','?'))}\n"
                f"• Final P&L: {signal.get('final_pnl', '0')} points\n\n"
                
                f"💼 INSTITUTIONAL P&L:\n"
                f"• Investment: ₹{investment:,.0f}\n"
                f"• Profit: ₹{profit:,.0f}\n" 
                f"• Loss: ₹{loss:,.0f}\n"
                f"• Net P&L: ₹{profit - loss:+,.0f}\n\n"
                
                f"⚡ Fakeout: {'YES' if signal.get('fakeout') else 'NO'}\n"
                f"📈 Index Price at Signal: {signal.get('index_price','?')}\n"
                f"🆔 Signal ID: {signal.get('signal_id','?')}\n"
                f"─────────────────────────────")
        
        send_telegram(msg)
        time.sleep(1)
    
    # Send institutional summary
    net_pnl = total_profit - total_loss
    
    summary_msg = (f"📈 DAY SUMMARY\n"
                   f"─────────────────────────────\n"
                   f"• Total Signals: {len(unique_signals)}\n"
                   f"• Total Investment: ₹{total_investment:,.0f}\n"
                   f"• Total Profit: ₹{total_profit:,.0f}\n"
                   f"• Total Loss: ₹{total_loss:,.0f}\n"
                   f"• Net P&L: ₹{net_pnl:+,.0f}\n"
                   f"• ROI: {(net_pnl/total_investment*100 if total_investment > 0 else 0):+.1f}%\n"
                   f"─────────────────────────────")
    
    send_telegram(summary_msg)
    
    # 🚨 COMPULSORY CONFIRMATION
    send_telegram("✅ END OF DAY REPORTS COMPLETED! See you tomorrow at 9:15 AM! 🚀")

# 🚨 FIXED: UPDATED SIGNAL SENDING WITH INSTITUTIONAL TARGETS 🚨
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    # 🚨 CRITICAL FIX: Each index uses its OWN isolated strike calculation
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
    strike = round_strike(index, signal_detection_price)
    
    if strike is None:
        send_telegram(f"⚠️ {index}: could not determine strike (price missing). Signal skipped.")
        return
        
    # 🚨 CHECK DEDUPLICATION AND COOLDOWN
    if not can_send_signal(index, strike, side):
        return
        
    # 🚨 FIXED: STRICT EXPIRY ENFORCEMENT - Only use specified expiries
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    
    if symbol is None:
        # 🚨 SILENT REJECTION - No Telegram message for wrong expiry!
        print(f"❌ STRICT EXPIRY ENFORCEMENT: {index} {strike}{side} - Only {EXPIRIES[index]} allowed")
        return  # Just exit quietly without sending any message
    
    option_price = fetch_option_price(symbol)
    if not option_price: 
        return
    
    entry = round(option_price)
    
    # 🚨 INSTITUTIONAL TARGETS & SL CALCULATION
    targets, sl = calculate_institutional_targets_sl(index, entry, side, df)
    
    targets_str = "//".join(str(t) for t in targets) + "++"
    
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key.upper())
    
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
        "fakeout": fakeout,
        "index_price": signal_detection_price,
        "entry_status": "PENDING",
        "targets_hit": 0,
        "max_price_reached": entry,
        "zero_targets": True,
        "no_new_highs": True,
        "final_pnl": "0"
    }
    
    # 🚨 UPDATE SIGNAL TRACKING
    update_signal_tracking(index, strike, side, signal_id)
    
    # 🚨 FIX: Track signal immediately for EOD reports
    all_generated_signals.append(signal_data.copy())
    
    msg = (f"🟢 {index} {strike} {side}\n"
           f"SYMBOL: {symbol}\n"
           f"ABOVE {entry}\n"
           f"TARGETS: {targets_str}\n"
           f"SL: {sl}\n"
           f"FAKEOUT: {'YES' if fakeout else 'NO'}\n"
           f"STRATEGY: {strategy_name}\n"
           f"SIGNAL ID: {signal_id}")
         
    thread_id = send_telegram(msg)
    
    trade_id = f"{symbol}_{int(time.time())}"
    active_trades[trade_id] = {
        "symbol": symbol, 
        "entry": entry, 
        "sl": sl, 
        "targets": targets, 
        "thread": thread_id, 
        "status": "OPEN",
        "index": index,
        "signal_data": signal_data
    }
    
    monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# --------- FIXED: UPDATED TRADE THREAD WITH ISOLATED INDICES ---------
def trade_thread(index):
    """Generate signals with completely isolated index processing"""
    result = analyze_index_signal(index)
    
    if not result:
        return
        
    if len(result) == 4:
        side, df, fakeout, strategy_key = result
    else:
        side, df, fakeout = result
        strategy_key = "unknown"
    
    # 🚨 CRITICAL FIX: Each index thread processes ONLY its own data
    # No cross-contamination between indices
    df5 = fetch_index_data(index, "5m", "2d")
    inst_signal = institutional_flow_signal(index, df5) if df5 is not None else None
    oi_signal = oi_delta_flow_signal(index)
    final_signal = oi_signal or inst_signal or side

    if final_signal == "BOTH":
        for s in ["CE", "PE"]:
            if institutional_flow_confirm(index, s, df5):
                send_signal(index, s, df, fakeout, strategy_key)
        return
    elif final_signal:
        if df is None: 
            df = df5
        if institutional_flow_confirm(index, final_signal, df5):
            send_signal(index, final_signal, df, fakeout, strategy_key)
    else:
        return

# --------- FIXED: MAIN LOOP (KEPT INDICES ONLY) ---------
def run_algo_parallel():
    if not is_market_open(): 
        print("❌ Market closed - skipping iteration")
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
            
        # 🚨 FIX: GUARANTEED EOD REPORTS
        if not EOD_REPORT_SENT:
            time.sleep(15)  # Wait for all monitoring threads to complete
            send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
            try:
                send_individual_signal_reports()
            except Exception as e:
                send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                time.sleep(10)
                send_individual_signal_reports()  # Retry once
            EOD_REPORT_SENT = True
            send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
        return
        
    threads = []
    # 🚨 ONLY KEPT INDICES
    kept_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in kept_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# --------- FIXED: START WITH WORKING EOD SYSTEM ---------
STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

# Initialize strategy tracking
initialize_strategy_tracking()

while True:
    try:
        # Get current IST time
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        current_datetime_ist = ist_now
        
        # Check if market is open
        market_open = is_market_open()
        
        # 🚨 MARKET CLOSED BEHAVIOR
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market is currently closed. Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            # 🚨 COMPULSORY EOD REPORT TRIGGER BETWEEN 3:30 PM - 4:00 PM
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
                time.sleep(10)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ EOD Report completed! Algorithm will resume tomorrow.")
            
            time.sleep(30)
            continue
        
        # 🚨 MARKET OPEN BEHAVIOR
        if not STARTED_SENT:
            send_telegram("🚀 INSTITUTIONAL ENHANCED ALGO STARTED - 4 Indices Running\n"
                         "✅ Only 7 Premium Strategies Active\n"
                         "✅ Institutional P&L Calculations\n"
                         "✅ Realistic Profit Targets\n"
                         "✅ Retail SL Hunt Detection\n"
                         "✅ Perfect Reversal Entries\n"
                         "✅ Guaranteed EOD Reports\n"
                         "✅ 🚨 STRICT EXPIRY ENFORCEMENT 🚨")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        # 🚨 MARKET CLOSE DETECTION WITH GUARANTEED EOD REPORT
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached! Preparing EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            # 🚨 GUARANTEED EOD REPORT - NO EXCEPTIONS
            if not EOD_REPORT_SENT:
                send_telegram("📊 FINALIZING TRADES...")
                time.sleep(20)  # Extra time for all threads to complete
                try:
                    send_individual_signal_reports()
                except Exception as e:
                    send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                    time.sleep(10)
                    send_individual_signal_reports()  # Retry once
                EOD_REPORT_SENT = True
                send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
            time.sleep(60)
            continue
            
        # 🚨 RUN MAIN ALGORITHM DURING MARKET HOURS
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)
