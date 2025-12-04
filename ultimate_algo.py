#INDEXBASED + EOD NOT COMMING - INSTITUTIONAL BLAST ENGINE

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

# ---------------- INSTITUTIONAL CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

# 🚨 **INSTITUTIONAL BLAST DETECTION THRESHOLDS** 🚨
BLAST_VOLUME_THRESHOLD = 3.5  # 3.5x volume spike for institutional moves
BLAST_PRICE_MOVE_PCT = 0.015  # 1.5% minimum move in single candle
SWEEP_DISTANCE_PCT = 0.01     # 1.0% sweep through levels
ABSORPTION_VOLUME_RATIO = 0.5  # Volume dries to 50% before blast
IMPLOSION_RANGE_RATIO = 0.3    # Range contracts to 30% before explosion

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "09 DEC 2025",
    "BANKNIFTY": "30 DEC 2025", 
    "SENSEX": "04 DEC 2025",
    "MIDCPNIFTY": "30 DEC 2025"
}

# --------- ONLY 3 INSTITUTIONAL STRATEGIES ---------
STRATEGY_NAMES = {
    "institutional_blast": "INSTITUTIONAL BLAST",
    "volume_spike_blast": "VOLUME SPIKE BLAST",
    "sweep_order_detection": "SWEEP ORDER DETECTION"
}

# --------- INSTITUTIONAL TRACKING ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []

# --------- SIGNAL DEDUPLICATION AND COOLDOWN TRACKING ---------
active_strikes = {}
last_signal_time = {}
signal_cooldown = 1800  # 30 minutes cooldown per index

def initialize_strategy_tracking():
    global strategy_performance
    strategy_performance = {
        "INSTITUTIONAL BLAST": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "VOLUME SPIKE BLAST": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "SWEEP ORDER DETECTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
    }

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

# --------- FETCH INDEX DATA WITH MULTI TIMEFRAME ---------
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

# --------- STRICT EXPIRY VALIDATION ---------
def validate_option_symbol(index, symbol, strike, opttype):
    try:
        expected_expiry = EXPIRIES.get(index)
        if not expected_expiry:
            return False
        expected_dt = datetime.strptime(expected_expiry, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = expected_dt.strftime("%y")
            month_code = expected_dt.strftime("%b").upper()
            day = expected_dt.strftime("%d")
            expected_pattern = f"SENSEX{day}{month_code}{year_short}"
            symbol_upper = symbol.upper()
            if expected_pattern in symbol_upper:
                return True
            else:
                return False
        else:
            expected_pattern = expected_dt.strftime("%d%b%y").upper()
            symbol_upper = symbol.upper()
            if expected_pattern in symbol_upper:
                return True
            else:
                return False
    except Exception as e:
        return False

# --------- GET OPTION SYMBOL WITH STRICT EXPIRY ---------
def get_option_symbol(index, expiry_str, strike, opttype):
    try:
        dt = datetime.strptime(expiry_str, "%d %b %Y")
        
        if index == "SENSEX":
            year_short = dt.strftime("%y")
            month_code = dt.strftime("%b").upper()
            day = dt.strftime("%d")
            symbol = f"SENSEX{day}{month_code}{year_short}{strike}{opttype}"
        elif index == "MIDCPNIFTY":
            symbol = f"MIDCPNIFTY{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        else:
            symbol = f"{index}{dt.strftime('%d%b%y').upper()}{strike}{opttype}"
        
        if validate_option_symbol(index, symbol, strike, opttype):
            return symbol
        else:
            return None
    except Exception as e:
        return None

# 🚨 **NEW: INSTITUTIONAL ABSORPTION DETECTOR** 🚨
def detect_institutional_absorption(df):
    """
    Detect when institutions are accumulating before a blast
    Returns: "CE" if absorbing for up move, "PE" if for down move
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 10:
            return None
        
        # Current candle data
        current_close = close.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_volume = volume.iloc[-1]
        
        # Previous 3 candles data
        prev_close_3 = close.iloc[-4]
        prev_high_3 = high.iloc[-4]
        prev_low_3 = low.iloc[-4]
        prev_volume_3 = volume.iloc[-4]
        
        # Calculate ranges
        current_range = current_high - current_low
        prev_range_3 = prev_high_3 - prev_low_3
        
        # Calculate volume averages
        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        
        # 🚨 **IMPLOSION DETECTION**: Range contracts significantly
        range_contraction = current_range / prev_range_3 if prev_range_3 > 0 else 1
        volume_contraction = current_volume / vol_avg_10
        
        # Bullish Absorption (CE setup)
        if (range_contraction < IMPLOSION_RANGE_RATIO and  # Range contracts
            volume_contraction < ABSORPTION_VOLUME_RATIO and  # Volume dries up
            current_close > (current_high + current_low) / 2 and  # Closes in upper half
            current_close > close.iloc[-2] and  # Higher close than previous
            close.iloc[-2] > close.iloc[-3]):   # Upward momentum building
            return "CE"
        
        # Bearish Absorption (PE setup)
        if (range_contraction < IMPLOSION_RANGE_RATIO and  # Range contracts
            volume_contraction < ABSORPTION_VOLUME_RATIO and  # Volume dries up
            current_close < (current_high + current_low) / 2 and  # Closes in lower half
            current_close < close.iloc[-2] and  # Lower close than previous
            close.iloc[-2] < close.iloc[-3]):   # Downward momentum building
            return "PE"
            
    except Exception as e:
        return None
    return None

# 🚨 **ENHANCED INSTITUTIONAL BLAST DETECTOR** 🚨
def detect_institutional_blast(df):
    """
    Enhanced blast detector with absorption confirmation
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        open_price = ensure_series(df['Open'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 8:
            return None
        
        # Check for absorption first (institutions loading)
        absorption_signal = detect_institutional_absorption(df.iloc[:-1])  # Check previous candles
        
        # Current candle data
        current_open = open_price.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        current_volume = volume.iloc[-1]
        
        # Previous candle data
        prev_open = open_price.iloc[-2]
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        prev_close = close.iloc[-2]
        prev_volume = volume.iloc[-2]
        
        # Calculate moves
        current_body = abs(current_close - current_open)
        prev_body = abs(prev_close - prev_open)
        
        # Volume averages
        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        
        # 🚨 **BULLISH BLAST CANDLE (CE)** 🚨
        # Conditions: Big green candle, high volume, closes near high, follows absorption
        if (current_close > current_open and  # Green candle
            current_body > prev_body * 2.5 and  # Body 2.5x previous
            current_volume > vol_avg_10 * BLAST_VOLUME_THRESHOLD and  # High volume
            (current_close - current_low) > current_body * 2.0 and  # Small lower wick
            current_close > prev_high and  # Breaks previous high
            (current_close - prev_close) / prev_close > BLAST_PRICE_MOVE_PCT and  # 1.5%+ move
            (absorption_signal == "CE" or absorption_signal is None)):  # Follows absorption or neutral
            return "CE"
        
        # 🚨 **BEARISH BLAST CANDLE (PE)** 🚨
        # Conditions: Big red candle, high volume, closes near low, follows absorption
        elif (current_close < current_open and  # Red candle
              current_body > prev_body * 2.5 and  # Body 2.5x previous
              current_volume > vol_avg_10 * BLAST_VOLUME_THRESHOLD and  # High volume
              (current_high - current_close) > current_body * 2.0 and  # Small upper wick
              current_close < prev_low and  # Breaks previous low
              (prev_close - current_close) / prev_close > BLAST_PRICE_MOVE_PCT and  # 1.5%+ move
              (absorption_signal == "PE" or absorption_signal is None)):  # Follows absorption or neutral
            return "PE"
            
    except Exception as e:
        return None
    return None

# 🚨 **ENHANCED VOLUME SPIKE BLAST DETECTION** 🚨
def detect_volume_spike_blast(df):
    """
    Detect sudden volume spikes with price movement AND absorption
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(volume) < 20:
            return None
        
        # Check for absorption first
        absorption_signal = detect_institutional_absorption(df.iloc[:-1])
        
        # Current values
        current_close = close.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_volume = volume.iloc[-1]
        
        # Volume calculations
        vol_avg_20 = volume.rolling(20).mean().iloc[-1]
        vol_ratio = current_volume / vol_avg_20 if vol_avg_20 > 0 else 0
        
        # CHECK: Volume spike (3.5x average) with price confirmation
        if vol_ratio > BLAST_VOLUME_THRESHOLD:
            # Check price movement direction
            prev_close = close.iloc[-2]
            price_change_pct = (current_close - prev_close) / prev_close
            
            # 🚨 **BULLISH VOLUME SPIKE** 🚨
            if (price_change_pct > BLAST_PRICE_MOVE_PCT and
                current_close > current_open and  # Green candle
                current_close > (current_high + current_low) / 2 and  # Closes in upper half
                (absorption_signal == "CE" or absorption_signal is None)):  # Follows absorption
                return "CE"
            
            # 🚨 **BEARISH VOLUME SPIKE** 🚨
            elif (price_change_pct < -BLAST_PRICE_MOVE_PCT and
                  current_close < current_open and  # Red candle
                  current_close < (current_high + current_low) / 2 and  # Closes in lower half
                  (absorption_signal == "PE" or absorption_signal is None)):  # Follows absorption
                return "PE"
    
    except Exception:
        return None
    return None

# 🚨 **ENHANCED SWEEP ORDER DETECTION** 🚨
def detect_sweep_orders(df):
    """
    Detect institutional sweep orders through key levels WITH ABSORPTION
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 12:
            return None
        
        # Check for absorption first
        absorption_signal = detect_institutional_absorption(df.iloc[:-2])  # Check earlier candles
        
        # Find recent liquidity levels (last 8 candles, excluding current)
        recent_highs = high.iloc[-12:-2]
        recent_lows = low.iloc[-12:-2]
        
        liquidity_high = recent_highs.max()
        liquidity_low = recent_lows.min()
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        # Volume average
        vol_avg = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # 🚨 **BEARISH SWEEP DETECTION (PE)** 🚨
        # Price sweeps above liquidity then closes below
        if (current_high > liquidity_high * (1 + SWEEP_DISTANCE_PCT) and
            current_close < liquidity_high * 0.995 and  # Closes well below sweep level
            current_vol > vol_avg * 3.0 and  # High volume on sweep
            (absorption_signal == "PE" or absorption_signal is None)):  # Follows absorption
            return "PE"
        
        # 🚨 **BULLISH SWEEP DETECTION (CE)** 🚨
        # Price sweeps below liquidity then closes above
        elif (current_low < liquidity_low * (1 - SWEEP_DISTANCE_PCT) and
              current_close > liquidity_low * 1.005 and  # Closes well above sweep level
              current_vol > vol_avg * 3.0 and  # High volume on sweep
              (absorption_signal == "CE" or absorption_signal is None)):  # Follows absorption
            return "CE"
    
    except Exception:
        return None
    return None

# 🚨 **INSTITUTIONAL MOMENTUM CONFIRMATION** 🚨
def institutional_momentum_confirmation(index, df, proposed_signal):
    """
    Strict confirmation for institutional moves
    """
    try:
        close = ensure_series(df['Close'])
        volume = ensure_series(df['Volume'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 6:
            return False
        
        # Multi-timeframe confirmation (check 15min chart too)
        df15 = fetch_index_data(index, "15m", "1d")
        if df15 is not None:
            close15 = ensure_series(df15['Close'])
            if len(close15) >= 3:
                if proposed_signal == "CE":
                    if not (close15.iloc[-1] > close15.iloc[-2]):
                        return False
                elif proposed_signal == "PE":
                    if not (close15.iloc[-1] < close15.iloc[-2]):
                        return False
        
        # Volume must be increasing
        vol_avg = volume.rolling(5).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if current_vol < vol_avg * 1.2:  # Volume must be 20% above average
            return False
        
        # Price must show momentum
        if proposed_signal == "CE":
            if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
                return False
            # Candle must have power (small wicks relative to body)
            current_body = abs(close.iloc[-1] - close.iloc[-2])
            upper_wick = high.iloc[-1] - max(close.iloc[-1], close.iloc[-2])
            if upper_wick > current_body * 0.5:  # Upper wick less than 50% of body
                return False
                
        elif proposed_signal == "PE":
            if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
                return False
            # Candle must have power (small wicks relative to body)
            current_body = abs(close.iloc[-1] - close.iloc[-2])
            lower_wick = min(close.iloc[-1], close.iloc[-2]) - low.iloc[-1]
            if lower_wick > current_body * 0.5:  # Lower wick less than 50% of body
                return False
        
        return True
        
    except Exception:
        return False

# --------- SIGNAL DEDUPLICATION AND COOLDOWN CHECK ---------
def can_send_signal(index, strike, option_type):
    current_time = time.time()
    strike_key = f"{index}_{strike}_{option_type}"
    
    if strike_key in active_strikes:
        return False
        
    if index in last_signal_time:
        time_since_last = current_time - last_signal_time[index]
        if time_since_last < signal_cooldown:
            return False
    
    return True

def update_signal_tracking(index, strike, option_type, signal_id):
    global active_strikes, last_signal_time
    
    strike_key = f"{index}_{strike}_{option_type}"
    active_strikes[strike_key] = {
        'signal_id': signal_id,
        'timestamp': time.time(),
        'targets_hit': 0
    }
    
    last_signal_time[index] = time.time()

def update_signal_progress(signal_id, targets_hit):
    for strike_key, data in active_strikes.items():
        if data['signal_id'] == signal_id:
            active_strikes[strike_key]['targets_hit'] = targets_hit
            break

def clear_completed_signal(signal_id):
    global active_strikes
    active_strikes = {k: v for k, v in active_strikes.items() if v['signal_id'] != signal_id}

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

    if vol_latest > vol_avg*2.0 and abs(last_close-prev_close)/prev_close>0.005:
        return "BOTH"
    elif last_close>prev_close and vol_latest>vol_avg*1.5:
        return "CE"
    elif last_close<prev_close and vol_latest>vol_avg*1.5:
        return "PE"
    
    return None

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
            
            if not in_trade and (current_time - last_activity_time) > 1200:
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
                    
                    current_targets_hit = sum(targets_hit)
                    for i, target in enumerate(targets):
                        if price >= target and not targets_hit[i]:
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                            targets_hit[i] = True
                            current_targets_hit = sum(targets_hit)
                            update_signal_progress(signal_id, current_targets_hit)
                    
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
                        clear_completed_signal(signal_id)
                        break
                        
                    if current_targets_hit >= 2:
                        update_signal_progress(signal_id, current_targets_hit)
                    
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

# --------- FIXED: WORKING EOD REPORT SYSTEM ---------
def send_individual_signal_reports():
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
        targets_hit_list = []
        if signal.get('targets_hit', 0) > 0:
            for j in range(signal.get('targets_hit', 0)):
                if j < len(signal.get('targets', [])):
                    targets_hit_list.append(str(signal['targets'][j]))
        
        targets_for_disp = signal.get('targets', [])
        while len(targets_for_disp) < 4:
            targets_for_disp.append('-')
        
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
                
                f"⚡ Fakeout: {'YES' if signal.get('fakeout') else 'NO'}\n"
                f"📈 Index Price at Signal: {signal.get('index_price','?')}\n"
                f"🆔 Signal ID: {signal.get('signal_id','?')}\n"
                f"─────────────────────────────")
        
        send_telegram(msg)
        time.sleep(1)
    
    total_pnl = 0.0
    successful_trades = 0
    for signal in unique_signals:
        pnl_str = signal.get("final_pnl", "0")
        try:
            if isinstance(pnl_str, str) and pnl_str.startswith("+"):
                total_pnl += float(pnl_str[1:])
                successful_trades += 1
            elif isinstance(pnl_str, str) and pnl_str.startswith("-"):
                total_pnl -= float(pnl_str[1:])
        except:
            pass
    
    summary_msg = (f"📈 DAY SUMMARY\n"
                   f"─────────────────────────────\n"
                   f"• Total Signals: {len(unique_signals)}\n"
                   f"• Successful Trades: {successful_trades}\n"
                   f"• Success Rate: {(successful_trades/len(unique_signals))*100:.1f}%\n"
                   f"• Total P&L: ₹{total_pnl:+.2f}\n"
                   f"─────────────────────────────")
    
    send_telegram(summary_msg)
    
    send_telegram("✅ END OF DAY REPORTS COMPLETED! See you tomorrow at 9:15 AM! 🚀")

# 🚨 **INSTITUTIONAL SIGNAL ANALYSIS** 🚨
def analyze_index_signal(index):
    """
    Institutional-grade signal analysis
    Returns only HIGH-CONVICTION setups
    """
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    # 🚨 **STRICT FILTER: NO SIGNALS DURING CHOP** 🚨
    # Check if market is in choppy range (last 5 candles have small range)
    recent_highs = ensure_series(df5['High']).iloc[-5:]
    recent_lows = ensure_series(df5['Low']).iloc[-5:]
    avg_range = (recent_highs.max() - recent_lows.min()) / close5.iloc[-5]
    if avg_range < 0.003:  # Less than 0.3% range = chop, avoid signals
        return None

    # 🚨 **PRIORITY 1: INSTITUTIONAL BLAST DETECTION (Highest Priority)** 🚨
    blast_signal = detect_institutional_blast(df5)
    if blast_signal:
        if institutional_momentum_confirmation(index, df5, blast_signal):
            return blast_signal, df5, False, "institutional_blast"

    # 🚨 **PRIORITY 2: VOLUME SPIKE BLAST** 🚨
    volume_blast = detect_volume_spike_blast(df5)
    if volume_blast:
        if institutional_momentum_confirmation(index, df5, volume_blast):
            return volume_blast, df5, False, "volume_spike_blast"

    # 🚨 **PRIORITY 3: SWEEP ORDER DETECTION** 🚨
    sweep_signal = detect_sweep_orders(df5)
    if sweep_signal:
        if institutional_momentum_confirmation(index, df5, sweep_signal):
            return sweep_signal, df5, False, "sweep_order_detection"

    return None

# 🚨 **INSTITUTIONAL SIGNAL SENDING** 🚨
def send_signal(index, side, df, fakeout, strategy_key):
    global signal_counter, all_generated_signals
    
    signal_detection_price = float(ensure_series(df["Close"]).iloc[-1])
    strike = round_strike(index, signal_detection_price)
    
    if strike is None:
        return
        
    if not can_send_signal(index, strike, side):
        return
        
    symbol = get_option_symbol(index, EXPIRIES[index], strike, side)
    
    if symbol is None:
        return
    
    option_price = fetch_option_price(symbol)
    if not option_price: 
        return
    
    entry = round(option_price)
    
    # 🚨 **INSTITUTIONAL TARGETS** 🚨
    # Bigger targets for institutional blasts
    if side == "CE":
        if strategy_key in ["institutional_blast", "volume_spike_blast", "sweep_order_detection"]:
            base_move = max(80, signal_detection_price * 0.004)  # 80 points or 0.4%
        else:
            base_move = max(60, signal_detection_price * 0.003)  # 60 points or 0.3%
        
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 2.0),
            round(entry + base_move * 3.0),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.7)
        
    else:  # PE
        if strategy_key in ["institutional_blast", "volume_spike_blast", "sweep_order_detection"]:
            base_move = max(80, signal_detection_price * 0.004)  # 80 points or 0.4%
        else:
            base_move = max(60, signal_detection_price * 0.003)  # 60 points or 0.3%
        
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 2.0),
            round(entry + base_move * 3.0),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.7)
    
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
    
    update_signal_tracking(index, strike, side, signal_id)
    
    all_generated_signals.append(signal_data.copy())
    
    # 🚨 **INSTITUTIONAL BLAST ALERT** 🚨
    msg = (f"💥 **INSTITUTIONAL BLAST DETECTED** 💥\n"
           f"📈 {index} {strike} {side}\n"
           f"SYMBOL: {symbol}\n"
           f"ENTRY ABOVE: ₹{entry}\n"
           f"TARGETS: {targets_str}\n"
           f"STOP LOSS: ₹{sl}\n"
           f"STRATEGY: {strategy_name}\n"
           f"SIGNAL ID: {signal_id}\n"
           f"⚠️ INSTITUTIONAL FLOW CONFIRMED - BIG MOVE EXPECTED")
    
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

# 🚨 **INSTITUTIONAL TRADE THREAD** 🚨
def trade_thread(index):
    result = analyze_index_signal(index)
    
    if not result:
        return
        
    side, df, fakeout, strategy_key = result
    
    # Final institutional confirmation
    df5 = fetch_index_data(index, "5m", "2d")
    inst_signal = institutional_flow_signal(index, df5) if df5 is not None else None
    
    if inst_signal and inst_signal != "BOTH" and inst_signal != side:
        return  # Flow doesn't confirm, skip
    
    if df is None: 
        df = df5
        
    send_signal(index, side, df, fakeout, strategy_key)

# 🚨 **INSTITUTIONAL MAIN LOOP** 🚨
def run_algo_parallel():
    if not is_market_open(): 
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Institutional Algorithm stopped")
            STOP_SENT = True
            
        if not EOD_REPORT_SENT:
            time.sleep(15)
            send_telegram("📊 GENERATING INSTITUTIONAL END-OF-DAY REPORT...")
            try:
                send_individual_signal_reports()
            except Exception as e:
                send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                time.sleep(10)
                send_individual_signal_reports()
            EOD_REPORT_SENT = True
            send_telegram("✅ INSTITUTIONAL TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
        return
        
    threads = []
    kept_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in kept_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# 🚨 **INSTITUTIONAL ALGO START** 🚨
STARTED_SENT = False
STOP_SENT = False
MARKET_CLOSED_SENT = False
EOD_REPORT_SENT = False

while True:
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time_ist = ist_now.time()
        current_datetime_ist = ist_now
        
        market_open = is_market_open()
        
        if not market_open:
            if not MARKET_CLOSED_SENT:
                send_telegram("🔴 Market is currently closed. Institutional Algorithm waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_telegram("📊 GENERATING INSTITUTIONAL END-OF-DAY REPORT...")
                time.sleep(10)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ Institutional EOD Report completed! Algorithm will resume tomorrow.")
            
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 **INSTITUTIONAL BLAST ENGINE ACTIVATED** 🚀\n"
                         "✅ Tracking: NIFTY, BANKNIFTY, SENSEX, MIDCPNIFTY\n"
                         "✅ Strategies: BLAST DETECTION ONLY\n"
                         "✅ Filters: Absorption + Implosion-Explosion\n"
                         "✅ Cooldown: 30 minutes between signals\n"
                         "✅ Targets: Institutional-grade (80+ points)\n"
                         "❌ REJECTING: All retail noise signals")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached! Preparing Institutional EOD Report...")
                STOP_SENT = True
                STARTED_SENT = False
            
            if not EOD_REPORT_SENT:
                send_telegram("📊 FINALIZING INSTITUTIONAL TRADES...")
                time.sleep(20)
                try:
                    send_individual_signal_reports()
                except Exception as e:
                    send_telegram(f"⚠️ EOD Report Error, retrying: {str(e)[:100]}")
                    time.sleep(10)
                    send_individual_signal_reports()
                EOD_REPORT_SENT = True
                send_telegram("✅ INSTITUTIONAL TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🎯")
            
            time.sleep(60)
            continue
            
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Institutional Algorithm error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)
