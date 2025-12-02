# PURE INSTITUTIONAL BLAST DETECTOR - NO INDICATORS, ONLY ORDER FLOW
import os
import time
import requests
import pandas as pd
import yfinance as yf
import math
import warnings
import pyotp
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
import numpy as np

warnings.filterwarnings("ignore")

# ---------------- PURE INSTITUTIONAL CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

# 🚨 PURE INSTITUTIONAL BLAST PARAMETERS - NO INDICATORS 🚨
BLAST_BODY_RATIO = 2.5        # Candle body 2.5x previous (BIG MOVE)
BLAST_WICK_RATIO = 0.25       # Very small wicks (25% of body) - STRONG MOVE
BREAKOUT_DISTANCE = 0.008     # 0.8% breakout from previous high/low
SWEEP_DISTANCE = 0.006        # 0.6% sweep through liquidity
REJECTION_WICK_RATIO = 2.0    # Long rejection wick for stop hunts
LIQUIDITY_ZONE_WIDTH = 0.003  # 0.3% zone for liquidity hunts

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "09 DEC 2025",
    "BANKNIFTY": "30 DEC 2025", 
    "SENSEX": "04 DEC 2025",
    "MIDCPNIFTY": "30 DEC 2025"
}

# --------- PURE INSTITUTIONAL STRATEGIES ONLY ---------
STRATEGY_NAMES = {
    "liquidity_sweep_blast": "LIQUIDITY SWEEP BLAST",
    "opening_range_blast": "OPENING RANGE BLAST", 
    "breakout_blast": "BREAKOUT BLAST",
    "rejection_blast": "REJECTION BLAST",
    "liquidity_zone_blast": "LIQUIDITY ZONE BLAST"
}

# --------- ENHANCED TRACKING ---------
all_generated_signals = []
signal_counter = 0
daily_signals = []
active_strikes = {}
last_signal_time = {}
signal_cooldown = 1200

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

# --------- STRIKE ROUNDING ---------
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

# --------- FETCH INDEX DATA ---------
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

# 🚨 **PURE INSTITUTIONAL: DETECT LIQUIDITY POOLS** 🚨
def detect_liquidity_pools(df, lookback=15):
    """
    Find institutional liquidity levels (stop clusters)
    """
    try:
        high = ensure_series(df['High']).dropna()
        low = ensure_series(df['Low']).dropna()
        
        if len(high) < lookback:
            return None, None
            
        # Recent highs and lows where stops accumulate
        recent_high_pool = high.rolling(lookback).max().iloc[-2]
        recent_low_pool = low.rolling(lookback).min().iloc[-2]
        
        return round(recent_high_pool, 0), round(recent_low_pool, 0)
    except Exception:
        return None, None

# 🚨 **PURE INSTITUTIONAL: LIQUIDITY SWEEP BLAST** 🚨
def detect_liquidity_sweep_blast(df):
    """
    Detect institutional sweep orders through liquidity pools
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 10:
            return None
            
        # Find liquidity pools
        liquidity_high, liquidity_low = detect_liquidity_pools(df, 10)
        if liquidity_high is None or liquidity_low is None:
            return None
        
        current_close = close.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        prev_close = close.iloc[-2]
        
        # BEARISH SWEEP BLAST: Price sweeps above liquidity then reverses
        if (current_high > liquidity_high * (1 + SWEEP_DISTANCE) and
            current_close < liquidity_high * 0.995 and
            current_close < prev_close):
            return "PE"
        
        # BULLISH SWEEP BLAST: Price sweeps below liquidity then reverses
        if (current_low < liquidity_low * (1 - SWEEP_DISTANCE) and
            current_close > liquidity_low * 1.005 and
            current_close > prev_close):
            return "CE"
            
    except Exception:
        return None
    return None

# 🚨 **PURE INSTITUTIONAL: OPENING RANGE BLAST** 🚨
def detect_opening_range_blast(df):
    """
    Institutional opening range breakout
    """
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        current_time = ist_now.time()
        
        # Only check during opening hours
        if not (OPENING_START <= current_time <= OPENING_END):
            return None
            
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 5:
            return None
            
        # Opening range (first 30 minutes - 6 candles)
        opening_candles = min(6, len(close))
        opening_high = high.iloc[:opening_candles].max()
        opening_low = low.iloc[:opening_candles].min()
        
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        # BULLISH OPENING BLAST: Breaks above opening range
        if (current_close > opening_high and
            current_close > prev_close and
            (current_close - opening_high) / opening_high > BREAKOUT_DISTANCE):
            return "CE"
        
        # BEARISH OPENING BLAST: Breaks below opening range
        if (current_close < opening_low and
            current_close < prev_close and
            (opening_low - current_close) / opening_low > BREAKOUT_DISTANCE):
            return "PE"
            
    except Exception:
        return None
    return None

# 🚨 **PURE INSTITUTIONAL: BREAKOUT BLAST (BIG CANDLE)** 🚨
def detect_breakout_blast(df):
    """
    Detect BIG single candle breakouts (like in your screenshots)
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        open_price = ensure_series(df['Open'])
        
        if len(close) < 5:
            return None
            
        current_close = close.iloc[-1]
        current_open = open_price.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        
        prev_close = close.iloc[-2]
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        
        # Calculate candle characteristics
        current_body = abs(current_close - current_open)
        prev_body = abs(prev_close - open_price.iloc[-2])
        
        # BULLISH BREAKOUT BLAST: BIG green candle breaking previous high
        if (current_close > current_open and  # Green candle
            current_body > prev_body * BLAST_BODY_RATIO and  # Body 2.5x bigger
            (current_close - current_low) > current_body * BLAST_WICK_RATIO and  # Small lower wick
            current_close > prev_high and  # Breaks previous high
            (current_close - prev_close) / prev_close > BREAKOUT_DISTANCE):  # 0.8%+ move
            return "CE"
        
        # BEARISH BREAKOUT BLAST: BIG red candle breaking previous low
        if (current_close < current_open and  # Red candle
            current_body > prev_body * BLAST_BODY_RATIO and  # Body 2.5x bigger
            (current_high - current_close) > current_body * BLAST_WICK_RATIO and  # Small upper wick
            current_close < prev_low and  # Breaks previous low
            (prev_close - current_close) / prev_close > BREAKOUT_DISTANCE):  # 0.8%+ move
            return "PE"
            
    except Exception:
        return None
    return None

# 🚨 **PURE INSTITUTIONAL: REJECTION BLAST (STOP HUNT)** 🚨
def detect_rejection_blast(df):
    """
    Detect rejection at key levels (stop hunts)
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        open_price = ensure_series(df['Open'])
        
        if len(close) < 5:
            return None
            
        current_close = close.iloc[-1]
        current_open = open_price.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        
        # Find recent highs/lows for rejection
        recent_high = high.iloc[-8:-2].max()
        recent_low = low.iloc[-8:-2].min()
        
        current_body = abs(current_close - current_open)
        candle_range = current_high - current_low
        
        if candle_range == 0:
            return None
            
        upper_wick = current_high - max(current_close, current_open)
        lower_wick = min(current_close, current_open) - current_low
        
        # BEARISH REJECTION: Long upper wick at resistance
        if (upper_wick > current_body * REJECTION_WICK_RATIO and
            current_high > recent_high * (1 + LIQUIDITY_ZONE_WIDTH) and
            current_close < recent_high * 0.998):
            return "PE"
        
        # BULLISH REJECTION: Long lower wick at support
        if (lower_wick > current_body * REJECTION_WICK_RATIO and
            current_low < recent_low * (1 - LIQUIDITY_ZONE_WIDTH) and
            current_close > recent_low * 1.002):
            return "CE"
            
    except Exception:
        return None
    return None

# 🚨 **PURE INSTITUTIONAL: LIQUIDITY ZONE BLAST** 🚨
def detect_liquidity_zone_blast(df):
    """
    Entry at institutional liquidity zones
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        
        if len(close) < 15:
            return None
            
        # Detect liquidity zones
        liquidity_high, liquidity_low = detect_liquidity_pools(df, 15)
        if liquidity_high is None or liquidity_low is None:
            return None
            
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        # Check proximity to liquidity zones
        high_zone_distance = abs(current_close - liquidity_high) / liquidity_high
        low_zone_distance = abs(current_close - liquidity_low) / liquidity_low
        
        # BEARISH: Price at upper liquidity zone and rejecting
        if (high_zone_distance < LIQUIDITY_ZONE_WIDTH and
            current_close < prev_close):
            return "PE"
        
        # BULLISH: Price at lower liquidity zone and bouncing
        if (low_zone_distance < LIQUIDITY_ZONE_WIDTH and
            current_close > prev_close):
            return "CE"
            
    except Exception:
        return None
    return None

# 🚨 **PURE INSTITUTIONAL: SENSEX SPECIFIC BLAST DETECTION** 🚨
def detect_sensex_blast(df):
    """
    Special detection for SENSEX big moves
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        open_price = ensure_series(df['Open'])
        
        if len(close) < 5:
            return None
            
        current_close = close.iloc[-1]
        current_open = open_price.iloc[-1]
        prev_close = close.iloc[-2]
        
        # SENSEX moves are bigger in points
        move_points = abs(current_close - prev_close)
        move_percent = move_points / prev_close
        
        # BIG SENSEX MOVE: 150+ points or 0.5%+
        if move_points > 150 or move_percent > 0.005:
            if current_close > prev_close:
                return "CE"
            else:
                return "PE"
                
    except Exception:
        return None
    return None

# 🚨 **PURE INSTITUTIONAL: MAIN SIGNAL ANALYSIS** 🚨
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 10 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    # 🚨 **PRIORITY 1: BREAKOUT BLAST (BIG SINGLE CANDLE)** 🚨
    breakout_signal = detect_breakout_blast(df5)
    if breakout_signal:
        return breakout_signal, df5, False, "breakout_blast"

    # 🚨 **PRIORITY 2: SENSEX SPECIFIC BLAST (FOR SENSEX ONLY)** 🚨
    if index == "SENSEX":
        sensex_signal = detect_sensex_blast(df5)
        if sensex_signal:
            return sensex_signal, df5, False, "breakout_blast"

    # 🚨 **PRIORITY 3: LIQUIDITY SWEEP BLAST** 🚨
    sweep_signal = detect_liquidity_sweep_blast(df5)
    if sweep_signal:
        return sweep_signal, df5, False, "liquidity_sweep_blast"

    # 🚨 **PRIORITY 4: OPENING RANGE BLAST** 🚨
    try:
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        t = ist_now.time()
        if OPENING_PLAY_ENABLED and (OPENING_START <= t <= OPENING_END):
            opening_signal = detect_opening_range_blast(df5)
            if opening_signal:
                return opening_signal, df5, False, "opening_range_blast"
    except:
        pass

    # 🚨 **PRIORITY 5: REJECTION BLAST** 🚨
    rejection_signal = detect_rejection_blast(df5)
    if rejection_signal:
        return rejection_signal, df5, False, "rejection_blast"

    # 🚨 **PRIORITY 6: LIQUIDITY ZONE BLAST** 🚨
    zone_signal = detect_liquidity_zone_blast(df5)
    if zone_signal:
        return zone_signal, df5, False, "liquidity_zone_blast"

    return None

# --------- SIGNAL DEDUPLICATION ---------
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

def clear_completed_signal(signal_id):
    global active_strikes
    active_strikes = {k: v for k, v in active_strikes.items() if v['signal_id'] != signal_id}

# --------- TRADE MONITORING ---------
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
        in_trade = False
        entry_price_achieved = False
        max_price_reached = entry
        targets_hit = [False] * len(targets)
        signal_id = signal_data.get('signal_id')
        
        while True:
            if should_stop_trading():
                try:
                    final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                except Exception:
                    final_pnl = "0"
                signal_data.update({
                    "entry_status": "NOT_ENTERED" if not entry_price_achieved else "ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price_reached,
                    "final_pnl": final_pnl
                })
                daily_signals.append(signal_data)
                clear_completed_signal(signal_id)
                break
                
            price = fetch_option_price(symbol)
            if price:
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
                    
                    for i, target in enumerate(targets):
                        if price >= target and not targets_hit[i]:
                            send_telegram(f"🎯 {symbol}: Target {i+1} hit at ₹{target}", reply_to=thread_id)
                            targets_hit[i] = True
                    
                    if price <= sl:
                        send_telegram(f"🔗 {symbol}: Stop Loss {sl} hit. Exit trade.", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": sum(targets_hit),
                            "max_price_reached": max_price_reached,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
                        
                    if all(targets_hit):
                        send_telegram(f"🏆 {symbol}: ALL TARGETS HIT!", reply_to=thread_id)
                        try:
                            final_pnl = calculate_pnl(entry, max_price_reached, targets, targets_hit, sl)
                        except Exception:
                            final_pnl = "0"
                        signal_data.update({
                            "targets_hit": len(targets),
                            "max_price_reached": max_price_reached,
                            "final_pnl": final_pnl
                        })
                        daily_signals.append(signal_data)
                        clear_completed_signal(signal_id)
                        break
            
            time.sleep(10)
    
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

# --------- EOD REPORT ---------
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
        targets_for_disp = signal.get('targets', [])
        while len(targets_for_disp) < 4:
            targets_for_disp.append('-')
        
        msg = (f"📊 SIGNAL #{i}\n"
               f"📈 {signal.get('index','?')} {signal.get('strike','?')} {signal.get('option_type','?')}\n"
               f"🕒 Time: {signal.get('timestamp','?')}\n"
               f"🏷️ Strategy: {signal.get('strategy','?')}\n"
               f"💰 ENTRY: ₹{signal.get('entry_price','?')}\n"
               f"🎯 TARGETS: {targets_for_disp[0]} // {targets_for_disp[1]} // {targets_for_disp[2]} // {targets_for_disp[3]}\n"
               f"🛑 SL: ₹{signal.get('sl','?')}\n"
               f"📊 Max Price: ₹{signal.get('max_price_reached', signal.get('entry_price','?'))}\n"
               f"✅ Targets Hit: {signal.get('targets_hit', 0)}/4\n"
               f"💰 P&L: {signal.get('final_pnl', '0')}\n"
               f"─────────────────────────────")
        
        send_telegram(msg)
        time.sleep(1)
    
    send_telegram("✅ END OF DAY REPORTS COMPLETED! See you tomorrow at 9:15 AM! 🚀")

# 🚨 **PURE INSTITUTIONAL: SEND SIGNAL** 🚨
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
    
    # 🚨 **INSTITUTIONAL TARGETS: BIGGER FOR BLASTS** 🚨
    if side == "CE":
        # For blasts, use bigger targets
        if strategy_key in ["breakout_blast", "liquidity_sweep_blast"]:
            base_move = 80  # Bigger move for blasts
        else:
            base_move = 50  # Normal move
        
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 2.0),
            round(entry + base_move * 3.5),
            round(entry + base_move * 5.0)
        ]
        sl = round(entry - base_move * 0.6)
        
    else:
        if strategy_key in ["breakout_blast", "liquidity_sweep_blast"]:
            base_move = 80
        else:
            base_move = 50
        
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 2.0),
            round(entry + base_move * 3.5),
            round(entry + base_move * 5.0)
        ]
        sl = round(entry - base_move * 0.6)
    
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
        "final_pnl": "0"
    }
    
    update_signal_tracking(index, strike, side, signal_id)
    all_generated_signals.append(signal_data.copy())
    
    # 🚨 **INSTITUTIONAL ALERT FORMAT** 🚨
    if side == "CE":
        emoji = "💥" if strategy_key == "breakout_blast" else "🟢"
    else:
        emoji = "💥" if strategy_key == "breakout_blast" else "🔴"
    
    if strategy_key == "breakout_blast":
        msg = (f"{emoji} **INSTITUTIONAL BLAST DETECTED** {emoji}\n"
               f"📈 {index} {strike} {side}\n"
               f"📊 Symbol: {symbol}\n"
               f"💰 Entry Above: ₹{entry}\n"
               f"🎯 Targets: {targets_str}\n"
               f"🛑 Stop Loss: ₹{sl}\n"
               f"🏷️ Strategy: {strategy_name}\n"
               f"🆔 Signal ID: {signal_id}\n"
               f"⚠️ **BIG MOVE EXPECTED - INSTITUTIONAL FLOW**")
    else:
        msg = (f"{emoji} {index} {strike} {side}\n"
               f"📊 Symbol: {symbol}\n"
               f"💰 Above: ₹{entry}\n"
               f"🎯 Targets: {targets_str}\n"
               f"🛑 SL: ₹{sl}\n"
               f"🏷️ Strategy: {strategy_name}\n"
               f"🆔 Signal ID: {signal_id}")
    
    thread_id = send_telegram(msg)
    
    trade_id = f"{symbol}_{int(time.time())}"
    active_trades[trade_id] = {
        "symbol": symbol, 
        "entry": entry, 
        "sl": sl, 
        "targets": targets, 
        "thread": thread_id, 
        "status": "OPEN",
        "signal_data": signal_data
    }
    
    monitor_price_live(symbol, entry, targets, sl, fakeout, thread_id, strategy_name, signal_data)

# --------- TRADE THREAD ---------
def trade_thread(index):
    result = analyze_index_signal(index)
    if result:
        side, df, fakeout, strategy_key = result
        send_signal(index, side, df, fakeout, strategy_key)

# --------- MAIN LOOP ---------
def run_algo_parallel():
    if not is_market_open(): 
        return
        
    if should_stop_trading():
        global STOP_SENT, EOD_REPORT_SENT
        if not STOP_SENT:
            send_telegram("🛑 Market closed at 3:30 PM IST - Algorithm stopped")
            STOP_SENT = True
            
        if not EOD_REPORT_SENT:
            time.sleep(15)
            send_telegram("📊 GENERATING END-OF-DAY REPORT...")
            try:
                send_individual_signal_reports()
            except Exception as e:
                send_telegram(f"⚠️ EOD Report Error: {str(e)[:100]}")
                time.sleep(10)
                send_individual_signal_reports()
            EOD_REPORT_SENT = True
            send_telegram("✅ TRADING DAY COMPLETED! See you tomorrow at 9:15 AM! 🚀")
            
        return
        
    threads = []
    kept_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in kept_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

# --------- START ALGO ---------
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
                send_telegram("🔴 Market is currently closed. Waiting for 9:15 AM...")
                MARKET_CLOSED_SENT = True
                STARTED_SENT = False
                STOP_SENT = False
                EOD_REPORT_SENT = False
            
            if current_time_ist >= dtime(15,30) and current_time_ist <= dtime(16,0) and not EOD_REPORT_SENT:
                send_telegram("📊 GENERATING END-OF-DAY REPORT...")
                time.sleep(10)
                send_individual_signal_reports()
                EOD_REPORT_SENT = True
            
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🚀 **PURE INSTITUTIONAL BLAST ALGO STARTED** 🚀\n"
                         "✅ 4 Indices Running Simultaneously\n"
                         "✅ SENSEX Big Move Detection: ACTIVE\n"
                         "✅ Breakout Blast Detection: ACTIVE\n"
                         "✅ Liquidity Sweep Detection: ACTIVE\n"
                         "✅ Opening Range Blast: ACTIVE\n"
                         "✅ NO INDICATORS - PURE ORDER FLOW\n"
                         "💥 READY FOR INSTITUTIONAL MOVES!")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached! Stopping algorithm...")
                STOP_SENT = True
                STARTED_SENT = False
            
            time.sleep(60)
            continue
            
        run_algo_parallel()
        time.sleep(30)
        
    except Exception as e:
        error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
        send_telegram(error_msg)
        time.sleep(60)
