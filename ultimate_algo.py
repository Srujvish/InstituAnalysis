# ULTIMATE INSTITUTIONAL ALGO - MONTHLY/WEEKLY CONTEXT
import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import pyotp
import math
import numpy as np
from datetime import datetime, time as dtime, timedelta
from SmartApi.smartConnect import SmartConnect
import threading
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------- INSTITUTIONAL CONFIG ----------------
OPENING_PLAY_ENABLED = True
OPENING_START = dtime(9,15)
OPENING_END = dtime(9,45)

# TIME FRAMES FOR INSTITUTIONAL ANALYSIS
TIMEFRAMES = {
    "MONTHLY": "1mo",      # 20+ years data
    "WEEKLY": "1wk",       # 5+ years data  
    "DAILY": "1d",         # 6 months data
    "HOURLY": "60m",       # 1 month data
    "15MIN": "15m",        # 1 week data
    "5MIN": "5m"           # 2 days data
}

# INSTITUTIONAL THRESHOLDS
VWAP_DEV_THRESHOLD = 0.005      # 0.5% from VWAP
VOLUME_SURGE_THRESHOLD = 2.5    # 2.5x average volume
PCR_THRESHOLD = 1.8             # Put/Call Ratio for extremes
OI_CHANGE_THRESHOLD = 0.25      # 25% OI change
DELTA_RATIO_THRESHOLD = 0.6     # 60% Delta ratio

# LIQUIDITY ZONE THRESHOLDS (Dynamic based on timeframe)
ZONE_THRESHOLDS = {
    "MONTHLY": 200,     # 200 points for monthly
    "WEEKLY": 100,      # 100 points for weekly
    "DAILY": 50,        # 50 points for daily
    "HOURLY": 30,       # 30 points for hourly
    "15MIN": 20,        # 20 points for 15min
    "5MIN": 10          # 10 points for 5min
}

# --------- EXPIRIES FOR KEPT INDICES ---------
EXPIRIES = {
    "NIFTY": "09 DEC 2025",
    "BANKNIFTY": "30 DEC 2025", 
    "SENSEX": "04 DEC 2025",
    "MIDCPNIFTY": "30 DEC 2025"
}

# --------- STRATEGIES ---------
STRATEGY_NAMES = {
    "institutional_liquidity_bounce": "INSTITUTIONAL LIQUIDITY BOUNCE",
    "vwap_rejection": "VWAP REJECTION",
    "volume_profile_poc": "VOLUME PROFILE POC",
    "option_chain_alignment": "OPTION CHAIN ALIGNMENT",
    "market_profile_balance": "MARKET PROFILE BALANCE",
    "institutional_orderflow": "INSTITUTIONAL ORDERFLOW"
}

# --------- INSTITUTIONAL TRACKING ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []
active_strikes = {}
last_signal_time = {}
signal_cooldown = 1800  # 30 minutes cooldown

def initialize_strategy_tracking():
    global strategy_performance
    for strategy in STRATEGY_NAMES.values():
        strategy_performance[strategy] = {
            "total": 0, 
            "successful": 0,
            "total_pnl": 0,
            "avg_holding_time": 0
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

# --------- INSTITUTIONAL DATA FETCHING ---------
def fetch_institutional_data(index, timeframe="1d", period="1y"):
    """Fetch institutional-grade data with multiple timeframes"""
    symbol_map = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "SENSEX": "^BSESN",
        "MIDCPNIFTY": "NIFTY_MID_SELECT.NS"
    }
    
    try:
        # Fetch with institutional period
        if timeframe == "1mo":
            period = "20y"
        elif timeframe == "1wk":
            period = "5y"
        elif timeframe == "1d":
            period = "1y"
        elif timeframe == "60m":
            period = "60d"
        elif timeframe == "15m":
            period = "30d"
        else:  # 5m
            period = "7d"
        
        df = yf.download(symbol_map[index], period=period, 
                        interval=timeframe, auto_adjust=True, 
                        progress=False, threads=True)
        
        if df.empty:
            return None
            
        # Calculate institutional indicators
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price channels
        df['Upper_Band'] = df['High'].rolling(20).max()
        df['Lower_Band'] = df['Low'].rolling(20).min()
        
        return df
        
    except Exception as e:
        print(f"Data fetch error: {e}")
        return None

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

# --------- INSTITUTIONAL LIQUIDITY DETECTION ---------
def detect_institutional_liquidity(index):
    """
    Detect liquidity zones across MULTIPLE timeframes
    Institutions look at: Monthly pivots, Weekly highs/lows, Daily VWAP
    """
    liquidity_data = {}
    
    for tf_name, tf in [("MONTHLY", "1mo"), ("WEEKLY", "1wk"), ("DAILY", "1d")]:
        df = fetch_institutional_data(index, tf)
        if df is None or len(df) < 10:
            continue
            
        # Key levels for each timeframe
        monthly_data = {
            "pivot": (df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 3,
            "r1": 2 * liquidity_data[tf_name]["pivot"] - df['Low'].iloc[-1] if tf_name in liquidity_data else None,
            "s1": 2 * liquidity_data[tf_name]["pivot"] - df['High'].iloc[-1] if tf_name in liquidity_data else None,
            "high": float(df['High'].iloc[-1]),
            "low": float(df['Low'].iloc[-1]),
            "vwap": float(df['VWAP'].iloc[-1]) if 'VWAP' in df.columns else None,
            "volume_nodes": detect_volume_nodes(df)
        }
        liquidity_data[tf_name] = monthly_data
    
    # Combine all timeframe levels
    all_levels = []
    for tf in liquidity_data.values():
        all_levels.extend([v for v in tf.values() if isinstance(v, (int, float))])
    
    # Cluster similar levels (institutions cluster orders)
    if len(all_levels) > 5:
        all_levels = np.array(all_levels).reshape(-1, 1)
        clustering = AgglomerativeClustering(n_clusters=min(10, len(all_levels)//2))
        clusters = clustering.fit_predict(all_levels)
        
        # Get cluster centers (institutional order clusters)
        clustered_levels = []
        for i in range(clustering.n_clusters_):
            cluster_points = all_levels[clusters == i]
            if len(cluster_points) > 0:
                clustered_levels.append(float(np.mean(cluster_points)))
        
        return sorted(clustered_levels)
    
    return sorted(list(set(all_levels)))

def detect_volume_nodes(df, num_bins=20):
    """Detect high volume nodes (Volume Profile)"""
    try:
        # Create price bins
        price_range = df['High'].max() - df['Low'].min()
        bin_size = price_range / num_bins
        bins = np.arange(df['Low'].min(), df['High'].max() + bin_size, bin_size)
        
        # Calculate volume at each price level
        volume_at_price = []
        for i in range(len(bins)-1):
            mask = (df['Close'] >= bins[i]) & (df['Close'] < bins[i+1])
            volume_in_bin = df['Volume'][mask].sum()
            price_level = (bins[i] + bins[i+1]) / 2
            volume_at_price.append((price_level, volume_in_bin))
        
        # Find high volume nodes (top 30%)
        volume_at_price.sort(key=lambda x: x[1], reverse=True)
        top_nodes = volume_at_price[:max(3, len(volume_at_price)//3)]
        
        return [node[0] for node in top_nodes]
        
    except:
        return []

# --------- INSTITUTIONAL ORDER FLOW ANALYSIS ---------
def analyze_option_chain(index):
    """Analyze option chain for institutional activity"""
    try:
        url="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df=pd.DataFrame(requests.get(url,timeout=10).json())
        df=df[df['exch_seg'].str.upper().isin(["NFO", "BFO"])]
        df['symbol']=df['symbol'].str.upper()
        
        # Filter for current index and expiry
        current_expiry = EXPIRIES.get(index)
        if not current_expiry:
            return None
            
        expiry_pattern = datetime.strptime(current_expiry, "%d %b %Y").strftime("%d%b%y").upper()
        df_index = df[df['symbol'].str.contains(f"{index}{expiry_pattern}")]
        
        if df_index.empty:
            return None
        
        # Calculate OI and change
        df_index['oi'] = pd.to_numeric(df_index['oi'], errors='coerce').fillna(0)
        df_index['prev_oi'] = df_index['oi'].shift(1).fillna(0)
        df_index['oi_change'] = df_index['oi'] - df_index['prev_oi']
        
        # Separate CE and PE
        ce_data = df_index[df_index['symbol'].str.endswith("CE")]
        pe_data = df_index[df_index['symbol'].str.endswith("PE")]
        
        # Find max OI strikes
        max_ce_oi_strike = ce_data.loc[ce_data['oi'].idxmax()]['symbol'] if not ce_data.empty else None
        max_pe_oi_strike = pe_data.loc[pe_data['oi'].idxmax()]['symbol'] if not pe_data.empty else None
        
        # Calculate PCR
        total_ce_oi = ce_data['oi'].sum()
        total_pe_oi = pe_data['oi'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        # Find max OI change
        max_ce_oi_change = ce_data.loc[ce_data['oi_change'].idxmax()] if not ce_data.empty else None
        max_pe_oi_change = pe_data.loc[pe_data['oi_change'].idxmax()] if not pe_data.empty else None
        
        return {
            'pcr': pcr,
            'max_ce_strike': max_ce_oi_strike,
            'max_pe_strike': max_pe_oi_strike,
            'max_ce_change': max_ce_oi_change,
            'max_pe_change': max_pe_oi_change,
            'total_ce_oi': total_ce_oi,
            'total_pe_oi': total_pe_oi,
            'ce_oi_change': ce_data['oi_change'].sum(),
            'pe_oi_change': pe_data['oi_change'].sum()
        }
        
    except Exception as e:
        print(f"Option chain error: {e}")
        return None

# --------- INSTITUTIONAL ENTRY SIGNAL GENERATION ---------
def generate_institutional_signal(index):
    """
    Institutional signal generation with:
    1. Multi-timeframe analysis
    2. Volume profile alignment
    3. Option chain analysis
    4. Order flow confirmation
    """
    # Get data across timeframes
    df_daily = fetch_institutional_data(index, "1d")
    df_hourly = fetch_institutional_data(index, "60m")
    df_15min = fetch_institutional_data(index, "15m")
    df_5min = fetch_institutional_data(index, "5m")
    
    if any(df is None for df in [df_daily, df_hourly, df_15min, df_5min]):
        return None
    
    # Current price
    current_price = float(df_5min['Close'].iloc[-1])
    
    # 1. Detect institutional liquidity zones
    liquidity_levels = detect_institutional_liquidity(index)
    
    # 2. Analyze option chain
    option_data = analyze_option_chain(index)
    
    # 3. Check multi-timeframe alignment
    daily_trend = "BULLISH" if df_daily['Close'].iloc[-1] > df_daily['Open'].iloc[-1] else "BEARISH"
    hourly_trend = "BULLISH" if df_hourly['Close'].iloc[-1] > df_hourly['Open'].iloc[-1] else "BEARISH"
    
    # 4. Check volume profile
    volume_nodes = detect_volume_nodes(df_daily)
    
    # 5. VWAP analysis
    vwap_daily = df_daily['VWAP'].iloc[-1]
    vwap_hourly = df_hourly['VWAP'].iloc[-1] if 'VWAP' in df_hourly.columns else vwap_daily
    
    # Signal Generation Logic
    signals = []
    
    # A. LIQUIDITY BOUNCE SIGNAL
    for level in liquidity_levels:
        distance_pct = abs(current_price - level) / level
        
        # Check for bounce from support (CE signal)
        if current_price > level and distance_pct < 0.002:  # Within 0.2%
            # Volume confirmation
            if df_5min['Volume_Ratio'].iloc[-1] > VOLUME_SURGE_THRESHOLD:
                # Option chain confirmation (PCR high for bounce)
                if option_data and option_data['pcr'] > PCR_THRESHOLD:
                    # Multi-timeframe alignment
                    if daily_trend == "BULLISH" or hourly_trend == "BULLISH":
                        signals.append({
                            'type': 'CE',
                            'level': level,
                            'strategy': 'institutional_liquidity_bounce',
                            'confidence': 0.8,
                            'reason': f"Bounce from institutional level {level}"
                        })
        
        # Check for rejection from resistance (PE signal)
        elif current_price < level and distance_pct < 0.002:
            # Volume confirmation
            if df_5min['Volume_Ratio'].iloc[-1] > VOLUME_SURGE_THRESHOLD:
                # Option chain confirmation (PCR low for rejection)
                if option_data and option_data['pcr'] < 1/PCR_THRESHOLD:
                    # Multi-timeframe alignment
                    if daily_trend == "BEARISH" or hourly_trend == "BEARISH":
                        signals.append({
                            'type': 'PE',
                            'level': level,
                            'strategy': 'institutional_liquidity_bounce',
                            'confidence': 0.8,
                            'reason': f"Rejection from institutional level {level}"
                        })
    
    # B. VWAP REJECTION SIGNAL
    vwap_distance_pct = abs(current_price - vwap_hourly) / vwap_hourly
    if vwap_distance_pct < VWAP_DEV_THRESHOLD:
        # Price at VWAP with volume surge
        if df_5min['Volume_Ratio'].iloc[-1] > VOLUME_SURGE_THRESHOLD:
            # Determine direction based on trend
            if daily_trend == "BULLISH" and current_price > vwap_hourly:
                signals.append({
                    'type': 'CE',
                    'level': vwap_hourly,
                    'strategy': 'vwap_rejection',
                    'confidence': 0.75,
                    'reason': f"Bounce from VWAP {vwap_hourly}"
                })
            elif daily_trend == "BEARISH" and current_price < vwap_hourly:
                signals.append({
                    'type': 'PE',
                    'level': vwap_hourly,
                    'strategy': 'vwap_rejection',
                    'confidence': 0.75,
                    'reason': f"Rejection from VWAP {vwap_hourly}"
                })
    
    # C. VOLUME PROFILE POC SIGNAL
    for poc in volume_nodes:
        poc_distance_pct = abs(current_price - poc) / poc
        if poc_distance_pct < 0.0015:  # Within 0.15% of POC
            # High volume at POC
            if df_5min['Volume'].iloc[-1] > df_5min['Volume_MA'].iloc[-1] * 2:
                signals.append({
                    'type': 'CE' if current_price > poc else 'PE',
                    'level': poc,
                    'strategy': 'volume_profile_poc',
                    'confidence': 0.7,
                    'reason': f"At Volume POC {poc}"
                })
    
    # Return highest confidence signal
    if signals:
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals[0]
    
    return None

# --------- STRIKE SELECTION ---------
def select_institutional_strike(index, signal_type, current_price):
    """Institutional strike selection based on delta and liquidity"""
    try:
        # Get option chain data
        option_data = analyze_option_chain(index)
        if not option_data:
            # Fallback: Round to nearest strike
            if index == "NIFTY":
                strike_interval = 50
            elif index == "BANKNIFTY":
                strike_interval = 100
            elif index == "SENSEX":
                strike_interval = 100
            elif index == "MIDCPNIFTY":
                strike_interval = 25
            else:
                strike_interval = 50
            
            if signal_type == "CE":
                strike = math.ceil(current_price / strike_interval) * strike_interval
            else:
                strike = math.floor(current_price / strike_interval) * strike_interval
            
            return strike
        
        # Institutional strike selection logic
        if signal_type == "CE":
            # For CE: At-the-money or slightly in-the-money
            strike = round(current_price / 50) * 50
            # Adjust based on PCR
            if option_data['pcr'] > 1.5:
                # High PCR: Buy slightly OTM (more aggressive)
                strike += 50
        else:
            # For PE: At-the-money or slightly in-the-money
            strike = round(current_price / 50) * 50
            # Adjust based on PCR
            if option_data['pcr'] < 0.67:
                # Low PCR: Buy slightly OTM (more aggressive)
                strike -= 50
        
        return strike
        
    except:
        # Simple fallback
        if index == "NIFTY":
            strike_interval = 50
        else:
            strike_interval = 100
        
        if signal_type == "CE":
            return math.ceil(current_price / strike_interval) * strike_interval
        else:
            return math.floor(current_price / strike_interval) * strike_interval

# --------- OPTION SYMBOL GENERATION ---------
def get_option_symbol(index, strike, opttype):
    try:
        expiry_str = EXPIRIES.get(index)
        if not expiry_str:
            return None
            
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
        
        return symbol
    except:
        return None

# --------- SIGNAL DEDUPLICATION ---------
def can_send_signal(index, strike, option_type):
    current_time = time.time()
    strike_key = f"{index}_{strike}_{option_type}"
    
    # Check if same strike is active
    if strike_key in active_strikes:
        return False
        
    # Check index cooldown
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
        'index': index,
        'strike': strike,
        'type': option_type
    }
    
    last_signal_time[index] = time.time()

# --------- INSTITUTIONAL TARGET CALCULATION ---------
def calculate_institutional_targets(entry, signal_type, volatility, index):
    """Calculate targets based on institutional risk parameters"""
    
    # Base move based on index volatility
    if index == "NIFTY":
        base_move = 40 * (volatility + 1)
    elif index == "BANKNIFTY":
        base_move = 80 * (volatility + 1)
    elif index == "SENSEX":
        base_move = 100 * (volatility + 1)
    else:
        base_move = 30 * (volatility + 1)
    
    if signal_type == "CE":
        targets = [
            round(entry + base_move * 1.0),
            round(entry + base_move * 1.8),
            round(entry + base_move * 2.8),
            round(entry + base_move * 4.0)
        ]
        sl = round(entry - base_move * 0.8)
    else:  # PE
        targets = [
            round(entry - base_move * 1.0),
            round(entry - base_move * 1.8),
            round(entry - base_move * 2.8),
            round(entry - base_move * 4.0)
        ]
        sl = round(entry + base_move * 0.8)
    
    return targets, sl

# --------- INSTITUTIONAL SIGNAL SENDING ---------
def send_institutional_signal(index, signal_data):
    global signal_counter, all_generated_signals
    
    signal_type = signal_data['type']
    current_price = float(fetch_institutional_data(index, "5m")['Close'].iloc[-1])
    
    # Select strike
    strike = select_institutional_strike(index, signal_type, current_price)
    
    # Check deduplication
    if not can_send_signal(index, strike, signal_type):
        return
    
    # Generate option symbol
    symbol = get_option_symbol(index, strike, signal_type)
    if not symbol:
        return
    
    # Get option price
    option_price = fetch_option_price(symbol)
    if not option_price:
        return
    
    entry = round(option_price)
    
    # Calculate volatility (simplified)
    df_5min = fetch_institutional_data(index, "5m")
    if df_5min is None or len(df_5min) < 10:
        volatility = 1.0
    else:
        returns = df_5min['Close'].pct_change().dropna()
        volatility = returns.std() * math.sqrt(252)  # Annualized
    
    # Calculate targets
    targets, sl = calculate_institutional_targets(entry, signal_type, volatility, index)
    
    # Generate signal ID
    signal_id = f"INST{signal_counter:04d}"
    signal_counter += 1
    
    # Prepare signal data
    signal_info = {
        "signal_id": signal_id,
        "timestamp": (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M:%S"),
        "index": index,
        "strike": strike,
        "option_type": signal_type,
        "strategy": STRATEGY_NAMES.get(signal_data['strategy'], signal_data['strategy'].upper()),
        "entry_price": entry,
        "targets": targets,
        "sl": sl,
        "level": signal_data.get('level', current_price),
        "confidence": signal_data.get('confidence', 0.5),
        "reason": signal_data.get('reason', ''),
        "index_price": current_price,
        "entry_status": "PENDING",
        "targets_hit": 0,
        "max_price_reached": entry,
        "final_pnl": "0"
    }
    
    # Update tracking
    update_signal_tracking(index, strike, signal_type, signal_id)
    all_generated_signals.append(signal_info.copy())
    
    # Format message
    targets_str = "//".join(str(t) for t in targets)
    msg = (f"🏛️ INSTITUTIONAL SIGNAL - {signal_info['strategy']}\n"
           f"📈 {index} {strike} {signal_type}\n"
           f"📊 Symbol: {symbol}\n"
           f"💰 Entry: ₹{entry}\n"
           f"🎯 Targets: {targets_str}\n"
           f"🛑 SL: ₹{sl}\n"
           f"📊 Level: {signal_data.get('level', 'N/A')}\n"
           f"⚡ Confidence: {signal_data.get('confidence', 0)*100:.0f}%\n"
           f"📝 Reason: {signal_data.get('reason', '')}\n"
           f"🆔 Signal ID: {signal_id}")
    
    # Send to Telegram
    thread_id = send_telegram(msg)
    
    # Start monitoring
    monitor_institutional_trade(symbol, entry, targets, sl, thread_id, signal_info)
    
    return signal_id

# --------- INSTITUTIONAL TRADE MONITORING ---------
def monitor_institutional_trade(symbol, entry, targets, sl, thread_id, signal_info):
    def monitoring_thread():
        global daily_signals
        
        last_price = entry
        max_price = entry
        min_price = entry
        targets_hit = [False] * len(targets)
        entry_triggered = False
        signal_id = signal_info['signal_id']
        
        while True:
            # Check market hours
            if not is_market_open() or should_stop_trading():
                # Finalize trade
                signal_info.update({
                    "entry_status": "ENTERED" if entry_triggered else "NOT_ENTERED",
                    "targets_hit": sum(targets_hit),
                    "max_price_reached": max_price,
                    "final_pnl": calculate_pnl(entry, max_price, targets, targets_hit, sl)
                })
                daily_signals.append(signal_info)
                break
            
            # Fetch current price
            price = fetch_option_price(symbol)
            if not price:
                time.sleep(10)
                continue
            
            # Update extremes
            if price > max_price:
                max_price = price
                if entry_triggered:
                    send_telegram(f"📈 {symbol} making new high: ₹{price}", reply_to=thread_id)
            
            if price < min_price:
                min_price = price
            
            # Check entry trigger
            if not entry_triggered:
                if (signal_info['option_type'] == "CE" and price >= entry) or \
                   (signal_info['option_type'] == "PE" and price <= entry):
                    send_telegram(f"✅ ENTRY TRIGGERED at ₹{price}", reply_to=thread_id)
                    entry_triggered = True
                    signal_info["entry_status"] = "ENTERED"
            
            # Check targets
            if entry_triggered:
                for i, target in enumerate(targets):
                    if not targets_hit[i]:
                        if (signal_info['option_type'] == "CE" and price >= target) or \
                           (signal_info['option_type'] == "PE" and price <= target):
                            send_telegram(f"🎯 Target {i+1} hit at ₹{target}", reply_to=thread_id)
                            targets_hit[i] = True
                            signal_info["targets_hit"] = sum(targets_hit)
                
                # Check SL
                if (signal_info['option_type'] == "CE" and price <= sl) or \
                   (signal_info['option_type'] == "PE" and price >= sl):
                    send_telegram(f"🛑 SL hit at ₹{sl}. Trade closed.", reply_to=thread_id)
                    break
            
            # Check if all targets hit
            if all(targets_hit):
                send_telegram(f"🏆 ALL TARGETS HIT! Trade completed successfully!", reply_to=thread_id)
                break
            
            time.sleep(10)
        
        # Final update
        signal_info.update({
            "targets_hit": sum(targets_hit),
            "max_price_reached": max_price,
            "final_pnl": calculate_pnl(entry, max_price, targets, targets_hit, sl)
        })
    
    # Start monitoring thread
    thread = threading.Thread(target=monitoring_thread)
    thread.daemon = True
    thread.start()

def calculate_pnl(entry, max_price, targets, targets_hit, sl):
    try:
        # Simple PnL calculation
        if isinstance(targets_hit, list) and any(targets_hit):
            hit_targets = [t for t, hit in zip(targets, targets_hit) if hit]
            exit_price = hit_targets[-1] if hit_targets else max_price
        else:
            exit_price = max_price
        
        pnl = exit_price - entry
        return f"+{pnl:.2f}" if pnl > 0 else f"{pnl:.2f}"
    except:
        return "0.00"

# --------- EOD REPORT ---------
def send_institutional_eod_report():
    global daily_signals, all_generated_signals
    
    all_signals = daily_signals + all_generated_signals
    
    if not all_signals:
        send_telegram("📊 END OF DAY REPORT\nNo institutional signals generated today.")
        return
    
    # Summary statistics
    total_signals = len(all_signals)
    entered_signals = [s for s in all_signals if s.get('entry_status') == 'ENTERED']
    successful_signals = [s for s in entered_signals if float(s.get('final_pnl', '0').replace('+', '')) > 0]
    
    # Send summary
    summary_msg = (f"🏛️ INSTITUTIONAL EOD REPORT\n"
                   f"══════════════════════════\n"
                   f"📅 Date: {(datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime('%d-%b-%Y')}\n"
                   f"📊 Total Signals: {total_signals}\n"
                   f"✅ Entered Trades: {len(entered_signals)}\n"
                   f"🏆 Successful Trades: {len(successful_signals)}\n"
                   f"📈 Success Rate: {len(successful_signals)/len(entered_signals)*100:.1f}%"
                   f" if {len(entered_signals)}>0 else 0.0}%\n"
                   f"══════════════════════════")
    
    send_telegram(summary_msg)
    
    # Send detailed reports
    for i, signal in enumerate(all_signals, 1):
        strategy = signal.get('strategy', 'UNKNOWN')
        pnl = signal.get('final_pnl', '0')
        status = signal.get('entry_status', 'PENDING')
        
        detail_msg = (f"#{i} {signal.get('index')} {signal.get('strike')} {signal.get('option_type')}\n"
                     f"Strategy: {strategy}\n"
                     f"Status: {status}\n"
                     f"PNL: {pnl}\n"
                     f"────────────────────")
        
        send_telegram(detail_msg)
        time.sleep(0.5)

# --------- MAIN ALGO LOOP ---------
def run_institutional_algo():
    """Main institutional algo loop"""
    if not is_market_open():
        return
    
    kept_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in kept_indices:
        try:
            # Generate institutional signal
            signal_data = generate_institutional_signal(index)
            
            if signal_data and signal_data.get('confidence', 0) > 0.7:
                # Send signal
                send_institutional_signal(index, signal_data)
                
                # Small delay between indices
                time.sleep(2)
                
        except Exception as e:
            print(f"Error processing {index}: {e}")
            continue

# --------- MAIN EXECUTION ---------
def main():
    global STARTED_SENT, STOP_SENT, MARKET_CLOSED_SENT, EOD_REPORT_SENT
    
    send_telegram("🏛️ INSTITUTIONAL ALGO STARTED\n"
                 "✅ Multi-timeframe analysis (Monthly/Weekly/Daily)\n"
                 "✅ Volume Profile & Liquidity Zones\n"
                 "✅ Option Chain Analysis\n"
                 "✅ Institutional Order Flow\n"
                 "✅ 30-minute signal cooldown")
    
    while True:
        try:
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            current_time_ist = ist_now.time()
            
            # Check market hours
            if not is_market_open():
                if not MARKET_CLOSED_SENT:
                    send_telegram("🔴 Market Closed. Algo paused.")
                    MARKET_CLOSED_SENT = True
                    STARTED_SENT = False
                
                # Generate EOD report at 3:45 PM
                if dtime(15, 35) <= current_time_ist <= dtime(15, 45) and not EOD_REPORT_SENT:
                    send_telegram("📊 Generating Institutional EOD Report...")
                    send_institutional_eod_report()
                    EOD_REPORT_SENT = True
                
                time.sleep(60)
                continue
            
            # Market is open
            if not STARTED_SENT:
                send_telegram("✅ Market Open. Institutional Algo running...")
                STARTED_SENT = True
                MARKET_CLOSED_SENT = False
                EOD_REPORT_SENT = False
            
            # Check stop time
            if should_stop_trading():
                if not STOP_SENT:
                    send_telegram("🛑 Approaching market close. Finalizing trades...")
                    STOP_SENT = True
                
                # Wait for EOD report
                time.sleep(30)
                continue
            
            # Run institutional algo
            run_institutional_algo()
            
            # Sleep between scans (institutions don't scalp every second)
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            error_msg = f"⚠️ Main loop error: {str(e)[:100]}"
            send_telegram(error_msg)
            time.sleep(120)

if __name__ == "__main__":
    main()
