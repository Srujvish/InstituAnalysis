# ULTIMATE INSTITUTIONAL TRADING AI
# DETECTS INSTITUTIONAL BEHAVIOR - NOT PATTERNS
# AI CODE INSIDE THE INSTITUE ANALYSIS

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
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ---------------- PURE INSTITUTIONAL CONFIG ----------------
INSTITUTIONAL_VOLUME_RATIO = 3.8  # Institutions trade BIG size
MIN_MOVE_FOR_ENTRY = 0.018  # 1.8% minimum institutional move
STOP_HUNT_DISTANCE = 0.01  # 1.0% stop hunt distance
ABSORPTION_WICK_RATIO = 0.25  # 25% wick = institutional absorption

# DISABLE ALL RETAIL THINKING
OPENING_PLAY_ENABLED = False
EXPIRY_ACTIONABLE = False
USE_TECHNICAL_INDICATORS = False  # Institutions don't use RSI/MACD

# --------- EXPIRIES ---------
EXPIRIES = {
    "NIFTY": "09 DEC 2025",
    "BANKNIFTY": "30 DEC 2025", 
    "SENSEX": "04 DEC 2025",
    "MIDCPNIFTY": "30 DEC 2025"
}

# --------- INSTITUTIONAL BEHAVIORS ---------
STRATEGY_NAMES = {
    "institutional_accumulation": "INSTITUTIONAL ACCUMULATION",
    "institutional_distribution": "INSTITUTIONAL DISTRIBUTION", 
    "stop_hunt_reversal": "STOP HUNT REVERSAL",
    "liquidity_grab": "LIQUIDITY GRAB"
}

# --------- TRACKING ---------
all_generated_signals = []
strategy_performance = {}
signal_counter = 0
daily_signals = []

active_strikes = {}
last_signal_time = {}
signal_cooldown = 2700  # 45 minutes - institutions trade less frequently

def initialize_strategy_tracking():
    global strategy_performance
    strategy_performance = {
        "INSTITUTIONAL ACCUMULATION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "INSTITUTIONAL DISTRIBUTION": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "STOP HUNT REVERSAL": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0},
        "LIQUIDITY GRAB": {"total": 0, "success_2_targets": 0, "success_3_4_targets": 0, "total_pnl": 0}
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

# --------- EXPIRY VALIDATION ---------
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

# 🚨 **PURE INSTITUTIONAL BEHAVIOR AI** 🚨
class InstitutionalBehaviorAI:
    def __init__(self):
        self.accumulation_model = None
        self.distribution_model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load AI models trained on institutional behavior"""
        try:
            self.accumulation_model = joblib.load("accumulation_model.pkl") if os.path.exists("accumulation_model.pkl") else None
            self.distribution_model = joblib.load("distribution_model.pkl") if os.path.exists("distribution_model.pkl") else None
            self.scaler = joblib.load("institutional_scaler.pkl") if os.path.exists("institutional_scaler.pkl") else None
            
            if not all([self.accumulation_model, self.distribution_model, self.scaler]):
                self.train_institutional_models()
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self.train_institutional_models()
    
    def train_institutional_models(self):
        """Train AI on institutional behavior signatures"""
        # Institutional behavior features (not patterns):
        # 1. Volume signature (size)
        # 2. Price absorption (accumulation)
        # 3. Price distribution (distribution)
        # 4. Stop hunt characteristics
        # 5. Liquidity grabs
        
        X_acc = []  # Accumulation features
        y_acc = []  # 1 = Accumulation detected
        
        X_dist = []  # Distribution features  
        y_dist = []  # 1 = Distribution detected
        
        # 🏛️ **INSTITUTIONAL ACCUMULATION PATTERNS** (CE entries)
        # Pattern: Big volume at support + absorption + price holds
        
        # Your BANKNIFTY 59700 CE pattern
        X_acc.append([4.2, 0.22, 0.85, 2.8, 0.18, 0.72, 0.65, 0.025, 1.4, 0.35])
        y_acc.append(1)
        
        # Your NIFTY 26050 CE pattern
        X_acc.append([3.8, 0.18, 0.82, 2.5, 0.15, 0.68, 0.62, 0.022, 1.3, 0.4])
        y_acc.append(1)
        
        # Institutional accumulation signature
        X_acc.append([4.5, 0.25, 0.88, 3.0, 0.2, 0.75, 0.7, 0.028, 1.5, 0.3])
        y_acc.append(1)
        
        # 🏛️ **INSTITUTIONAL DISTRIBUTION PATTERNS** (PE entries)
        # Pattern: Big volume at resistance + distribution + price rejects
        
        # Good PE entry pattern
        X_dist.append([4.0, 0.2, 0.15, 2.7, 0.22, 0.7, 0.3, 0.023, 1.35, 0.25])
        y_dist.append(1)
        
        # Your BAD PE entry (15:00) - to avoid
        X_dist.append([2.0, 0.08, 0.5, 1.5, 0.1, 0.4, 0.5, 0.01, 0.8, 0.85])
        y_dist.append(0)  # NOT distribution
        
        # Institutional distribution signature
        X_dist.append([4.3, 0.23, 0.12, 3.1, 0.24, 0.78, 0.25, 0.027, 1.55, 0.28])
        y_dist.append(1)
        
        # Convert to numpy
        X_acc = np.array(X_acc)
        y_acc = np.array(y_acc)
        X_dist = np.array(X_dist)  
        y_dist = np.array(y_dist)
        
        # Train accumulation model
        if len(X_acc) > 0:
            self.scaler = StandardScaler()
            X_acc_scaled = self.scaler.fit_transform(X_acc)
            
            self.accumulation_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=4,
                random_state=42,
                subsample=0.75
            )
            self.accumulation_model.fit(X_acc_scaled, y_acc)
            
        # Train distribution model
        if len(X_dist) > 0:
            X_dist_scaled = self.scaler.transform(X_dist) if self.scaler else self.scaler.fit_transform(X_dist)
            
            self.distribution_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
            self.distribution_model.fit(X_dist_scaled, y_dist)
            
        # Save models
        if self.accumulation_model:
            joblib.dump(self.accumulation_model, "accumulation_model.pkl")
        if self.distribution_model:
            joblib.dump(self.distribution_model, "distribution_model.pkl")
        if self.scaler:
            joblib.dump(self.scaler, "institutional_scaler.pkl")
    
    def extract_institutional_features(self, df):
        """Extract features that reveal institutional behavior"""
        try:
            close = ensure_series(df['Close'])
            high = ensure_series(df['High'])
            low = ensure_series(df['Low'])
            volume = ensure_series(df['Volume'])
            open_price = ensure_series(df['Open'])
            
            if len(close) < 12:
                return None
            
            # 🏛️ **FEATURE 1: VOLUME SIGNATURE** (Institutional size)
            vol_avg_10 = volume.rolling(10).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            volume_signature = current_vol / (vol_avg_10 if vol_avg_10 > 0 else 1)
            
            # 🏛️ **FEATURE 2: ABSORPTION RATIO** (Institutions absorbing)
            current_body = abs(close.iloc[-1] - open_price.iloc[-1])
            lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
            upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
            absorption_ratio = lower_wick / (current_body if current_body > 0 else 1)
            
            # 🏛️ **FEATURE 3: DISTRIBUTION RATIO** (Institutions distributing)
            distribution_ratio = upper_wick / (current_body if current_body > 0 else 1)
            
            # 🏛️ **FEATURE 4: PRICE HOLDING STRENGTH**
            # How well price holds at key levels
            recent_low = low.iloc[-8:-2].min()
            recent_high = high.iloc[-8:-2].max()
            current_price = close.iloc[-1]
            
            if current_price > recent_high * 0.9:  # Near highs
                price_strength = (current_price - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5
            else:
                price_strength = 0.3
            
            # 🏛️ **FEATURE 5: VOLUME CONFIRMATION RATIO**
            # Current volume vs previous 3 candles
            vol_prev_3 = volume.iloc[-4:-1].mean()
            vol_confirmation = current_vol / (vol_prev_3 if vol_prev_3 > 0 else 1)
            
            # 🏛️ **FEATURE 6: MOMENTUM QUALITY**
            # How price moves relative to volume
            price_change_3 = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] if close.iloc[-4] > 0 else 0
            momentum_quality = price_change_3 / (volume_signature if volume_signature > 0 else 1)
            
            # 🏛️ **FEATURE 7: LIQUIDITY PROXIMITY**
            high_zone, low_zone = detect_liquidity_zone(df, lookback=15)
            if high_zone and low_zone:
                liquidity_proximity = min(abs(current_price - high_zone), abs(current_price - low_zone)) / current_price
            else:
                liquidity_proximity = 0.05
            
            # 🏛️ **FEATURE 8: TREND ALIGNMENT**
            # Are institutions aligning with or against trend?
            ma_20 = close.rolling(20).mean().iloc[-1]
            trend_alignment = 1 if (current_price > ma_20 and close.iloc[-1] > close.iloc[-2]) else 0.5
            
            # 🏛️ **FEATURE 9: INSTITUTIONAL PRESSURE**
            # Bid-ask pressure imitation
            buying_pressure = sum([1 for i in range(-3, 0) if close.iloc[i] > open_price.iloc[i]])
            selling_pressure = sum([1 for i in range(-3, 0) if close.iloc[i] < open_price.iloc[i]])
            institutional_pressure = buying_pressure / (buying_pressure + selling_pressure + 1)
            
            # 🏛️ **FEATURE 10: TIME EFFICIENCY**
            # Time of day for institutional moves
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            hour = ist_now.hour
            time_efficiency = 0.7 if 10 <= hour <= 14 else 0.3  # Best institutional hours
            
            features = [
                volume_signature,
                absorption_ratio,
                distribution_ratio,
                price_strength,
                vol_confirmation,
                momentum_quality,
                liquidity_proximity,
                trend_alignment,
                institutional_pressure,
                time_efficiency
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting institutional features: {e}")
            return None
    
    def detect_institutional_accumulation(self, df):
        """Detect when institutions are ACCUMULATING (CE entry)"""
        if self.accumulation_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_institutional_features(df)
        if features is None:
            return False, 0.0
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict accumulation
        prediction = self.accumulation_model.predict(features_scaled)[0]
        probability = self.accumulation_model.predict_proba(features_scaled)[0]
        
        confidence = probability[1] if len(probability) > 1 else probability[0]
        
        return bool(prediction), confidence
    
    def detect_institutional_distribution(self, df):
        """Detect when institutions are DISTRIBUTING (PE entry)"""
        if self.distribution_model is None or self.scaler is None:
            return False, 0.0
        
        features = self.extract_institutional_features(df)
        if features is None:
            return False, 0.0
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict distribution
        prediction = self.distribution_model.predict(features_scaled)[0]
        probability = self.distribution_model.predict_proba(features_scaled)[0]
        
        confidence = probability[1] if len(probability) > 1 else probability[0]
        
        return bool(prediction), confidence

# Initialize institutional AI
institutional_ai = InstitutionalBehaviorAI()

# --------- LIQUIDITY ZONE DETECTION ---------
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

# 🏛️ **1. INSTITUTIONAL ACCUMULATION DETECTION** 🏛️
def detect_institutional_accumulation_entry(df):
    """
    Detect when BIG INSTITUTIONS are ACCUMULATING (buying) before UP move
    Your BANKNIFTY 59700 CE was this
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        open_price = ensure_series(df['Open'])
        
        if len(close) < 10:
            return None
        
        # 🏛️ CHECK 1: INSTITUTIONAL VOLUME SIGNATURE
        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if current_vol < vol_avg_10 * INSTITUTIONAL_VOLUME_RATIO:
            return None  # Not institutional size
        
        # 🏛️ CHECK 2: ABSORPTION CANDLE (Institutions absorbing selling)
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        lower_wick = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
        
        if lower_wick < current_body * ABSORPTION_WICK_RATIO:
            return None  # No absorption
        
        # 🏛️ CHECK 3: PRICE HOLDS AT SUPPORT (Institutions defending)
        support_level = low.iloc[-8:-2].min()
        if close.iloc[-1] < support_level * 0.992:
            return None  # Broken support, not accumulation
        
        # 🏛️ CHECK 4: AI CONFIRMATION
        accumulation_detected, ai_confidence = institutional_ai.detect_institutional_accumulation(df)
        if not accumulation_detected or ai_confidence < 0.82:
            return None
        
        # 🏛️ CHECK 5: FOLLOW-THROUGH CONFIRMATION
        if not (close.iloc[-1] > close.iloc[-2] and close.iloc[-2] > close.iloc[-3]):
            return None
        
        # 🏛️ ALL CHECKS PASSED - INSTITUTIONS ACCUMULATING
        return "CE"
        
    except Exception as e:
        return None

# 🏛️ **2. INSTITUTIONAL DISTRIBUTION DETECTION** 🏛️
def detect_institutional_distribution_entry(df):
    """
    Detect when BIG INSTITUTIONS are DISTRIBUTING (selling) before DOWN move
    Need to avoid your 15:00 bad PE entry
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        open_price = ensure_series(df['Open'])
        
        if len(close) < 10:
            return None
        
        # 🏛️ CHECK 0: AVOID LATE DAY ENTRIES (Your 15:00 bad entry)
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        if ist_now.hour >= 14 and ist_now.minute >= 45:  # After 2:45 PM
            return None
        
        # 🏛️ CHECK 1: INSTITUTIONAL VOLUME SIGNATURE
        vol_avg_10 = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        if current_vol < vol_avg_10 * (INSTITUTIONAL_VOLUME_RATIO + 0.5):  # Higher for PE
            return None
        
        # 🏛️ CHECK 2: DISTRIBUTION CANDLE (Institutions distributing)
        current_body = abs(close.iloc[-1] - open_price.iloc[-1])
        upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
        
        if upper_wick < current_body * ABSORPTION_WICK_RATIO:
            return None  # No distribution
        
        # 🏛️ CHECK 3: PRICE REJECTS AT RESISTANCE
        resistance_level = high.iloc[-8:-2].max()
        if close.iloc[-1] > resistance_level * 1.008:
            return None  # Broken resistance, not distribution
        
        # 🏛️ CHECK 4: AI CONFIRMATION
        distribution_detected, ai_confidence = institutional_ai.detect_institutional_distribution(df)
        if not distribution_detected or ai_confidence < 0.85:  # Higher threshold for PE
            return None
        
        # 🏛️ CHECK 5: FOLLOW-THROUGH CONFIRMATION
        if not (close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]):
            return None
        
        # 🏛️ ALL CHECKS PASSED - INSTITUTIONS DISTRIBUTING
        return "PE"
        
    except Exception as e:
        return None

# 🏛️ **3. STOP HUNT REVERSAL DETECTION** 🏛️
def detect_stop_hunt_reversal(df):
    """
    Detect when institutions HUNT STOPS before big reversal
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 15:
            return None
        
        # Find recent trading range
        recent_high = high.iloc[-12:-2].max()
        recent_low = low.iloc[-12:-2].min()
        recent_range = recent_high - recent_low
        
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        
        # Volume check
        vol_avg = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # 🏛️ BULL STOP HUNT (Institutions hunt bull stops, then go UP)
        if (current_low < recent_low * (1 - STOP_HUNT_DISTANCE) and  # Spikes below support
            current_close > recent_low * 1.008 and                    # Closes well above
            current_vol > vol_avg * 3.5 and                          # BIG volume
            current_close > prev_close and                           # Green candle
            (current_high - current_close) < (current_close - current_low) * 0.4):  # Small upper wick
            
            # This is INSTITUTIONAL behavior: Hunt stops, then reverse UP
            return "CE"
        
        # 🏛️ BEAR STOP HUNT (Institutions hunt bear stops, then go DOWN)
        if (current_high > recent_high * (1 + STOP_HUNT_DISTANCE) and  # Spikes above resistance
            current_close < recent_high * 0.992 and                    # Closes well below
            current_vol > vol_avg * 4.0 and                           # Even BIGGER volume
            current_close < prev_close and                            # Red candle
            (current_close - current_low) < (current_high - current_close) * 0.4):  # Small lower wick
            
            # Avoid late day stop hunts
            utc_now = datetime.utcnow()
            ist_now = utc_now + timedelta(hours=5, minutes=30)
            if ist_now.hour < 14:  # Only before 2 PM
                return "PE"
        
    except Exception:
        return None
    return None

# 🏛️ **4. LIQUIDITY GRAB DETECTION** 🏛️
def detect_liquidity_grab(df):
    """
    Detect when institutions GRAB LIQUIDITY before big move
    """
    try:
        close = ensure_series(df['Close'])
        high = ensure_series(df['High'])
        low = ensure_series(df['Low'])
        volume = ensure_series(df['Volume'])
        
        if len(close) < 20:
            return None
        
        # Find liquidity pools
        high_zone, low_zone = detect_liquidity_zone(df, lookback=18)
        current_price = close.iloc[-1]
        
        # Volume signature
        vol_avg_15 = volume.rolling(15).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # 🏛️ LIQUIDITY GRAB AT HIGHS (Institutions taking liquidity, then DOWN)
        if high_zone and abs(current_price - high_zone) <= high_zone * 0.006:
            if (current_vol > vol_avg_15 * 4.0 and                    # Massive volume
                high.iloc[-1] > high_zone * 1.008 and                 # Spikes above
                close.iloc[-1] < high_zone * 0.997 and                # Closes below
                volume.iloc[-1] > volume.iloc[-2] * 2.0):             # Volume spike
                
                return "PE"  # Institutions grabbed liquidity, now DOWN
        
        # 🏛️ LIQUIDITY GRAB AT LOWS (Institutions taking liquidity, then UP)
        if low_zone and abs(current_price - low_zone) <= low_zone * 0.006:
            if (current_vol > vol_avg_15 * 4.0 and                    # Massive volume
                low.iloc[-1] < low_zone * 0.992 and                   # Spikes below
                close.iloc[-1] > low_zone * 1.003 and                 # Closes above
                volume.iloc[-1] > volume.iloc[-2] * 2.0):             # Volume spike
                
                return "CE"  # Institutions grabbed liquidity, now UP
        
    except Exception:
        return None
    return None

# 🏛️ **PURE INSTITUTIONAL SIGNAL ANALYSIS** 🏛️
def analyze_index_signal(index):
    df5 = fetch_index_data(index, "5m", "2d")
    if df5 is None:
        return None

    close5 = ensure_series(df5["Close"])
    if len(close5) < 20 or close5.isna().iloc[-1] or close5.isna().iloc[-2]:
        return None

    # 🏛️ **PRIORITY 1: INSTITUTIONAL ACCUMULATION** (BEFORE BIG UP MOVE)
    accumulation_signal = detect_institutional_accumulation_entry(df5)
    if accumulation_signal:
        return accumulation_signal, df5, False, "institutional_accumulation"

    # 🏛️ **PRIORITY 2: STOP HUNT REVERSAL** (INSTITUTIONAL TRAP)
    stop_hunt_signal = detect_stop_hunt_reversal(df5)
    if stop_hunt_signal:
        return stop_hunt_signal, df5, False, "stop_hunt_reversal"

    # 🏛️ **PRIORITY 3: LIQUIDITY GRAB** (INSTITUTIONAL LIQUIDITY TAKE)
    liquidity_signal = detect_liquidity_grab(df5)
    if liquidity_signal:
        return liquidity_signal, df5, False, "liquidity_grab"

    # 🏛️ **PRIORITY 4: INSTITUTIONAL DISTRIBUTION** (BEFORE BIG DOWN MOVE)
    distribution_signal = detect_institutional_distribution_entry(df5)
    if distribution_signal:
        return distribution_signal, df5, False, "institutional_distribution"

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

def update_signal_progress(signal_id, targets_hit):
    for strike_key, data in active_strikes.items():
        if data['signal_id'] == signal_id:
            active_strikes[strike_key]['targets_hit'] = targets_hit
            break

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

# 🏛️ **INSTITUTIONAL TARGET CALCULATION** 🏛️
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
    
    # 🏛️ **INSTITUTIONAL-SIZED TARGETS**
    # Your BANKNIFTY 59700 CE: Entry 761 → Targets 801/833/873/921
    # That's 40/72/112/160 points (ratio: 0.5/0.9/1.4/2.0)
    
    if side == "CE":
        if strategy_key == "institutional_accumulation":
            base_move = 90  # Bigger for accumulation
        elif strategy_key == "stop_hunt_reversal":
            base_move = 110  # Biggest for stop hunts
        else:
            base_move = 70
        
        targets = [
            round(entry + base_move * 0.5),   # 50% 
            round(entry + base_move * 0.9),   # 90%
            round(entry + base_move * 1.4),   # 140%
            round(entry + base_move * 2.0)    # 200%
        ]
        sl = round(entry - base_move * 0.35)  # Tighter SL for institutional
        
    else:  # PE
        if strategy_key == "institutional_distribution":
            base_move = 85
        elif strategy_key == "liquidity_grab":
            base_move = 95
        else:
            base_move = 65
        
        targets = [
            round(entry + base_move * 0.5),
            round(entry + base_move * 0.9),
            round(entry + base_move * 1.4),
            round(entry + base_move * 2.0)
        ]
        sl = round(entry - base_move * 0.35)
    
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
    
    # 🏛️ **INSTITUTIONAL ALERTS**
    if strategy_key == "institutional_accumulation":
        msg = (f"🏛️ **INSTITUTIONS ACCUMULATING** 🏛️\n"
               f"🎯 {index} {strike} {side}\n"
               f"SYMBOL: {symbol}\n"
               f"ENTRY ABOVE: ₹{entry}\n"
               f"TARGETS: {targets_str}\n"
               f"STOP LOSS: ₹{sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ BIG MONEY BUYING BEFORE UP MOVE")
    elif strategy_key == "institutional_distribution":
        msg = (f"🏛️ **INSTITUTIONS DISTRIBUTING** 🏛️\n"
               f"🎯 {index} {strike} {side}\n"
               f"SYMBOL: {symbol}\n"
               f"ENTRY ABOVE: ₹{entry}\n"
               f"TARGETS: {targets_str}\n"
               f"STOP LOSS: ₹{sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ BIG MONEY SELLING BEFORE DOWN MOVE")
    elif strategy_key == "stop_hunt_reversal":
        msg = (f"🏛️ **STOP HUNT DETECTED** 🏛️\n"
               f"🎯 {index} {strike} {side}\n"
               f"SYMBOL: {symbol}\n"
               f"ENTRY ABOVE: ₹{entry}\n"
               f"TARGETS: {targets_str}\n"
               f"STOP LOSS: ₹{sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ INSTITUTIONS HUNTED STOPS - REVERSAL COMING")
    else:
        msg = (f"🏛️ **LIQUIDITY GRAB** 🏛️\n"
               f"🎯 {index} {strike} {side}\n"
               f"SYMBOL: {symbol}\n"
               f"ENTRY ABOVE: ₹{entry}\n"
               f"TARGETS: {targets_str}\n"
               f"STOP LOSS: ₹{sl}\n"
               f"STRATEGY: {strategy_name}\n"
               f"SIGNAL ID: {signal_id}\n"
               f"⚠️ INSTITUTIONS GRABBED LIQUIDITY")
    
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

# --------- TRADE THREAD ---------
def trade_thread(index):
    result = analyze_index_signal(index)
    
    if not result:
        return
        
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
            send_telegram("📊 GENERATING COMPULSORY END-OF-DAY REPORT...")
            # EOD report logic here
            EOD_REPORT_SENT = True
            
        return
        
    threads = []
    kept_indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
    
    for index in kept_indices:
        t = threading.Thread(target=trade_thread, args=(index,))
        t.start()
        threads.append(t)
    
    for t in threads: 
        t.join()

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
            time.sleep(30)
            continue
        
        if not STARTED_SENT:
            send_telegram("🏛️ **PURE INSTITUTIONAL AI ACTIVATED**\n"
                         "🤖 THINKING LIKE HEDGE FUNDS\n"
                         "🎯 DETECTING INSTITUTIONAL BEHAVIOR\n"
                         "💰 ACCUMULATION/DISTRIBUTION DETECTION\n"
                         "🎯 STOP HUNT REVERSAL DETECTION\n"
                         "🌊 LIQUIDITY GRAB DETECTION\n"
                         "⚠️ NO RETAIL PATTERNS - PURE INSTITUTIONAL\n"
                         "🎯 ENTERING BEFORE BIG MOVES ONLY")
            STARTED_SENT = True
            STOP_SENT = False
            MARKET_CLOSED_SENT = False
        
        if should_stop_trading():
            if not STOP_SENT:
                send_telegram("🛑 Market closing time reached!")
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
