# ULTIMATE INSTITUTIONAL INTELLIGENCE ANALYZER WITH ENHANCED OPTION ANALYSIS

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
from collections import deque

warnings.filterwarnings("ignore")

# --------- INSTITUTIONAL MONITORING CONFIG ---------
MOVE_THRESHOLD = 40  # Multi-candle moves
SINGLE_CANDLE_MOVE_THRESHOLD = 25  # Single candle moves
MOVE_TIME_WINDOW = 20
MULTI_CANDLE_COOLDOWN = 30
SINGLE_CANDLE_COOLDOWN = 15

# --------- ANGEL ONE LOGIN ---------
API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")

def angel_one_login():
    """Login to Angel One without error messages"""
    try:
        TOTP = pyotp.TOTP(TOTP_SECRET).now()
        client = SmartConnect(api_key=API_KEY)
        session = client.generateSession(CLIENT_CODE, PASSWORD, TOTP)
        return client
    except Exception:
        return None

# --------- TELEGRAM ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=10)
        return True
    except:
        return False

# --------- MARKET HOURS ---------
def is_market_open():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.time()
    return dtime(9,15) <= current_time_ist <= dtime(15,30)

# --------- DATA FETCHING ---------
def fetch_index_data_safe(index, interval="5m", period="1d"):
    """Fetch data with complete error handling"""
    try:
        symbol_map = {
            "NIFTY": "^NSEI", 
            "BANKNIFTY": "^NSEBANK", 
            "SENSEX": "^BSESN"
        }
        df = yf.download(symbol_map[index], period=period, interval=interval, progress=False)
        return df if not df.empty else None
    except Exception:
        return None

def ensure_series(data):
    try:
        return data.iloc[:,0] if isinstance(data, pd.DataFrame) else data.squeeze()
    except Exception:
        return None

# --------- ENHANCED OPTION STRIKE ANALYSIS ---------
def get_option_strike_details(index, strike_price, option_type, current_price, move_direction):
    """Get detailed information about the recommended option strike"""
    try:
        if strike_price == "Couldn't fetch" or option_type == "Couldn't fetch":
            return {"details": "Option strike calculation failed"}
        
        strike_price = int(strike_price)
        moneyness = ""
        
        # Calculate moneyness
        if option_type == "CE":
            if strike_price < current_price:
                moneyness = "ITM (In the Money)"
            elif strike_price == current_price:
                moneyness = "ATM (At the Money)"
            else:
                moneyness = "OTM (Out of the Money)"
        else:  # PE
            if strike_price > current_price:
                moneyness = "ITM (In the Money)"
            elif strike_price == current_price:
                moneyness = "ATM (At the Money)"
            else:
                moneyness = "OTM (Out of the Money)"
        
        # Calculate distance from current price
        distance = abs(strike_price - current_price)
        distance_percentage = (distance / current_price) * 100
        
        # Probability estimation based on distance
        if distance_percentage < 0.2:
            probability = "HIGH"
            risk = "LOW"
        elif distance_percentage < 0.5:
            probability = "MEDIUM"
            risk = "MODERATE"
        else:
            probability = "LOW"
            risk = "HIGH"
        
        # Premium estimation (rough calculation)
        if index == "NIFTY":
            base_premium = max(20, distance * 0.4)
        elif index == "BANKNIFTY":
            base_premium = max(40, distance * 0.3)
        else:
            base_premium = max(30, distance * 0.35)
        
        # Adjust premium based on moneyness
        if moneyness.startswith("ITM"):
            premium = base_premium * 1.5
        elif moneyness.startswith("ATM"):
            premium = base_premium
        else:
            premium = base_premium * 0.6
        
        return {
            "strike": strike_price,
            "type": option_type,
            "moneyness": moneyness,
            "distance_points": round(distance, 1),
            "distance_percentage": round(distance_percentage, 2),
            "probability": probability,
            "risk_level": risk,
            "estimated_premium": round(premium, 1),
            "current_price": current_price,
            "details": f"{strike_price} {option_type} | {moneyness} | {distance} points away"
        }
    except Exception:
        return {"details": "Option details calculation failed"}

def find_nearest_option_strike(index, current_price, move_direction):
    """Find nearest option strike based on move direction"""
    try:
        if index == "NIFTY":
            strike_step = 50
        elif index == "BANKNIFTY":
            strike_step = 100
        elif index == "SENSEX":
            strike_step = 100
        else:
            strike_step = 50
        
        if move_direction == "UP" or move_direction == "CE":
            strike = math.ceil(current_price / strike_step) * strike_step
            return int(strike), "CE"
        else:
            strike = math.floor(current_price / strike_step) * strike_step
            return int(strike), "PE"
    except Exception:
        return "Couldn't fetch", "Couldn't fetch"

# --------- COMPLETE INSTITUTIONAL ANALYSIS ENGINE ---------
class InstitutionalAnalysisEngine:
    def __init__(self):
        self.client = angel_one_login()
        self.historical_data = {}
    
    def calculate_all_institutional_parameters(self, index, df, move_info=None):
        """Calculate ALL institutional parameters with independent error handling"""
        try:
            if df is None or len(df) < 10:
                return self.get_empty_analysis()
            
            # Calculate each parameter independently
            analyses = {}
            
            # Each analysis is wrapped in try-except to prevent failure of one from affecting others
            analyses['order_book_imbalance'] = self.safe_analyze(self.analyze_order_book_imbalance, df)
            analyses['market_delta'] = self.safe_analyze(self.analyze_market_delta, df)
            analyses['cumulative_volume_delta'] = self.safe_analyze(self.analyze_cumulative_volume_delta, df)
            analyses['order_flow_pressure'] = self.safe_analyze(self.analyze_order_flow_pressure, df)
            analyses['iceberg_orders'] = self.safe_analyze(self.detect_iceberg_orders, df)
            analyses['liquidity_grab'] = self.safe_analyze(self.detect_liquidity_grab, df)
            analyses['spread_expansion'] = self.safe_analyze(self.analyze_spread_expansion, df)
            analyses['market_depth_thinning'] = self.safe_analyze(self.analyze_market_depth_thinning, df)
            analyses['volume_imbalance'] = self.safe_analyze(self.analyze_volume_imbalance, df)
            analyses['fvg_size_analysis'] = self.safe_analyze(self.analyze_fvg_size, df)
            analyses['vwap_deviation'] = self.safe_analyze(self.analyze_vwap_deviation, df)
            analyses['volatility_compression'] = self.safe_analyze(self.analyze_volatility_compression, df)
            analyses['whale_footprints'] = self.safe_analyze(self.detect_whale_footprints, df)
            
            # Overall sentiment
            analyses['institutional_sentiment'] = self.safe_analyze(
                self.calculate_overall_sentiment, analyses
            )
            
            return analyses
            
        except Exception as e:
            return self.get_empty_analysis()
    
    def safe_analyze(self, analysis_func, *args):
        """Safely execute analysis function with error handling"""
        try:
            return analysis_func(*args)
        except Exception:
            # Return appropriate empty structure based on function
            if analysis_func.__name__ in ['detect_iceberg_orders', 'detect_liquidity_grab', 'detect_whale_footprints']:
                return {"detected": "Insufficient data", "score": "N/A", "confidence": "N/A"}
            elif analysis_func.__name__ == 'analyze_fvg_size':
                return {"size": "Insufficient data", "type": "N/A", "interpretation": "Calculation failed"}
            else:
                return {"value": "Insufficient data", "interpretation": "Calculation failed"}
    
    def analyze_order_book_imbalance(self, df):
        """1. ORDER BOOK IMBALANCE (OBI)"""
        try:
            volume = ensure_series(df['Volume'])
            if volume is None or len(volume) < 5:
                return {"value": "Insufficient data", "interpretation": "Insufficient data"}
            
            # Estimate OBI based on price-volume relationship
            price_change = (df['Close'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1]
            volume_ratio = volume.iloc[-1] / volume.iloc[-5:].mean()
            
            # Simulate OBI calculation
            if price_change > 0.001 and volume_ratio > 1.5:
                obi_value = 0.7 + (price_change * 100)
            elif price_change < -0.001 and volume_ratio > 1.5:
                obi_value = -0.7 + (price_change * 100)
            else:
                obi_value = price_change * 50
            
            obi_value = max(-0.9, min(0.9, obi_value))
            
            interpretation = "Neutral"
            if obi_value >= 0.6:
                interpretation = "Strong Institutions Buying"
            elif obi_value <= -0.6:
                interpretation = "Strong Institutions Selling"
            elif obi_value >= 0.3:
                interpretation = "Moderate Buying"
            elif obi_value <= -0.3:
                interpretation = "Moderate Selling"
            
            return {
                "value": f"{obi_value:.3f}",
                "interpretation": interpretation,
                "signal": "STRONG" if abs(obi_value) >= 0.6 else "MODERATE" if abs(obi_value) >= 0.3 else "WEAK"
            }
        except Exception:
            return {"value": "Insufficient data", "interpretation": "Calculation failed"}
    
    def analyze_market_delta(self, df):
        """2. MARKET DELTA (Buy – Sell Volume)"""
        try:
            volume = ensure_series(df['Volume'])
            if volume is None or len(volume) < 3:
                return {"value": "Insufficient data", "interpretation": "Insufficient data"}
            
            # Estimate delta based on price movement and volume
            price_change = df['Close'].iloc[-1] - df['Open'].iloc[-1]
            current_volume = volume.iloc[-1]
            avg_volume = volume.iloc[-3:].mean()
            
            # Simulate delta calculation
            delta_ratio = (price_change / df['Open'].iloc[-1]) * current_volume / avg_volume
            delta_value = delta_ratio * 1000  # Scale to reasonable range
            
            interpretation = "Neutral"
            if delta_value > 500:
                interpretation = "Strong Aggressive Buying"
            elif delta_value < -500:
                interpretation = "Strong Aggressive Selling"
            elif delta_value > 200:
                interpretation = "Moderate Buying"
            elif delta_value < -200:
                interpretation = "Moderate Selling"
            
            return {
                "value": f"{delta_value:+.0f}",
                "interpretation": interpretation,
                "absorption": "High" if abs(delta_value) > 300 and abs(price_change) < 0.001 else "Low"
            }
        except Exception:
            return {"value": "Insufficient data", "interpretation": "Calculation failed"}
    
    def analyze_cumulative_volume_delta(self, df):
        """3. CUMULATIVE VOLUME DELTA (CVD)"""
        try:
            volume = ensure_series(df['Volume'])
            if volume is None or len(volume) < 10:
                return {"value": "Insufficient data", "interpretation": "Insufficient data"}
            
            # Calculate simple CVD approximation
            price_changes = df['Close'].pct_change().dropna()
            volume_changes = volume.pct_change().dropna()
            
            cvd_value = (price_changes * volume_changes).tail(5).sum() * 10000
            
            interpretation = "Neutral"
            if cvd_value > 50:
                interpretation = "CVD Rising - Bullish Breakout Likely"
            elif cvd_value < -50:
                interpretation = "CVD Falling - Bearish Breakdown Likely"
            elif cvd_value > 20:
                interpretation = "Moderate Buying Pressure"
            elif cvd_value < -20:
                interpretation = "Moderate Selling Pressure"
            
            return {
                "value": f"{cvd_value:+.2f}",
                "interpretation": interpretation,
                "trend": "BULLISH" if cvd_value > 0 else "BEARISH"
            }
        except Exception:
            return {"value": "Insufficient data", "interpretation": "Calculation failed"}
    
    def analyze_order_flow_pressure(self, df):
        """4. ORDER FLOW PRESSURE (Aggression Ratio)"""
        try:
            body_size = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
            range_size = df['High'].iloc[-1] - df['Low'].iloc[-1]
            
            if range_size > 0:
                aggression_ratio = body_size / range_size
            else:
                aggression_ratio = 0.5
            
            interpretation = "Balanced"
            if aggression_ratio > 0.7:
                interpretation = "Buyers Dominating"
            elif aggression_ratio < 0.3:
                interpretation = "Sellers Dominating"
            elif aggression_ratio > 0.6:
                interpretation = "Buyers Aggressive"
            elif aggression_ratio < 0.4:
                interpretation = "Sellers Aggressive"
            
            return {
                "value": f"{aggression_ratio:.3f}",
                "interpretation": interpretation,
                "pressure": "HIGH" if aggression_ratio > 0.7 or aggression_ratio < 0.3 else "MODERATE"
            }
        except Exception:
            return {"value": "Insufficient data", "interpretation": "Calculation failed"}
    
    def detect_iceberg_orders(self, df):
        """5. ICEBERG ORDER DETECTION"""
        try:
            volume = ensure_series(df['Volume'])
            if volume is None or len(volume) < 10:
                return {"detected": "Insufficient data", "score": "N/A", "confidence": "N/A"}
            
            # Look for volume patterns suggesting iceberg orders
            recent_vol = volume.iloc[-5:]

            # Check for consistent volume at similar price levels
            price_volatility = (df['High'].iloc[-5:] - df['Low'].iloc[-5:]).mean() / df['Close'].iloc[-5]
            volume_consistency = recent_vol.std() / recent_vol.mean()
            
            iceberg_score = 0
            if volume_consistency < 0.3 and price_volatility < 0.002:
                iceberg_score = 0.8
            elif volume_consistency < 0.5 and price_volatility < 0.003:
                iceberg_score = 0.6
            else:
                iceberg_score = 0.3
            
            detected = "Yes" if iceberg_score > 0.7 else "Possible" if iceberg_score > 0.5 else "No"
            
            return {
                "detected": detected,
                "score": f"{iceberg_score:.2f}",
                "confidence": "HIGH" if iceberg_score > 0.7 else "MEDIUM" if iceberg_score > 0.5 else "LOW"
            }
        except Exception:
            return {"detected": "Insufficient data", "score": "N/A", "confidence": "N/A"}
    
    def detect_liquidity_grab(self, df):
        """6. LIQUIDITY GRAB SIGNAL (Stop Hunt)"""
        try:
            if len(df) < 3:
                return {"detected": "Insufficient data", "signal_strength": "N/A", "implication": "Calculation failed"}
            
            current_high = df['High'].iloc[-1]
            current_low = df['Low'].iloc[-1]
            current_open = df['Open'].iloc[-1]
            current_close = df['Close'].iloc[-1]
            
            prev_high = df['High'].iloc[-2]
            prev_low = df['Low'].iloc[-2]
            
            body_size = abs(current_close - current_open)
            upper_wick = current_high - max(current_open, current_close)
            lower_wick = min(current_open, current_close) - current_low
            
            # Liquidity grab conditions
            wick_breaks_high = current_high > prev_high and upper_wick > 2 * body_size
            wick_breaks_low = current_low < prev_low and lower_wick > 2 * body_size
            
            detected = "Yes" if wick_breaks_high or wick_breaks_low else "No"
            direction = "UP" if wick_breaks_low else "DOWN" if wick_breaks_high else "NONE"
            
            return {
                "detected": detected,
                "direction": direction,
                "signal_strength": "STRONG" if detected == "Yes" else "WEAK",
                "implication": "25+ Move Expected Next" if detected == "Yes" else "No Clear Signal"
            }
        except Exception:
            return {"detected": "Insufficient data", "signal_strength": "N/A", "implication": "Calculation failed"}
    
    def analyze_spread_expansion(self, df):
        """7. SPREAD EXPANSION"""
        try:
            if len(df) < 10:
                return {"expansion": "Insufficient data", "interpretation": "Insufficient data"}
            
            # Calculate volatility as proxy for spread
            recent_atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 5).average_true_range().iloc[-1]
            avg_atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 10).average_true_range().iloc[-1]
            
            if avg_atr > 0:
                spread_change = ((recent_atr - avg_atr) / avg_atr) * 100
            else:
                spread_change = 0
            
            interpretation = "Normal"
            if spread_change > 50:
                interpretation = "Institutions Preparing Breakout"
            elif spread_change > 30:
                interpretation = "Moderate Spread Expansion"
            elif spread_change < -30:
                interpretation = "Spread Compression"
            
            return {
                "expansion": f"{spread_change:+.1f}%",
                "interpretation": interpretation,
                "signal": "STRONG" if spread_change > 50 else "MODERATE" if spread_change > 30 else "WEAK"
            }
        except Exception:
            return {"expansion": "Insufficient data", "interpretation": "Calculation failed"}
    
    def analyze_market_depth_thinning(self, df):
        """8. MARKET DEPTH THINNING"""
        try:
            volume = ensure_series(df['Volume'])
            if volume is None or len(volume) < 10:
                return {"ratio": "Insufficient data", "interpretation": "Insufficient data"}
            
            # Use volume volatility as proxy for depth thinning
            recent_vol = volume.iloc[-5:].std()
            avg_vol = volume.iloc[-10:].std()
            
            if avg_vol > 0:
                depth_ratio = recent_vol / avg_vol
            else:
                depth_ratio = 1.0
            
            interpretation = "Normal Depth"
            if depth_ratio < 0.5:
                interpretation = "Depth Thin - Violent Move Expected"
            elif depth_ratio < 0.8:
                interpretation = "Moderate Thinning"
            elif depth_ratio > 1.5:
                interpretation = "Depth Building"
            
            return {
                "ratio": f"{depth_ratio:.2f}",
                "interpretation": interpretation,
                "alert": "HIGH" if depth_ratio < 0.5 else "MEDIUM" if depth_ratio < 0.8 else "LOW"
            }
        except Exception:
            return {"ratio": "Insufficient data", "interpretation": "Calculation failed"}
    
    def analyze_volume_imbalance(self, df):
        """9. VOLUME IMBALANCE (VI)"""
        try:
            volume = ensure_series(df['Volume'])
            if volume is None or len(volume) < 5:
                return {"imbalance": "Insufficient data", "interpretation": "Insufficient data"}
            
            # Estimate volume imbalance
            price_change = (df['Close'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1]
            volume_surge = volume.iloc[-1] / volume.iloc[-5:].mean()
            
            # Simulate volume imbalance
            vi_value = price_change * volume_surge * 10
            vi_value = max(-0.9, min(0.9, vi_value))
            
            interpretation = "Balanced"
            if vi_value >= 0.6:
                interpretation = "Strong Buying Imbalance"
            elif vi_value <= -0.6:
                interpretation = "Strong Selling Imbalance"
            elif vi_value >= 0.3:
                interpretation = "Moderate Buying"
            elif vi_value <= -0.3:
                interpretation = "Moderate Selling"
            
            return {
                "imbalance": f"{vi_value:.3f}",
                "interpretation": interpretation,
                "strength": "STRONG" if abs(vi_value) >= 0.6 else "MODERATE"
            }
        except Exception:
            return {"imbalance": "Insufficient data", "interpretation": "Calculation failed"}
    
    def analyze_fvg_size(self, df):
        """10. FVG SIZE ANALYSIS"""
        try:
            if len(df) < 3:
                return {"size": "Insufficient data", "type": "N/A", "interpretation": "Insufficient data"}
            
            current_low = df['Low'].iloc[-1]
            current_high = df['High'].iloc[-1]
            prev_high = df['High'].iloc[-2]
            prev_low = df['Low'].iloc[-2]
            
            bullish_fvg = current_low - prev_high if current_low > prev_high else 0
            bearish_fvg = prev_low - current_high if current_high < prev_low else 0
            
            fvg_size = max(bullish_fvg, bearish_fvg)
            fvg_type = "BULLISH" if bullish_fvg > 0 else "BEARISH" if bearish_fvg > 0 else "NONE"
            
            interpretation = "No Significant FVG"
            if fvg_size > 20:
                interpretation = "Large FVG - Institutional Displacement"
            elif fvg_size > 10:
                interpretation = "Moderate FVG"
            elif fvg_size > 0:
                interpretation = "Small FVG"
            
            return {
                "size": f"{fvg_size:.1f}",
                "type": fvg_type,
                "interpretation": interpretation
            }
        except Exception:
            return {"size": "Insufficient data", "type": "N/A", "interpretation": "Calculation failed"}
    
    def analyze_vwap_deviation(self, df):
        """11. VWAP DEVIATION"""
        try:
            if len(df) < 20:
                return {"deviation": "Insufficient data", "interpretation": "Insufficient data"}
            
            vwap = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price().iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            deviation = ((current_price - vwap) / vwap) * 100
            
            interpretation = "At VWAP"
            if deviation > 1.0:
                interpretation = "Far Above VWAP - Mean Reversion Likely"
            elif deviation < -1.0:
                interpretation = "Far Below VWAP - Mean Reversion Likely"
            elif deviation > 0.5:
                interpretation = "Above VWAP"
            elif deviation < -0.5:
                interpretation = "Below VWAP"
            
            return {
                "deviation": f"{deviation:+.2f}%",
                "interpretation": interpretation,
                "vwap_level": vwap
            }
        except Exception:
            return {"deviation": "Insufficient data", "interpretation": "Calculation failed"}
    
    def analyze_volatility_compression(self, df):
        """12. VOLATILITY COMPRESSION"""
        try:
            if len(df) < 10:
                return {"compression": "Insufficient data", "interpretation": "Insufficient data"}
            
            atr_5 = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 5).average_true_range().iloc[-1]
            atr_10 = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 10).average_true_range().iloc[-1]
            
            if atr_10 > 0:
                compression_ratio = atr_5 / atr_10
            else:
                compression_ratio = 1.0
            
            interpretation = "Normal Volatility"
            if compression_ratio < 0.7:
                interpretation = "Volatility Compression - Squeeze Building"
            elif compression_ratio > 1.3:
                interpretation = "Volatility Expansion"
            
            return {
                "compression": f"{compression_ratio:.2f}",
                "interpretation": interpretation,
                "signal": "COMPRESSION" if compression_ratio < 0.7 else "EXPANSION" if compression_ratio > 1.3 else "NORMAL"
            }
        except Exception:
            return {"compression": "Insufficient data", "interpretation": "Calculation failed"}
    
    def detect_whale_footprints(self, df):
        """13. WHALE FOOTPRINTS"""
        try:
            volume = ensure_series(df['Volume'])
            if volume is None or len(volume) < 10:
                return {"detected": "Insufficient data", "confidence": "N/A", "activity": "N/A"}
            
            # Look for whale footprint patterns
            large_trades = volume > volume.rolling(10).mean() * 2
            large_trades_recent = large_trades.tail(5).sum()
            
            price_consistency = df['Close'].pct_change().abs().tail(5).mean()
            
            whale_score = min(1.0, large_trades_recent / 3) * (1 - price_consistency * 100)
            
            detected = "Yes" if whale_score > 0.7 else "Possible" if whale_score > 0.4 else "No"
            
            return {
                "detected": detected,
                "confidence": f"{whale_score:.2f}",
                "activity": "HIGH" if whale_score > 0.7 else "MODERATE" if whale_score > 0.4 else "LOW"
            }
        except Exception:
            return {"detected": "Insufficient data", "confidence": "N/A", "activity": "N/A"}
    
    def calculate_overall_sentiment(self, analyses):
        """Calculate overall institutional sentiment"""
        try:
            score = 0
            
            # OBI contribution
            obi = analyses['order_book_imbalance']
            if obi['value'] != "Insufficient data":
                obi_val = float(obi['value'])
                score += obi_val * 10
            
            # Delta contribution
            delta = analyses['market_delta']
            if delta['value'] != "Insufficient data" and delta['value'].replace('+', '').replace('-', '').isdigit():
                delta_val = float(delta['value'].replace('+', ''))
                score += delta_val / 100
            
            # CVD contribution
            cvd = analyses['cumulative_volume_delta']
            if cvd['value'] != "Insufficient data" and cvd['value'].replace('+', '').replace('-', '').replace('.', '').isdigit():
                cvd_val = float(cvd['value'].replace('+', ''))
                score += cvd_val / 10
            
            # Order flow contribution
            order_flow = analyses['order_flow_pressure']
            if order_flow['value'] != "Insufficient data":
                flow_val = float(order_flow['value'])
                if flow_val > 0.6:
                    score += 2
                elif flow_val < 0.4:
                    score -= 2
            
            sentiment = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "NEUTRAL"
            confidence = "HIGH" if abs(score) >= 5 else "MEDIUM" if abs(score) >= 3 else "LOW"
            
            return {
                "score": round(score, 2),
                "sentiment": sentiment,
                "confidence": confidence
            }
        except Exception:
            return {"score": 0, "sentiment": "NEUTRAL", "confidence": "LOW"}
    
    def get_empty_analysis(self):
        """Return empty analysis when data is insufficient"""
        empty_val = {"value": "Insufficient data", "interpretation": "Insufficient data"}
        return {
            'order_book_imbalance': empty_val,
            'market_delta': empty_val,
            'cumulative_volume_delta': empty_val,
            'order_flow_pressure': empty_val,
            'iceberg_orders': {"detected": "Insufficient data", "score": "N/A", "confidence": "N/A"},
            'liquidity_grab': {"detected": "Insufficient data", "signal_strength": "N/A", "implication": "Calculation failed"},
            'spread_expansion': empty_val,
            'market_depth_thinning': empty_val,
            'volume_imbalance': empty_val,
            'fvg_size_analysis': {"size": "Insufficient data", "type": "N/A", "interpretation": "Calculation failed"},
            'vwap_deviation': empty_val,
            'volatility_compression': empty_val,
            'whale_footprints': {"detected": "Insufficient data", "confidence": "N/A", "activity": "N/A"},
            'institutional_sentiment': {"score": 0, "sentiment": "NEUTRAL", "confidence": "LOW"}
        }

# --------- SINGLE CANDLE BIG MOVE ANALYZER ---------
class SingleCandleAnalyzer:
    def __init__(self):
        self.institutional_engine = InstitutionalAnalysisEngine()
        self.last_analysis_time = {}
        self.analyzed_candles = set()
    
    def detect_single_candle_big_move(self, index, df_5min, df_1min):
        """Detect 25+ point moves in single candles"""
        try:
            # Check 5-minute data first
            if df_5min is not None and len(df_5min) >= 3:
                move_5min = self.analyze_single_candle_move(df_5min, "5min")
                if move_5min:
                    return move_5min
            
            # Check 1-minute data
            if df_1min is not None and len(df_1min) >= 3:
                move_1min = self.analyze_single_candle_move(df_1min, "1min") 
                if move_1min:
                    return move_1min
                    
        except Exception:
            pass
        return None
    
    def analyze_single_candle_move(self, df, interval):
        """Analyze single candle moves"""
        try:
            close = ensure_series(df['Close'])
            high = ensure_series(df['High'])
            low = ensure_series(df['Low'])
            open_price = ensure_series(df['Open'])
            
            if len(close) < 3:
                return None
            
            current_candle_range = high.iloc[-1] - low.iloc[-1]
            current_candle_change = close.iloc[-1] - open_price.iloc[-1]
            
            if abs(current_candle_change) >= SINGLE_CANDLE_MOVE_THRESHOLD:
                direction = "UP" if current_candle_change > 0 else "DOWN"
                
                # Analyze previous candle
                prev_candle_range = high.iloc[-2] - low.iloc[-2]
                prev_candle_change = close.iloc[-2] - open_price.iloc[-2]
                
                # Detect FVG
                fvg_present = self.detect_fvg(high, low, close)
                
                return {
                    'interval': interval,
                    'direction': direction,
                    'points': abs(current_candle_change),
                    'range': current_candle_range,
                    'timestamp': df.index[-1],
                    'current_candle': {
                        'open': open_price.iloc[-1],
                        'high': high.iloc[-1],
                        'low': low.iloc[-1],
                        'close': close.iloc[-1]
                    },
                    'previous_candle': {
                        'open': open_price.iloc[-2],
                        'high': high.iloc[-2],
                        'low': low.iloc[-2],
                        'close': close.iloc[-2],
                        'change': prev_candle_change,
                        'range': prev_candle_range
                    },
                    'fvg_present': fvg_present,
                    'fvg_type': self.get_fvg_type(high, low, direction)
                }
                
        except Exception:
            pass
        return None
    
    def detect_fvg(self, high, low, close):
        """Detect Fair Value Gap"""
        try:
            if len(high) < 3:
                return False
            
            # Bullish FVG: Current low > Previous high
            bullish_fvg = low.iloc[-1] > high.iloc[-2]
            
            # Bearish FVG: Current high < Previous low  
            bearish_fvg = high.iloc[-1] < low.iloc[-2]
            
            return bullish_fvg or bearish_fvg
        except Exception:
            return False
    
    def get_fvg_type(self, high, low, direction):
        """Get FVG type"""
        try:
            if len(high) < 3:
                return "UNKNOWN"
            
            if direction == "UP" and low.iloc[-1] > high.iloc[-2]:
                return "BULLISH_FVG"
            elif direction == "DOWN" and high.iloc[-1] < low.iloc[-2]:
                return "BEARISH_FVG"
            else:
                return "NO_FVG"
        except Exception:
            return "UNKNOWN"
    
    def analyze_single_candle_institutional_data(self, index, move_info, df_5min, df_1min):
        """Comprehensive institutional analysis of single candle move"""
        try:
            # Find nearest option strike
            current_price = move_info['current_candle']['close']
            option_strike, option_type = find_nearest_option_strike(index, current_price, move_info['direction'])
            
            # Get option strike details
            option_details = get_option_strike_details(index, option_strike, option_type, current_price, move_info['direction'])
            
            # Get institutional analysis
            institutional_data = self.institutional_engine.calculate_all_institutional_parameters(
                index, df_5min if move_info['interval'] == '5min' else df_1min, move_info)
            
            return {
                'index': index,
                'move_info': move_info,
                'analysis_time': datetime.utcnow() + timedelta(hours=5, minutes=30),
                'option_strike': option_strike,
                'option_type': option_type,
                'option_details': option_details,
                'institutional_analysis': institutional_data,
                'trading_implications': self.generate_trading_implications(move_info, institutional_data, option_details)
            }
            
        except Exception as e:
            return {'error': f"Single candle analysis failed: {str(e)}"}
    
    def generate_trading_implications(self, move_info, institutional_data, option_details):
        """Generate trading implications"""
        direction = move_info['direction']
        fvg_type = move_info.get('fvg_type', 'NO_FVG')
        sentiment = institutional_data['institutional_sentiment']
        
        implications = []
        
        if direction == "UP":
            if fvg_type == "BULLISH_FVG":
                implications.append("STRONG BULLISH FVG - High probability CE movement")
            else:
                implications.append(f"BULLISH MOVE {move_info['points']} points - Consider CE positions")
        else:
            if fvg_type == "BEARISH_FVG":
                implications.append("STRONG BEARISH FVG - High probability PE movement") 
            else:
                implications.append(f"BEARISH MOVE {move_info['points']} points - Consider PE positions")
        
        # Add institutional implications
        if sentiment.get('sentiment') == 'BULLISH':
            implications.append("Strong institutional buying detected")
        elif sentiment.get('sentiment') == 'BEARISH':
            implications.append("Strong institutional selling detected")
        
        # Add option-specific implications
        if 'probability' in option_details:
            implications.append(f"Option Probability: {option_details['probability']}")
            implications.append(f"Risk Level: {option_details['risk_level']}")
        
        # Add specific institutional signals
        obi = institutional_data['order_book_imbalance']
        if obi.get('signal') == 'STRONG':
            implications.append(f"Strong order book imbalance: {obi.get('interpretation', '')}")
        
        liquidity = institutional_data['liquidity_grab']
        if liquidity.get('detected') == 'Yes':
            implications.append("Liquidity grab detected - expect follow-through")
        
        return implications
    
    def format_single_candle_analysis(self, analysis):
        """Format single candle analysis message"""
        try:
            if 'error' in analysis:
                return f"⚠️ Single Candle Analysis Error: {analysis['error']}"
            
            move = analysis['move_info']
            inst = analysis['institutional_analysis']
            option_details = analysis['option_details']
            
            msg = f"""
🚨 <b>SINGLE CANDLE BIG MOVE DETECTED</b> 🚨

📊 <b>INDEX</b>: {analysis['index']}
⏰ <b>INTERVAL</b>: {move['interval']}
🎯 <b>MOVE</b>: {move['direction']} {move['points']} points
🕒 <b>TIME</b>: {analysis['analysis_time'].strftime('%H:%M:%S')}

📈 <b>CURRENT CANDLE</b> ({move['timestamp'].strftime('%H:%M')}):
• Open: {move['current_candle']['open']:.1f} → Close: {move['current_candle']['close']:.1f}
• Range: {move['range']:.1f} points
• FVG: {move['fvg_present']} ({move.get('fvg_type', 'N/A')})

<b>🚨 INSTITUTIONAL ORDER FLOW ANALYSIS</b>

1. <b>ORDER BOOK IMBALANCE</b>: {inst['order_book_imbalance']['value']}
   → {inst['order_book_imbalance']['interpretation']}

2. <b>MARKET DELTA</b>: {inst['market_delta']['value']}
   → {inst['market_delta']['interpretation']}

3. <b>CUMULATIVE VOLUME DELTA</b>: {inst['cumulative_volume_delta']['value']}
   → {inst['cumulative_volume_delta']['interpretation']}

4. <b>ORDER FLOW PRESSURE</b>: {inst['order_flow_pressure']['value']}
   → {inst['order_flow_pressure']['interpretation']}

5. <b>ICEBERG ORDERS</b>: {inst['iceberg_orders']['detected']}
   → Score: {inst['iceberg_orders']['score']}

6. <b>LIQUIDITY GRAB</b>: {inst['liquidity_grab']['detected']}
   → {inst['liquidity_grab']['implication']}

7. <b>SPREAD EXPANSION</b>: {inst['spread_expansion']['expansion']}
   → {inst['spread_expansion']['interpretation']}

8. <b>MARKET DEPTH</b>: {inst['market_depth_thinning']['ratio']}
   → {inst['market_depth_thinning']['interpretation']}

9. <b>VOLUME IMBALANCE</b>: {inst['volume_imbalance']['imbalance']}
   → {inst['volume_imbalance']['interpretation']}

10. <b>FVG SIZE</b>: {inst['fvg_size_analysis']['size']}
    → {inst['fvg_size_analysis']['type']} - {inst['fvg_size_analysis']['interpretation']}

11. <b>VWAP DEVIATION</b>: {inst['vwap_deviation']['deviation']}
    → {inst['vwap_deviation']['interpretation']}

12. <b>VOLATILITY COMPRESSION</b>: {inst['volatility_compression']['compression']}
    → {inst['volatility_compression']['interpretation']}

13. <b>WHALE FOOTPRINTS</b>: {inst['whale_footprints']['detected']}
    → Confidence: {inst['whale_footprints']['confidence']}

💼 <b>OVERALL INSTITUTIONAL SENTIMENT</b>:
• Sentiment: {inst['institutional_sentiment']['sentiment']}
• Confidence: {inst['institutional_sentiment']['confidence']}
• Score: {inst['institutional_sentiment']['score']}/10

🎯 <b>OPTION STRIKE ANALYSIS</b>:
• Strike: {option_details.get('strike', 'N/A')} {option_details.get('type', 'N/A')}
• Moneyness: {option_details.get('moneyness', 'N/A')}
• Distance: {option_details.get('distance_points', 'N/A')} points ({option_details.get('distance_percentage', 'N/A')}%)
• Probability: {option_details.get('probability', 'N/A')}
• Risk Level: {option_details.get('risk_level', 'N/A')}
• Estimated Premium: ₹{option_details.get('estimated_premium', 'N/A')}
• Current Price: {option_details.get('current_price', 'N/A')}

💡 <b>TRADING IMPLICATIONS</b>:
{' | '.join(analysis.get('trading_implications', ['No clear implications']))}

─────────────────────────────
            """
            return msg
        except Exception as e:
            return f"Error formatting single candle message: {str(e)}"
    
    def should_analyze_single_candle(self, index):
        """Cooldown check for single candle analysis"""
        try:
            current_time = time.time()
            if index in self.last_analysis_time:
                time_since_last = current_time - self.last_analysis_time[index]
                if time_since_last < SINGLE_CANDLE_COOLDOWN * 60:
                    return False
            return True
        except Exception:
            return True
    
    def update_single_candle_analysis_time(self, index):
        """Update analysis time for single candle"""
        try:
            self.last_analysis_time[index] = time.time()
        except Exception:
            pass

# --------- MULTI-CANDLE ANALYZER ---------
class MultiCandleAnalyzer:
    def __init__(self):
        self.institutional_engine = InstitutionalAnalysisEngine()
        self.last_analysis_time = {}
        self.analyzed_moves = set()
    
    def detect_big_move(self, index, df):
        """Detect 40+ point moves over multiple candles"""
        try:
            if df is None or len(df) < 10:
                return None
            
            close_data = ensure_series(df['Close'])
            current_price = close_data.iloc[-1]
            
            lookback_bars = min(MOVE_TIME_WINDOW // 5, len(close_data) - 1)
            start_price = close_data.iloc[-lookback_bars]
            
            move_points = current_price - start_price
            move_percentage = (move_points / start_price) * 100
            
            if abs(move_points) >= MOVE_THRESHOLD:
                # Find nearest option strike
                option_strike, option_type = find_nearest_option_strike(index, current_price, 
                                                                       "UP" if move_points > 0 else "DOWN")
                
                return {
                    'direction': "UP" if move_points > 0 else "DOWN",
                    'points': abs(move_points),
                    'percentage': abs(move_percentage),
                    'start_price': start_price,
                    'current_price': current_price,
                    'start_time': df.index[-lookback_bars],
                    'current_time': df.index[-1],
                    'option_strike': option_strike,
                    'option_type': option_type,
                    'move_strength': 'STRONG' if abs(move_percentage) > 0.4 else 'MODERATE'
                }
        except Exception:
            return None
        return None
    
    def analyze_multi_candle_institutional_data(self, index, move_info, df):
        """Comprehensive multi-candle analysis"""
        try:
            # Get option strike details
            option_details = get_option_strike_details(index, move_info['option_strike'], move_info['option_type'], 
                                                     move_info['current_price'], move_info['direction'])
            
            # Get institutional analysis
            institutional_data = self.institutional_engine.calculate_all_institutional_parameters(index, df, move_info)
            
            return {
                'index': index,
                'move_info': move_info,
                'analysis_time': datetime.utcnow() + timedelta(hours=5, minutes=30),
                'option_details': option_details,
                'institutional_analysis': institutional_data,
                'trading_implications': self.generate_trading_implications(move_info, institutional_data, option_details)
            }
            
        except Exception as e:
            return {'error': f"Multi-candle analysis failed: {str(e)}"}
    
    def generate_trading_implications(self, move_info, institutional_data, option_details):
        """Generate trading implications"""
        direction = move_info['direction']
        strength = move_info['move_strength']
        sentiment = institutional_data['institutional_sentiment']
        
        implications = []
        
        if direction == 'UP':
            if sentiment['sentiment'] == 'BULLISH' and strength == 'STRONG':
                implications.append("STRONG BULLISH MOVE - High probability of continuation")
                implications.append("Consider CE positions on pullbacks")
            elif sentiment['sentiment'] == 'BULLISH':
                implications.append("MODERATE BULLISH MOVE - Watch for confirmation")
            else:
                implications.append("CAUTION: Bullish move but weak institutional support")
        else:
            if sentiment['sentiment'] == 'BEARISH' and strength == 'STRONG':
                implications.append("STRONG BEARISH MOVE - High probability of continuation")
                implications.append("Consider PE positions on bounces")
            elif sentiment['sentiment'] == 'BEARISH':
                implications.append("MODERATE BEARISH MOVE - Watch for confirmation")
            else:
                implications.append("CAUTION: Bearish move but weak institutional support")
        
        # Add option-specific implications
        if 'probability' in option_details:
            implications.append(f"Option Probability: {option_details['probability']}")
            implications.append(f"Risk Level: {option_details['risk_level']}")
        
        # Add specific institutional signals
        obi = institutional_data['order_book_imbalance']
        if obi.get('signal') == 'STRONG':
            implications.append(f"Strong order book imbalance: {obi.get('interpretation', '')}")
        
        return implications
    
    def format_multi_candle_analysis(self, analysis):
        """Format multi-candle analysis message"""
        try:
            if 'error' in analysis:
                return f"⚠️ Multi-Candle Analysis Error: {analysis['error']}"
            
            move = analysis['move_info']
            inst = analysis['institutional_analysis']
            option_details = analysis['option_details']
            
            msg = f"""
🏛️ <b>MULTI-CANDLE INSTITUTIONAL MOVE</b> 🏛️

📊 <b>INDEX</b>: {analysis['index']}
🎯 <b>MOVE</b>: {move['direction']} {move['points']} points ({move['percentage']:.2f}%)
💪 <b>STRENGTH</b>: {move['move_strength']}
🕒 <b>PERIOD</b>: {move['start_time'].strftime('%H:%M')} → {move['current_time'].strftime('%H:%M')}
⏰ <b>ANALYSIS</b>: {analysis['analysis_time'].strftime('%H:%M:%S')}

<b>🚨 INSTITUTIONAL ORDER FLOW ANALYSIS</b>

1. <b>ORDER BOOK IMBALANCE</b>: {inst['order_book_imbalance']['value']}
   → {inst['order_book_imbalance']['interpretation']}

2. <b>MARKET DELTA</b>: {inst['market_delta']['value']}
   → {inst['market_delta']['interpretation']}

3. <b>CUMULATIVE VOLUME DELTA</b>: {inst['cumulative_volume_delta']['value']}
   → {inst['cumulative_volume_delta']['interpretation']}

4. <b>ORDER FLOW PRESSURE</b>: {inst['order_flow_pressure']['value']}
   → {inst['order_flow_pressure']['interpretation']}

5. <b>ICEBERG ORDERS</b>: {inst['iceberg_orders']['detected']}
   → Score: {inst['iceberg_orders']['score']}

6. <b>LIQUIDITY GRAB</b>: {inst['liquidity_grab']['detected']}
   → {inst['liquidity_grab']['implication']}

7. <b>SPREAD EXPANSION</b>: {inst['spread_expansion']['expansion']}
   → {inst['spread_expansion']['interpretation']}

8. <b>MARKET DEPTH</b>: {inst['market_depth_thinning']['ratio']}
   → {inst['market_depth_thinning']['interpretation']}

9. <b>VOLUME IMBALANCE</b>: {inst['volume_imbalance']['imbalance']}
   → {inst['volume_imbalance']['interpretation']}

10. <b>FVG SIZE</b>: {inst['fvg_size_analysis']['size']}
    → {inst['fvg_size_analysis']['type']} - {inst['fvg_size_analysis']['interpretation']}

11. <b>VWAP DEVIATION</b>: {inst['vwap_deviation']['deviation']}
    → {inst['vwap_deviation']['interpretation']}

12. <b>VOLATILITY COMPRESSION</b>: {inst['volatility_compression']['compression']}
    → {inst['volatility_compression']['interpretation']}

13. <b>WHALE FOOTPRINTS</b>: {inst['whale_footprints']['detected']}
    → Confidence: {inst['whale_footprints']['confidence']}

💼 <b>OVERALL INSTITUTIONAL SENTIMENT</b>:
• Sentiment: {inst['institutional_sentiment']['sentiment']}
• Confidence: {inst['institutional_sentiment']['confidence']}
• Score: {inst['institutional_sentiment']['score']}/10

🎯 <b>OPTION STRIKE ANALYSIS</b>:
• Strike: {option_details.get('strike', 'N/A')} {option_details.get('type', 'N/A')}
• Moneyness: {option_details.get('moneyness', 'N/A')}
• Distance: {option_details.get('distance_points', 'N/A')} points ({option_details.get('distance_percentage', 'N/A')}%)
• Probability: {option_details.get('probability', 'N/A')}
• Risk Level: {option_details.get('risk_level', 'N/A')}
• Estimated Premium: ₹{option_details.get('estimated_premium', 'N/A')}
• Current Price: {option_details.get('current_price', 'N/A')}

💡 <b>TRADING IMPLICATIONS</b>:
{' | '.join(analysis.get('trading_implications', ['No clear implications']))}

─────────────────────────────
            """
            return msg
        except Exception as e:
            return f"Error formatting multi-candle message: {str(e)}"
    
    def should_analyze_multi_candle(self, index):
        """Cooldown check for multi-candle analysis"""
        try:
            current_time = time.time()
            if index in self.last_analysis_time:
                time_since_last = current_time - self.last_analysis_time[index]
                if time_since_last < MULTI_CANDLE_COOLDOWN * 60:
                    return False
            return True
        except Exception:
            return True
    
    def update_multi_candle_analysis_time(self, index):
        """Update analysis time for multi-candle"""
        try:
            self.last_analysis_time[index] = time.time()
        except Exception:
            pass

# --------- MAIN MONITORING LOOP ---------
def monitor_all_systems():
    """Monitor both single-candle and multi-candle moves"""
    single_analyzer = SingleCandleAnalyzer()
    multi_analyzer = MultiCandleAnalyzer()
    
    single_analyzed = set()
    multi_analyzed = set()
    
    send_telegram("🚀 <b>ULTIMATE INSTITUTIONAL INTELLIGENCE STARTED</b>\n"
                 "• Multi-Candle: 40+ points over 20min\n" 
                 "• Single-Candle: 25+ points in 1 candle\n"
                 "• Both 1min & 5min data\n"
                 "• Complete Institutional Parameters\n"
                 "• Enhanced Option Strike Analysis\n"
                 "• 13 Institutional Metrics Monitoring")
    
    while True:
        try:
            if not is_market_open():
                time.sleep(60)
                continue
            
            current_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
            time_key = current_time.strftime("%H:%M")
            
            for index in ["NIFTY", "BANKNIFTY", "SENSEX"]:
                try:
                    # Fetch data for both systems
                    df_5min = fetch_index_data_safe(index, "5m", "1d")
                    df_1min = fetch_index_data_safe(index, "1m", "1d")
                    
                    if df_5min is None and df_1min is None:
                        continue
                    
                    # 🆕 SINGLE CANDLE MONITORING
                    single_move = single_analyzer.detect_single_candle_big_move(index, df_5min, df_1min)
                    if (single_move and 
                        single_analyzer.should_analyze_single_candle(index)):
                        
                        move_id = f"SINGLE_{index}_{single_move['direction']}_{time_key}"
                        if move_id not in single_analyzed:
                            analysis = single_analyzer.analyze_single_candle_institutional_data(
                                index, single_move, df_5min, df_1min)
                            
                            message = single_analyzer.format_single_candle_analysis(analysis)
                            if send_telegram(message):
                                single_analyzed.add(move_id)
                                single_analyzer.update_single_candle_analysis_time(index)
                    
                    # 📊 MULTI-CANDLE MONITORING
                    if df_5min is not None:
                        multi_move = multi_analyzer.detect_big_move(index, df_5min)
                        if (multi_move and 
                            multi_analyzer.should_analyze_multi_candle(index)):
                            
                            move_id = f"MULTI_{index}_{multi_move['direction']}_{time_key}"
                            if move_id not in multi_analyzed:
                                analysis = multi_analyzer.analyze_multi_candle_institutional_data(
                                    index, multi_move, df_5min)
                                
                                message = multi_analyzer.format_multi_candle_analysis(analysis)
                                if send_telegram(message):
                                    multi_analyzed.add(move_id)
                                    multi_analyzer.update_multi_candle_analysis_time(index)
                
                except Exception as e:
                    # Log error but continue
                    print(f"Error monitoring {index}: {str(e)}")
                    continue
            
            # Cleanup old entries
            current_hour = datetime.utcnow().hour
            single_analyzed = {id for id in single_analyzed if int(id.split('_')[-1].split(':')[0]) >= current_hour - 2}
            multi_analyzed = {id for id in multi_analyzed if int(id.split('_')[-1].split(':')[0]) >= current_hour - 2}
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Main loop error: {str(e)}")
            time.sleep(60)

# --------- START THE ULTIMATE SYSTEM ---------
if __name__ == "__main__":
    monitor_all_systems()
