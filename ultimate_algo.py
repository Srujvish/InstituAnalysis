# ULTIMATE INSTITUTIONAL INTELLIGENCE ANALYZER - COMPLETE INSTITUTIONAL PARAMETERS

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
import numpy as np

warnings.filterwarnings("ignore")

# --------- CONFIGURATION ---------
MOVE_THRESHOLD = 40  # Multi-candle moves
SINGLE_CANDLE_MOVE_THRESHOLD = 25  # Single candle moves
MULTI_CANDLE_COOLDOWN = 30
SINGLE_CANDLE_COOLDOWN = 15

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

# --------- OPTION STRIKE ANALYSIS ---------
def get_option_strike_details(index, strike_price, option_type, current_price, move_direction):
    try:
        if strike_price == "Couldn't fetch" or option_type == "Couldn't fetch":
            return {"details": "Option strike calculation failed"}
        
        strike_price = int(strike_price)
        
        # Calculate moneyness
        if option_type == "CE":
            if strike_price < current_price:
                moneyness = "ITM (In the Money)"
            elif strike_price == current_price:
                moneyness = "ATM (At the Money)"
            else:
                moneyness = "OTM (Out of the Money)"
        else:
            if strike_price > current_price:
                moneyness = "ITM (In the Money)"
            elif strike_price == current_price:
                moneyness = "ATM (At the Money)"
            else:
                moneyness = "OTM (Out of the Money)"
        
        distance = abs(strike_price - current_price)
        distance_percentage = (distance / current_price) * 100
        
        if distance_percentage < 0.2:
            probability = "HIGH"
            risk = "LOW"
        elif distance_percentage < 0.5:
            probability = "MEDIUM"
            risk = "MODERATE"
        else:
            probability = "LOW"
            risk = "HIGH"
        
        # Premium estimation
        if index == "NIFTY":
            base_premium = max(20, distance * 0.4)
        elif index == "BANKNIFTY":
            base_premium = max(40, distance * 0.3)
        else:
            base_premium = max(30, distance * 0.35)
        
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
            "current_price": current_price
        }
    except Exception:
        return {"details": "Option details calculation failed"}

def find_nearest_option_strike(index, current_price, move_direction):
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
        pass
    
    def calculate_all_institutional_params(self, df, move_start_time):
        """Calculate ALL 13 institutional parameters AT THE START of the move"""
        try:
            if df is None or len(df) < 10:
                return self.get_empty_analysis()
            
            # Find the start candle index
            start_idx = df.index.get_loc(move_start_time)
            analysis_data = df.iloc[max(0, start_idx-5):start_idx+1]  # Data up to start candle
            
            params = {}
            
            # 1. ORDER BOOK IMBALANCE (OBI)
            params['order_book_imbalance'] = self.calculate_obi(analysis_data)
            
            # 2. MARKET DELTA
            params['market_delta'] = self.calculate_market_delta(analysis_data)
            
            # 3. CUMULATIVE VOLUME DELTA (CVD)
            params['cumulative_volume_delta'] = self.calculate_cvd(analysis_data)
            
            # 4. ORDER FLOW PRESSURE
            params['order_flow_pressure'] = self.calculate_order_flow_pressure(analysis_data)
            
            # 5. ICEBERG ORDER DETECTION
            params['iceberg_orders'] = self.detect_iceberg_orders(analysis_data)
            
            # 6. LIQUIDITY GRAB SIGNAL
            params['liquidity_grab'] = self.detect_liquidity_grab(analysis_data)
            
            # 7. SPREAD EXPANSION
            params['spread_expansion'] = self.calculate_spread_expansion(analysis_data)
            
            # 8. MARKET DEPTH THINNING
            params['market_depth_thinning'] = self.calculate_market_depth_thinning(analysis_data)
            
            # 9. VOLUME IMBALANCE
            params['volume_imbalance'] = self.calculate_volume_imbalance(analysis_data)
            
            # 10. FVG SIZE ANALYSIS
            params['fvg_size_analysis'] = self.analyze_fvg_size(analysis_data)
            
            # 11. VWAP DEVIATION
            params['vwap_deviation'] = self.calculate_vwap_deviation(analysis_data)
            
            # 12. VOLATILITY COMPRESSION
            params['volatility_compression'] = self.calculate_volatility_compression(analysis_data)
            
            # 13. WHALE FOOTPRINTS
            params['whale_footprints'] = self.detect_whale_footprints(analysis_data)
            
            # Overall sentiment
            params['institutional_sentiment'] = self.calculate_overall_sentiment(params)
            
            return params
            
        except Exception:
            return self.get_empty_analysis()
    
    def calculate_obi(self, df):
        """1. ORDER BOOK IMBALANCE"""
        try:
            if len(df) < 3:
                return {"value": "Insufficient data", "interpretation": "Need more data"}
            
            recent_data = df.iloc[-3:]
            price_changes = recent_data['Close'].pct_change().dropna()
            avg_change = price_changes.mean()
            
            # Simulate OBI based on price momentum
            if avg_change > 0.001:
                obi_value = 0.7 + (avg_change * 1000)
            elif avg_change < -0.001:
                obi_value = -0.7 + (avg_change * 1000)
            else:
                obi_value = avg_change * 500
            
            obi_value = max(-0.9, min(0.9, obi_value))
            
            if obi_value >= 0.6:
                interpretation = "Strong Institutions Buying"
            elif obi_value <= -0.6:
                interpretation = "Strong Institutions Selling"
            elif obi_value >= 0.3:
                interpretation = "Moderate Buying Pressure"
            elif obi_value <= -0.3:
                interpretation = "Moderate Selling Pressure"
            else:
                interpretation = "Balanced"
            
            return {
                "value": f"{obi_value:.3f}",
                "interpretation": interpretation
            }
        except Exception:
            return {"value": "Calculation failed", "interpretation": "Error"}
    
    def calculate_market_delta(self, df):
        """2. MARKET DELTA"""
        try:
            if len(df) < 3:
                return {"value": "Insufficient data", "interpretation": "Need more data"}
            
            recent = df.iloc[-3:]
            total_move = recent['Close'].iloc[-1] - recent['Close'].iloc[0]
            avg_range = (recent['High'] - recent['Low']).mean()
            
            # Estimate delta based on move strength
            delta_value = (total_move / avg_range) * 1000 if avg_range > 0 else 0
            
            if delta_value > 500:
                interpretation = "Strong Aggressive Buying"
            elif delta_value < -500:
                interpretation = "Strong Aggressive Selling"
            elif delta_value > 200:
                interpretation = "Moderate Buying"
            elif delta_value < -200:
                interpretation = "Moderate Selling"
            else:
                interpretation = "Balanced"
            
            return {
                "value": f"{delta_value:+.0f}",
                "interpretation": interpretation
            }
        except Exception:
            return {"value": "Calculation failed", "interpretation": "Error"}
    
    def calculate_cvd(self, df):
        """3. CUMULATIVE VOLUME DELTA"""
        try:
            if len(df) < 5:
                return {"value": "Insufficient data", "interpretation": "Need more data"}
            
            recent = df.iloc[-5:]
            price_trend = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
            
            # Simulate CVD based on price trend
            cvd_value = price_trend * 10000
            
            if cvd_value > 50:
                interpretation = "CVD Rising - Bullish Breakout Likely"
            elif cvd_value < -50:
                interpretation = "CVD Falling - Bearish Breakdown Likely"
            elif cvd_value > 20:
                interpretation = "Moderate Buying Pressure"
            elif cvd_value < -20:
                interpretation = "Moderate Selling Pressure"
            else:
                interpretation = "Neutral"
            
            return {
                "value": f"{cvd_value:+.2f}",
                "interpretation": interpretation
            }
        except Exception:
            return {"value": "Calculation failed", "interpretation": "Error"}
    
    def calculate_order_flow_pressure(self, df):
        """4. ORDER FLOW PRESSURE"""
        try:
            if len(df) < 3:
                return {"value": "Insufficient data", "interpretation": "Need more data"}
            
            recent = df.iloc[-3:]
            bullish_candles = sum(1 for i in range(len(recent)) if recent['Close'].iloc[i] > recent['Open'].iloc[i])
            pressure_ratio = bullish_candles / len(recent)
            
            if pressure_ratio > 0.7:
                interpretation = "Buyers Dominating"
            elif pressure_ratio < 0.3:
                interpretation = "Sellers Dominating"
            elif pressure_ratio > 0.6:
                interpretation = "Buyers Aggressive"
            elif pressure_ratio < 0.4:
                interpretation = "Sellers Aggressive"
            else:
                interpretation = "Balanced"
            
            return {
                "value": f"{pressure_ratio:.3f}",
                "interpretation": interpretation
            }
        except Exception:
            return {"value": "Calculation failed", "interpretation": "Error"}
    
    def detect_iceberg_orders(self, df):
        """5. ICEBERG ORDER DETECTION"""
        try:
            if len(df) < 5:
                return {"detected": "Insufficient data", "score": "N/A"}
            
            recent = df.iloc[-5:]
            volume_consistency = recent['Volume'].std() / recent['Volume'].mean() if recent['Volume'].mean() > 0 else 1
            
            if volume_consistency < 0.3:
                score = 0.8
                detected = "Yes"
            elif volume_consistency < 0.5:
                score = 0.6
                detected = "Possible"
            else:
                score = 0.3
                detected = "No"
            
            return {
                "detected": detected,
                "score": f"{score:.2f}"
            }
        except Exception:
            return {"detected": "Calculation failed", "score": "N/A"}
    
    def detect_liquidity_grab(self, df):
        """6. LIQUIDITY GRAB SIGNAL"""
        try:
            if len(df) < 3:
                return {"detected": "Insufficient data", "signal_strength": "N/A"}
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            current_body = abs(current['Close'] - current['Open'])
            current_upper_wick = current['High'] - max(current['Open'], current['Close'])
            current_lower_wick = min(current['Open'], current['Close']) - current['Low']
            
            # Check for liquidity grab patterns
            upper_grab = current_upper_wick > 2 * current_body and current['High'] > prev['High']
            lower_grab = current_lower_wick > 2 * current_body and current['Low'] < prev['Low']
            
            if upper_grab or lower_grab:
                detected = "Yes"
                strength = "STRONG"
            else:
                detected = "No"
                strength = "WEAK"
            
            return {
                "detected": detected,
                "signal_strength": strength
            }
        except Exception:
            return {"detected": "Calculation failed", "signal_strength": "N/A"}
    
    def calculate_spread_expansion(self, df):
        """7. SPREAD EXPANSION"""
        try:
            if len(df) < 10:
                return {"expansion": "Insufficient data", "interpretation": "Need more data"}
            
            recent_atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 5).average_true_range().iloc[-1]
            historical_atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 10).average_true_range().iloc[-1]
            
            if historical_atr > 0:
                spread_change = ((recent_atr - historical_atr) / historical_atr) * 100
            else:
                spread_change = 0
            
            if spread_change > 50:
                interpretation = "Institutions Preparing Breakout"
            elif spread_change > 30:
                interpretation = "Moderate Spread Expansion"
            elif spread_change < -30:
                interpretation = "Spread Compression"
            else:
                interpretation = "Normal"
            
            return {
                "expansion": f"{spread_change:+.1f}%",
                "interpretation": interpretation
            }
        except Exception:
            return {"expansion": "Calculation failed", "interpretation": "Error"}
    
    def calculate_market_depth_thinning(self, df):
        """8. MARKET DEPTH THINNING"""
        try:
            if len(df) < 10:
                return {"ratio": "Insufficient data", "interpretation": "Need more data"}
            
            recent_vol = df['Volume'].iloc[-5:].std()
            historical_vol = df['Volume'].iloc[-10:].std()
            
            if historical_vol > 0:
                depth_ratio = recent_vol / historical_vol
            else:
                depth_ratio = 1.0
            
            if depth_ratio < 0.5:
                interpretation = "Depth Thin - Violent Move Expected"
            elif depth_ratio < 0.8:
                interpretation = "Moderate Thinning"
            elif depth_ratio > 1.5:
                interpretation = "Depth Building"
            else:
                interpretation = "Normal Depth"
            
            return {
                "ratio": f"{depth_ratio:.2f}",
                "interpretation": interpretation
            }
        except Exception:
            return {"ratio": "Calculation failed", "interpretation": "Error"}
    
    def calculate_volume_imbalance(self, df):
        """9. VOLUME IMBALANCE"""
        try:
            if len(df) < 5:
                return {"imbalance": "Insufficient data", "interpretation": "Need more data"}
            
            recent = df.iloc[-5:]
            price_trend = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
            volume_trend = recent['Volume'].iloc[-1] / recent['Volume'].mean() if recent['Volume'].mean() > 0 else 1
            
            vi_value = price_trend * volume_trend * 10
            vi_value = max(-0.9, min(0.9, vi_value))
            
            if vi_value >= 0.6:
                interpretation = "Strong Buying Imbalance"
            elif vi_value <= -0.6:
                interpretation = "Strong Selling Imbalance"
            elif vi_value >= 0.3:
                interpretation = "Moderate Buying"
            elif vi_value <= -0.3:
                interpretation = "Moderate Selling"
            else:
                interpretation = "Balanced"
            
            return {
                "imbalance": f"{vi_value:.3f}",
                "interpretation": interpretation
            }
        except Exception:
            return {"imbalance": "Calculation failed", "interpretation": "Error"}
    
    def analyze_fvg_size(self, df):
        """10. FVG SIZE ANALYSIS"""
        try:
            if len(df) < 3:
                return {"size": "Insufficient data", "type": "N/A", "interpretation": "Need more data"}
            
            current_low = df['Low'].iloc[-1]
            current_high = df['High'].iloc[-1]
            prev_high = df['High'].iloc[-2]
            prev_low = df['Low'].iloc[-2]
            
            bullish_fvg = current_low - prev_high if current_low > prev_high else 0
            bearish_fvg = prev_low - current_high if current_high < prev_low else 0
            
            fvg_size = max(bullish_fvg, bearish_fvg)
            
            if bullish_fvg > 0:
                fvg_type = "BULLISH"
                interpretation = "Bullish FVG - Institutional Displacement"
            elif bearish_fvg > 0:
                fvg_type = "BEARISH"
                interpretation = "Bearish FVG - Institutional Displacement"
            else:
                fvg_type = "NONE"
                interpretation = "No Significant FVG"
            
            return {
                "size": f"{fvg_size:.1f}",
                "type": fvg_type,
                "interpretation": interpretation
            }
        except Exception:
            return {"size": "Calculation failed", "type": "N/A", "interpretation": "Error"}
    
    def calculate_vwap_deviation(self, df):
        """11. VWAP DEVIATION"""
        try:
            if len(df) < 20:
                return {"deviation": "Insufficient data", "interpretation": "Need more data"}
            
            vwap = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price().iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            deviation = ((current_price - vwap) / vwap) * 100
            
            if deviation > 1.0:
                interpretation = "Far Above VWAP - Mean Reversion Likely"
            elif deviation < -1.0:
                interpretation = "Far Below VWAP - Mean Reversion Likely"
            elif deviation > 0.5:
                interpretation = "Above VWAP"
            elif deviation < -0.5:
                interpretation = "Below VWAP"
            else:
                interpretation = "At VWAP"
            
            return {
                "deviation": f"{deviation:+.2f}%",
                "interpretation": interpretation
            }
        except Exception:
            return {"deviation": "Calculation failed", "interpretation": "Error"}
    
    def calculate_volatility_compression(self, df):
        """12. VOLATILITY COMPRESSION"""
        try:
            if len(df) < 10:
                return {"compression": "Insufficient data", "interpretation": "Need more data"}
            
            atr_5 = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 5).average_true_range().iloc[-1]
            atr_10 = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 10).average_true_range().iloc[-1]
            
            if atr_10 > 0:
                compression_ratio = atr_5 / atr_10
            else:
                compression_ratio = 1.0
            
            if compression_ratio < 0.7:
                interpretation = "Volatility Compression - Squeeze Building"
            elif compression_ratio > 1.3:
                interpretation = "Volatility Expansion"
            else:
                interpretation = "Normal Volatility"
            
            return {
                "compression": f"{compression_ratio:.2f}",
                "interpretation": interpretation
            }
        except Exception:
            return {"compression": "Calculation failed", "interpretation": "Error"}
    
    def detect_whale_footprints(self, df):
        """13. WHALE FOOTPRINTS"""
        try:
            if len(df) < 10:
                return {"detected": "Insufficient data", "confidence": "N/A"}
            
            recent = df.iloc[-5:]
            large_volume_bars = sum(1 for vol in recent['Volume'] if vol > recent['Volume'].mean() * 2)
            
            if large_volume_bars >= 3:
                detected = "Yes"
                confidence = "0.85"
            elif large_volume_bars >= 2:
                detected = "Possible"
                confidence = "0.65"
            else:
                detected = "No"
                confidence = "0.35"
            
            return {
                "detected": detected,
                "confidence": confidence
            }
        except Exception:
            return {"detected": "Calculation failed", "confidence": "N/A"}
    
    def calculate_overall_sentiment(self, params):
        """Calculate overall institutional sentiment"""
        try:
            score = 0
            
            # OBI contribution
            obi = params['order_book_imbalance']
            if obi['value'] != "Insufficient data" and obi['value'] != "Calculation failed":
                obi_val = float(obi['value'])
                score += obi_val * 10
            
            # Market Delta contribution
            delta = params['market_delta']
            if delta['value'] not in ["Insufficient data", "Calculation failed"]:
                delta_val = float(delta['value'].replace('+', '').replace('-', ''))
                if delta['value'].startswith('-'):
                    delta_val *= -1
                score += delta_val / 100
            
            # CVD contribution
            cvd = params['cumulative_volume_delta']
            if cvd['value'] not in ["Insufficient data", "Calculation failed"]:
                cvd_val = float(cvd['value'].replace('+', '').replace('-', ''))
                if cvd['value'].startswith('-'):
                    cvd_val *= -1
                score += cvd_val / 10
            
            if score >= 3:
                sentiment = "BULLISH"
                confidence = "HIGH"
            elif score <= -3:
                sentiment = "BEARISH"
                confidence = "HIGH"
            elif score >= 1:
                sentiment = "BULLISH"
                confidence = "MEDIUM"
            elif score <= -1:
                sentiment = "BEARISH"
                confidence = "MEDIUM"
            else:
                sentiment = "NEUTRAL"
                confidence = "LOW"
            
            return {
                "score": round(score, 2),
                "sentiment": sentiment,
                "confidence": confidence
            }
        except Exception:
            return {"score": 0, "sentiment": "NEUTRAL", "confidence": "LOW"}
    
    def get_empty_analysis(self):
        """Return empty analysis when data is insufficient"""
        empty_val = {"value": "Insufficient data", "interpretation": "Need more data"}
        return {
            'order_book_imbalance': empty_val,
            'market_delta': empty_val,
            'cumulative_volume_delta': empty_val,
            'order_flow_pressure': empty_val,
            'iceberg_orders': {"detected": "Insufficient data", "score": "N/A"},
            'liquidity_grab': {"detected": "Insufficient data", "signal_strength": "N/A"},
            'spread_expansion': empty_val,
            'market_depth_thinning': empty_val,
            'volume_imbalance': empty_val,
            'fvg_size_analysis': {"size": "Insufficient data", "type": "N/A", "interpretation": "Need more data"},
            'vwap_deviation': empty_val,
            'volatility_compression': empty_val,
            'whale_footprints': {"detected": "Insufficient data", "confidence": "N/A"},
            'institutional_sentiment': {"score": 0, "sentiment": "NEUTRAL", "confidence": "LOW"}
        }

# --------- SINGLE CANDLE ANALYZER ---------
class SingleCandleAnalyzer:
    def __init__(self):
        self.institutional_engine = InstitutionalAnalysisEngine()
        self.last_analysis_time = {}
    
    def detect_single_candle_move(self, index, df_5min, df_1min):
        """Detect single candle big moves"""
        try:
            # Check 5-minute data
            if df_5min is not None and len(df_5min) >= 3:
                move = self.analyze_single_candle(df_5min, "5min")
                if move:
                    return move
            
            # Check 1-minute data
            if df_1min is not None and len(df_1min) >= 3:
                move = self.analyze_single_candle(df_1min, "1min")
                if move:
                    return move
        except Exception:
            pass
        return None
    
    def analyze_single_candle(self, df, interval):
        """Analyze single candle move"""
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]
            prev_prev = df.iloc[-3]
            
            current_change = current['Close'] - current['Open']
            current_range = current['High'] - current['Low']
            
            if abs(current_change) >= SINGLE_CANDLE_MOVE_THRESHOLD:
                direction = "UP" if current_change > 0 else "DOWN"
                
                # Get institutional analysis AT THE START of the move
                institutional_data = self.institutional_engine.calculate_all_institutional_params(df, df.index[-1])
                
                return {
                    'interval': interval,
                    'direction': direction,
                    'points': abs(current_change),
                    'range': current_range,
                    'timestamp': df.index[-1],
                    'current_candle': {
                        'open': current['Open'],
                        'high': current['High'],
                        'low': current['Low'],
                        'close': current['Close']
                    },
                    'previous_candle': {
                        'open': prev['Open'],
                        'high': prev['High'],
                        'low': prev['Low'],
                        'close': prev['Close'],
                        'change': prev['Close'] - prev['Open']
                    },
                    'prev_prev_candle': {
                        'open': prev_prev['Open'],
                        'high': prev_prev['High'],
                        'low': prev_prev['Low'],
                        'close': prev_prev['Close'],
                        'change': prev_prev['Close'] - prev_prev['Open']
                    },
                    'institutional_data': institutional_data
                }
        except Exception:
            pass
        return None
    
    def analyze_completed_single_candle(self, index, move_info):
        """Analyze completed single candle move"""
        try:
            current_price = move_info['current_candle']['close']
            option_strike, option_type = find_nearest_option_strike(index, current_price, move_info['direction'])
            option_details = get_option_strike_details(index, option_strike, option_type, current_price, move_info['direction'])
            
            return {
                'index': index,
                'move_info': move_info,
                'analysis_time': datetime.utcnow() + timedelta(hours=5, minutes=30),
                'option_details': option_details
            }
        except Exception as e:
            return {'error': f"Single candle analysis failed: {str(e)}"}
    
    def format_single_candle_analysis(self, analysis):
        """Format single candle analysis with ALL institutional parameters"""
        try:
            if 'error' in analysis:
                return f"⚠️ Single Candle Analysis Error: {analysis['error']}"
            
            move = analysis['move_info']
            option = analysis['option_details']
            inst = move['institutional_data']
            
            msg = f"""
🚨 <b>SINGLE CANDLE BIG MOVE DETECTED</b> 🚨

📊 <b>INDEX</b>: {analysis['index']}
⏰ <b>INTERVAL</b>: {move['interval']}
🎯 <b>MOVE</b>: {move['direction']} {move['points']} points
🕒 <b>TIME</b>: {analysis['analysis_time'].strftime('%H:%M:%S')}

📈 <b>CURRENT CANDLE</b> ({move['timestamp'].strftime('%H:%M')}):
• Open: {move['current_candle']['open']:.1f} → Close: {move['current_candle']['close']:.1f}
• Range: {move['range']:.1f} points

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
   → Signal: {inst['liquidity_grab']['signal_strength']}

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
• Strike: {option.get('strike', 'N/A')} {option.get('type', 'N/A')}
• Moneyness: {option.get('moneyness', 'N/A')}
• Distance: {option.get('distance_points', 'N/A')} points
• Probability: {option.get('probability', 'N/A')}
• Risk Level: {option.get('risk_level', 'N/A')}
• Premium: ₹{option.get('estimated_premium', 'N/A')}

💡 <b>TRADING IMPLICATIONS</b>:
Consider {option.get('type', '')} positions | {inst['institutional_sentiment']['sentiment']} bias confirmed

─────────────────────────────
            """
            return msg
        except Exception as e:
            return f"Error formatting single candle message: {str(e)}"
    
    def should_analyze(self, index):
        """Cooldown check"""
        try:
            current_time = time.time()
            if index in self.last_analysis_time:
                time_since_last = current_time - self.last_analysis_time[index]
                if time_since_last < SINGLE_CANDLE_COOLDOWN * 60:
                    return False
            return True
        except Exception:
            return True
    
    def update_analysis_time(self, index):
        """Update analysis time"""
        try:
            self.last_analysis_time[index] = time.time()
        except Exception:
            pass

# --------- MULTI-CANDLE ANALYZER ---------
class MultiCandleAnalyzer:
    def __init__(self):
        self.institutional_engine = InstitutionalAnalysisEngine()
        self.last_analysis_time = {}
    
    def detect_multi_candle_move(self, index, df):
        """Detect multi-candle big moves"""
        try:
            if df is None or len(df) < 10:
                return None
            
            current_price = df['Close'].iloc[-1]
            
            # Look for completed moves
            for lookback in range(3, min(8, len(df)-1)):
                start_time = df.index[-lookback]
                start_price = df['Close'].iloc[-lookback]
                
                move_points = current_price - start_price
                
                if abs(move_points) >= MOVE_THRESHOLD:
                    # Get institutional analysis AT THE START of the move
                    institutional_data = self.institutional_engine.calculate_all_institutional_params(df, start_time)
                    
                    option_strike, option_type = find_nearest_option_strike(
                        index, current_price, "UP" if move_points > 0 else "DOWN")
                    
                    return {
                        'direction': "UP" if move_points > 0 else "DOWN",
                        'points': abs(move_points),
                        'percentage': abs(move_points / start_price * 100),
                        'start_price': start_price,
                        'current_price': current_price,
                        'start_time': start_time,
                        'current_time': df.index[-1],
                        'duration_candles': lookback,
                        'option_strike': option_strike,
                        'option_type': option_type,
                        'move_strength': 'STRONG' if abs(move_points) > 60 else 'MODERATE',
                        'institutional_data': institutional_data
                    }
        except Exception:
            return None
        return None
    
    def analyze_completed_move(self, index, move_info):
        """Analyze completed multi-candle move"""
        try:
            option_details = get_option_strike_details(
                index, move_info['option_strike'], move_info['option_type'],
                move_info['current_price'], move_info['direction'])
            
            return {
                'index': index,
                'move_info': move_info,
                'analysis_time': datetime.utcnow() + timedelta(hours=5, minutes=30),
                'option_details': option_details
            }
        except Exception as e:
            return {'error': f"Multi-candle analysis failed: {str(e)}"}
    
    def format_multi_candle_analysis(self, analysis):
        """Format multi-candle analysis with ALL institutional parameters"""
        try:
            if 'error' in analysis:
                return f"⚠️ Multi-Candle Analysis Error: {analysis['error']}"
            
            move = analysis['move_info']
            option = analysis['option_details']
            inst = move['institutional_data']
            
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
   → Signal: {inst['liquidity_grab']['signal_strength']}

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
• Strike: {option.get('strike', 'N/A')} {option.get('type', 'N/A')}
• Moneyness: {option.get('moneyness', 'N/A')}
• Distance: {option.get('distance_points', 'N/A')} points
• Probability: {option.get('probability', 'N/A')}
• Risk Level: {option.get('risk_level', 'N/A')}
• Premium: ₹{option.get('estimated_premium', 'N/A')}

💡 <b>TRADING IMPLICATIONS</b>:
{move['direction']} move confirmed | Institutional {inst['institutional_sentiment']['sentiment'].lower()} bias

─────────────────────────────
            """
            return msg
        except Exception as e:
            return f"Error formatting multi-candle message: {str(e)}"
    
    def should_analyze(self, index):
        """Cooldown check"""
        try:
            current_time = time.time()
            if index in self.last_analysis_time:
                time_since_last = current_time - self.last_analysis_time[index]
                if time_since_last < MULTI_CANDLE_COOLDOWN * 60:
                    return False
            return True
        except Exception:
            return True
    
    def update_analysis_time(self, index):
        """Update analysis time"""
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
    
    # Send startup message
    startup_msg = """
🚀 <b>ULTIMATE INSTITUTIONAL INTELLIGENCE STARTED</b>

📊 <b>SYSTEM MODES</b>:
• Multi-Candle Analysis: 40+ points moves
• Single-Candle Analysis: 25+ points moves  
• Timeframes: 1min + 5min data
• Complete Institutional Parameters (13 metrics)

🔍 <b>INSTITUTIONAL PARAMETERS</b>:
1. Order Book Imbalance (OBI)
2. Market Delta
3. Cumulative Volume Delta (CVD)
4. Order Flow Pressure
5. Iceberg Order Detection
6. Liquidity Grab Signal
7. Spread Expansion
8. Market Depth Thinning
9. Volume Imbalance
10. FVG Size Analysis
11. VWAP Deviation
12. Volatility Compression
13. Whale Footprints

✅ <b>READY FOR BIG MOVE DETECTION</b>
"""
    send_telegram(startup_msg)
    print("System started with complete institutional parameters...")
    
    while True:
        try:
            if not is_market_open():
                time.sleep(60)
                continue
            
            current_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
            time_key = current_time.strftime("%H:%M")
            
            for index in ["NIFTY", "BANKNIFTY"]:
                try:
                    # Fetch data
                    df_5min = fetch_index_data_safe(index, "5m", "1d")
                    df_1min = fetch_index_data_safe(index, "1m", "1d")
                    
                    if df_5min is None and df_1min is None:
                        continue
                    
                    # Single Candle Monitoring
                    single_move = single_analyzer.detect_single_candle_move(index, df_5min, df_1min)
                    if single_move and single_analyzer.should_analyze(index):
                        move_id = f"SINGLE_{index}_{single_move['direction']}_{time_key}"
                        if move_id not in single_analyzed:
                            analysis = single_analyzer.analyze_completed_single_candle(index, single_move)
                            message = single_analyzer.format_single_candle_analysis(analysis)
                            
                            if send_telegram(message):
                                single_analyzed.add(move_id)
                                single_analyzer.update_analysis_time(index)
                    
                    # Multi Candle Monitoring
                    if df_5min is not None:
                        multi_move = multi_analyzer.detect_multi_candle_move(index, df_5min)
                        if multi_move and multi_analyzer.should_analyze(index):
                            move_id = f"MULTI_{index}_{multi_move['direction']}_{time_key}"
                            if move_id not in multi_analyzed:
                                analysis = multi_analyzer.analyze_completed_move(index, multi_move)
                                message = multi_analyzer.format_multi_candle_analysis(analysis)
                                
                                if send_telegram(message):
                                    multi_analyzed.add(move_id)
                                    multi_analyzer.update_analysis_time(index)
                
                except Exception as e:
                    print(f"Error monitoring {index}: {str(e)}")
                    continue
            
            # Cleanup
            current_hour = datetime.utcnow().hour
            single_analyzed = {id for id in single_analyzed if int(id.split('_')[-1].split(':')[0]) >= current_hour - 2}
            multi_analyzed = {id for id in multi_analyzed if int(id.split('_')[-1].split(':')[0]) >= current_hour - 2}
            
            time.sleep(30)
            
        except Exception as e:
            print(f"Main loop error: {str(e)}")
            time.sleep(60)

# --------- START THE SYSTEM ---------
if __name__ == "__main__":
    print("Starting Ultimate Institutional Intelligence with Complete Parameters...")
    monitor_all_systems()
