# ULTIMATE ERROR-FREE HISTORICAL INSTITUTIONAL ANALYZER
import os
import time
import requests
import pandas as pd
import yfinance as yf
import warnings
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# --------- CONFIGURATION ---------
BIG_CANDLE_THRESHOLD = 20
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        response = requests.post(url, data=payload, timeout=10)
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# --------- SAFE DATA FETCHING ---------
def fetch_todays_data_safe(index, interval="1m"):
    try:
        symbol_map = {
            "NIFTY": "^NSEI", 
            "BANKNIFTY": "^NSEBANK", 
            "SENSEX": "^BSESN"
        }
        
        today = datetime.now().strftime("%Y-%m-%d")
        df = yf.download(symbol_map[index], start=today, interval=interval, progress=False)
        
        if df.empty:
            print(f"No data for {index} {interval}")
            return None
            
        print(f"✅ Fetched {len(df)} candles for {index} {interval}")
        return df
        
    except Exception as e:
        print(f"Data error {index} {interval}: {e}")
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

# --------- ERROR-FREE INSTITUTIONAL ANALYZER ---------
class SafeInstitutionalAnalyzer:
    def __init__(self):
        self.analyzed_candles = set()
    
    def analyze_big_candle_safe(self, df, big_candle_idx):
        """SAFE ANALYSIS - No Series ambiguity errors"""
        try:
            if len(df) <= big_candle_idx or big_candle_idx < 3:
                return None
            
            # SAFELY get candle data using .iloc
            current_row = df.iloc[big_candle_idx]
            prev1_row = df.iloc[big_candle_idx-1]
            prev2_row = df.iloc[big_candle_idx-2]  
            prev3_row = df.iloc[big_candle_idx-3]
            
            # SAFELY convert to scalar values
            current_open = safe_float(current_row['Open'])
            current_high = safe_float(current_row['High'])
            current_low = safe_float(current_row['Low'])
            current_close = safe_float(current_row['Close'])
            current_volume = safe_int(current_row['Volume'])
            
            # Calculate big candle move
            big_candle_move = abs(current_close - current_open)
            direction = "GREEN" if current_close > current_open else "RED"
            
            analysis = {
                'timestamp': df.index[big_candle_idx],
                'time_str': df.index[big_candle_idx].strftime('%H:%M:%S'),
                'direction': direction,
                'points_moved': round(big_candle_move, 2),
                'candle_range': round(current_high - current_low, 2),
                'volume': current_volume,
                'prev_candles': []
            }
            
            # Analyze previous 3 candles SAFELY
            prev_rows = [prev3_row, prev2_row, prev1_row]
            for i, row in enumerate(prev_rows):
                prev_open = safe_float(row['Open'])
                prev_high = safe_float(row['High'])
                prev_low = safe_float(row['Low'])
                prev_close = safe_float(row['Close'])
                prev_volume = safe_int(row['Volume'])
                
                candle_data = {
                    'time': df.index[big_candle_idx-3+i].strftime('%H:%M:%S'),
                    'open': round(prev_open, 2),
                    'high': round(prev_high, 2), 
                    'low': round(prev_low, 2),
                    'close': round(prev_close, 2),
                    'points_move': round(abs(prev_close - prev_open), 2),
                    'direction': "GREEN" if prev_close > prev_open else "RED",
                    'volume': prev_volume,
                    'range': round(prev_high - prev_low, 2)
                }
                analysis['prev_candles'].append(candle_data)
            
            # Calculate institutional metrics SAFELY
            analysis.update(self.calculate_metrics_safe(
                current_open, current_high, current_low, current_close, current_volume,
                prev_rows
            ))
            
            return analysis
            
        except Exception as e:
            print(f"Safe analysis error at index {big_candle_idx}: {e}")
            return None
    
    def calculate_metrics_safe(self, curr_open, curr_high, curr_low, curr_close, curr_volume, prev_rows):
        """SAFE metric calculations without Series objects"""
        try:
            # Extract previous candle data safely
            prev_volumes = []
            prev_closes = []
            prev_ranges_pct = []
            
            for row in prev_rows:
                prev_open = safe_float(row['Open'])
                prev_high = safe_float(row['High'])
                prev_low = safe_float(row['Low'])
                prev_close = safe_float(row['Close'])
                prev_volume = safe_int(row['Volume'])
                
                prev_volumes.append(prev_volume)
                prev_closes.append(prev_close)
                
                # Calculate range percentage
                if prev_open > 0:
                    range_pct = (prev_high - prev_low) / prev_open * 100
                    prev_ranges_pct.append(range_pct)
            
            # Volume Analysis
            avg_prev_volume = np.mean(prev_volumes) if prev_volumes else curr_volume
            volume_surge_ratio = round(curr_volume / max(1, avg_prev_volume), 2)
            volume_change_percent = round(((curr_volume - avg_prev_volume) / max(1, avg_prev_volume)) * 100, 2)
            
            # Price Momentum
            if len(prev_closes) >= 2:
                price_momentum = (prev_closes[-1] - prev_closes[0]) / prev_closes[0] * 100
            else:
                price_momentum = 0
            
            # Volatility Analysis
            current_range_pct = (curr_high - curr_low) / curr_open * 100 if curr_open > 0 else 0
            avg_prev_range = np.mean(prev_ranges_pct) if prev_ranges_pct else current_range_pct
            volatility_expansion = round(((current_range_pct - avg_prev_range) / max(0.1, avg_prev_range)) * 100, 2)
            
            # Order Flow Pressure
            green_candles = 0
            for row in prev_rows:
                prev_open = safe_float(row['Open'])
                prev_close = safe_float(row['Close'])
                if prev_close > prev_open:
                    green_candles += 1
            buying_pressure_ratio = round(green_candles / 3, 2)
            
            # Institutional Score
            score = 0
            if volume_surge_ratio > 2.0: score += 35
            elif volume_surge_ratio > 1.5: score += 25
            
            if volatility_expansion > 75: score += 30
            elif volatility_expansion > 50: score += 20
            
            if abs(price_momentum) > 0.15: score += 20
            elif abs(price_momentum) > 0.08: score += 15
            
            if abs(curr_close - curr_open) > 30: score += 15
            
            institutional_score = min(100, score)
            
            if institutional_score >= 70:
                confidence = "VERY_HIGH"
                activity = "STRONG_INSTITUTIONAL"
            elif institutional_score >= 50:
                confidence = "HIGH"
                activity = "MODERATE_INSTITUTIONAL"
            elif institutional_score >= 30:
                confidence = "MEDIUM"
                activity = "LIGHT_INSTITUTIONAL"
            else:
                confidence = "LOW"
                activity = "RETAIL_DOMINATED"
            
            return {
                'volume_surge_ratio': volume_surge_ratio,
                'volume_change_percent': volume_change_percent,
                'prev_momentum_percent': round(price_momentum, 2),
                'volatility_expansion': volatility_expansion,
                'buying_pressure_ratio': buying_pressure_ratio,
                'institutional_score': institutional_score,
                'institutional_confidence': confidence,
                'institutional_activity': activity
            }
            
        except Exception as e:
            print(f"Metrics error: {e}")
            return {}
    
    def find_all_big_candles_safe(self, df, threshold=20):
        """SAFE method to find all big candles"""
        big_candles = []
        try:
            if df is None or len(df) < 4:
                return big_candles
                
            for i in range(3, len(df)):
                try:
                    # SAFE candle move calculation
                    row = df.iloc[i]
                    open_val = safe_float(row['Open'])
                    close_val = safe_float(row['Close'])
                    candle_move = abs(close_val - open_val)
                    
                    if candle_move >= threshold:
                        analysis = self.analyze_big_candle_safe(df, i)
                        if analysis:
                            big_candles.append(analysis)
                except Exception as e:
                    continue  # Skip problematic candles
                        
            return big_candles
            
        except Exception as e:
            print(f"Find big candles error: {e}")
            return []

# --------- TELEGRAM MESSAGE FORMATTING ---------
def format_analysis_message(index, timeframe, analysis):
    """Format analysis for Telegram"""
    
    # Format previous candles
    prev_candles_text = ""
    for i, candle in enumerate(analysis['prev_candles'], 1):
        prev_candles_text += f"""
    {i}. {candle['time']} - {candle['direction']} {candle['points_move']} points
       O: {candle['open']} | H: {candle['high']} | L: {candle['low']} | C: {candle['close']}
       Range: {candle['range']} pts | Volume: {candle['volume']:,}"""
    
    msg = f"""
🔴🟢 **BIG CANDLE DETECTED - {index} {timeframe}** 🔴🟢

⏰ **TIME**: {analysis['time_str']}
🎯 **DIRECTION**: {analysis['direction']}
📈 **POINTS MOVED**: {analysis['points_moved']} points
📊 **CANDLE RANGE**: {analysis['candle_range']} points  
📦 **VOLUME**: {analysis['volume']:,}

📋 **PREVIOUS 3 CANDLES ANALYSIS**:{prev_candles_text}

📊 **INSTITUTIONAL METRICS**:
• Volume Surge: {analysis['volume_surge_ratio']}x
• Volume Change: {analysis['volume_change_percent']}%
• Previous Momentum: {analysis['prev_momentum_percent']}%
• Volatility Expansion: {analysis['volatility_expansion']}%
• Buying Pressure: {analysis['buying_pressure_ratio']}

🏛️ **INSTITUTIONAL ASSESSMENT**:
• Institutional Score: {analysis['institutional_score']}/100
• Confidence: {analysis['institutional_confidence']}
• Activity Type: {analysis['institutional_activity']}

🎯 **TRADING IMPLICATION**:
Consider {analysis['direction']} positions | {analysis['institutional_confidence']} confidence

─────────────────────────────
"""
    return msg

# --------- MAIN ANALYSIS FUNCTION ---------
def analyze_todays_data_safe():
    """SAFE analysis of today's data"""
    
    analyzer = SafeInstitutionalAnalyzer()
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    timeframes = ["1m", "5m"]
    
    startup_msg = f"""
📊 **SAFE INSTITUTIONAL ANALYSIS STARTED**
📅 Date: {datetime.now().strftime('%d %b %Y')}
🎯 Target: {BIG_CANDLE_THRESHOLD}+ points moves
📈 Analyzing: NIFTY, BANKNIFTY, SENSEX
⏰ Timeframes: 1min + 5min

**PROCESSING TODAY'S DATA SAFELY...**
"""
    send_telegram(startup_msg)
    print("Starting safe institutional analysis...")
    
    total_big_moves = 0
    
    for index in indices:
        index_moves = 0
        
        for timeframe in timeframes:
            try:
                print(f"🔍 Analyzing {index} {timeframe}...")
                
                # Fetch data
                df = fetch_todays_data_safe(index, timeframe)
                
                if df is not None and len(df) > 10:
                    # Find big candles safely
                    big_candles = analyzer.find_all_big_candles_safe(df, BIG_CANDLE_THRESHOLD)
                    
                    if big_candles:
                        # Send each analysis
                        for analysis in big_candles:
                            message = format_analysis_message(index, timeframe, analysis)
                            if send_telegram(message):
                                print(f"✅ Sent {index} {timeframe} at {analysis['time_str']}")
                                total_big_moves += 1
                                index_moves += 1
                            time.sleep(3)
                    
                    # Send summary
                    summary_msg = f"""
📋 **{index} {timeframe} SUMMARY**
{'✅' if big_candles else '❌'} Found {len(big_candles)} big moves (≥{BIG_CANDLE_THRESHOLD} points)
"""
                    send_telegram(summary_msg)
                    
                else:
                    no_data_msg = f"""
⚠️ **{index} {timeframe}**
📊 No data available
"""
                    send_telegram(no_data_msg)
                
                time.sleep(2)
                
            except Exception as e:
                error_msg = f"""
❌ **ERROR: {index} {timeframe}**
🔧 {str(e)}
"""
                send_telegram(error_msg)
                continue
        
        # Index completion
        if index_moves > 0:
            index_msg = f"""
🏁 **{index} COMPLETED**
📈 Found {index_moves} big moves
"""
            send_telegram(index_msg)
    
    # Final message
    completion_msg = f"""
🎉 **ANALYSIS COMPLETED** 🎉

📅 Date: {datetime.now().strftime('%d %b %Y')}
🕒 Finished: {datetime.now().strftime('%H:%M:%S')}
📊 Total Big Moves: {total_big_moves}
✅ All indices processed

**READY FOR TOMORROW'S MARKET**
"""
    send_telegram(completion_msg)
    print(f"✅ Safe analysis completed! Found {total_big_moves} big moves")

# --------- RUN SAFE ANALYSIS ---------
if __name__ == "__main__":
    print("🚀 Starting Safe Institutional Analysis...")
    analyze_todays_data_safe()
