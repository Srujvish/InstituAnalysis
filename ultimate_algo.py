# ULTIMATE HISTORICAL INSTITUTIONAL ANALYZER - TODAY'S COMPLETE ANALYSIS
import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import warnings
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# --------- CONFIGURATION ---------
BIG_CANDLE_THRESHOLD = 20  # 20+ points move
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        response = requests.post(url, data=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# --------- DATA FETCHING ---------
def fetch_todays_complete_data(index, interval="1m"):
    try:
        symbol_map = {
            "NIFTY": "^NSEI", 
            "BANKNIFTY": "^NSEBANK", 
            "SENSEX": "^BSESN"
        }
        # Get today's complete data
        today = datetime.now().strftime("%Y-%m-%d")
        df = yf.download(symbol_map[index], start=today, interval=interval, progress=False, timeout=60)
        
        if df.empty:
            print(f"No data found for {index} {interval}")
            return None
            
        print(f"Fetched {len(df)} candles for {index} {interval}")
        return df
        
    except Exception as e:
        print(f"Data fetch error for {index} {interval}: {e}")
        return None

# --------- COMPLETE INSTITUTIONAL ANALYSIS ---------
class CompleteInstitutionalAnalyzer:
    def __init__(self):
        pass
    
    def analyze_big_candle_complete(self, df, big_candle_idx):
        """COMPLETE ANALYSIS of big candle with previous 3 candles"""
        try:
            if len(df) <= big_candle_idx or big_candle_idx < 3:
                return None
            
            # Get current and previous candles
            current_candle = df.iloc[big_candle_idx]
            prev1_candle = df.iloc[big_candle_idx-1] if big_candle_idx-1 >= 0 else None
            prev2_candle = df.iloc[big_candle_idx-2] if big_candle_idx-2 >= 0 else None  
            prev3_candle = df.iloc[big_candle_idx-3] if big_candle_idx-3 >= 0 else None
            
            # Calculate big candle move
            big_candle_move = abs(current_candle['Close'] - current_candle['Open'])
            direction = "GREEN" if current_candle['Close'] > current_candle['Open'] else "RED"
            
            analysis = {
                # Basic candle information
                'timestamp': df.index[big_candle_idx],
                'time_str': df.index[big_candle_idx].strftime('%H:%M:%S'),
                'direction': direction,
                'points_moved': round(float(big_candle_move), 2),
                'candle_range': round(float(current_candle['High'] - current_candle['Low']), 2),
                'volume': int(current_candle['Volume']),
                
                # Previous 3 candles detailed information
                'prev_candles': {}
            }
            
            # Analyze previous 3 candles in detail
            prev_candles_data = []
            for i, candle in enumerate([prev3_candle, prev2_candle, prev1_candle], 1):
                if candle is not None:
                    candle_data = {
                        'time': df.index[big_candle_idx-i].strftime('%H:%M:%S'),
                        'open': round(float(candle['Open']), 2),
                        'high': round(float(candle['High']), 2), 
                        'low': round(float(candle['Low']), 2),
                        'close': round(float(candle['Close']), 2),
                        'points_move': round(abs(float(candle['Close']) - float(candle['Open'])), 2),
                        'direction': "GREEN" if candle['Close'] > candle['Open'] else "RED",
                        'volume': int(candle['Volume']),
                        'range': round(float(candle['High'] - candle['Low']), 2)
                    }
                    prev_candles_data.append(candle_data)
            
            analysis['prev_candles'] = prev_candles_data
            
            # Calculate institutional metrics
            # 1. Volume Analysis
            current_volume = float(current_candle['Volume'])
            prev_volumes = [float(c['Volume']) for c in [prev3_candle, prev2_candle, prev1_candle] if c is not None]
            avg_prev_volume = np.mean(prev_volumes) if prev_volumes else current_volume
            
            analysis['volume_surge_ratio'] = round(current_volume / max(1, avg_prev_volume), 2)
            analysis['volume_change_percent'] = round(((current_volume - avg_prev_volume) / max(1, avg_prev_volume)) * 100, 2)
            
            # 2. Price Momentum
            prev_closes = [float(c['Close']) for c in [prev3_candle, prev2_candle, prev1_candle] if c is not None]
            if len(prev_closes) >= 2:
                price_momentum = (prev_closes[-1] - prev_closes[0]) / prev_closes[0] * 100
                analysis['prev_momentum_percent'] = round(price_momentum, 2)
            else:
                analysis['prev_momentum_percent'] = 0.0
            
            # 3. Volatility Analysis
            current_range_pct = (float(current_candle['High']) - float(current_candle['Low'])) / float(current_candle['Open']) * 100
            prev_ranges = []
            for candle in [prev3_candle, prev2_candle, prev1_candle]:
                if candle is not None:
                    range_pct = (float(candle['High']) - float(candle['Low'])) / float(candle['Open']) * 100
                    prev_ranges.append(range_pct)
            
            avg_prev_range = np.mean(prev_ranges) if prev_ranges else current_range_pct
            analysis['volatility_expansion'] = round(((current_range_pct - avg_prev_range) / max(0.1, avg_prev_range)) * 100, 2)
            
            # 4. Order Flow Pressure
            green_candles = sum(1 for c in [prev3_candle, prev2_candle, prev1_candle] 
                              if c is not None and c['Close'] > c['Open'])
            total_prev_candles = sum(1 for c in [prev3_candle, prev2_candle, prev1_candle] if c is not None)
            analysis['buying_pressure_ratio'] = round(green_candles / max(1, total_prev_candles), 2)
            
            # 5. Institutional Probability Score
            score = 0
            if analysis['volume_surge_ratio'] > 1.5: score += 30
            if analysis['volatility_expansion'] > 50: score += 25
            if abs(analysis['prev_momentum_percent']) > 0.1: score += 20
            if analysis['points_moved'] > 30: score += 25
            
            analysis['institutional_score'] = min(100, score)
            analysis['institutional_confidence'] = "HIGH" if score >= 70 else "MEDIUM" if score >= 50 else "LOW"
            
            return analysis
            
        except Exception as e:
            print(f"Complete analysis error: {e}")
            return None
    
    def find_all_big_candles_today(self, df, threshold=20):
        """Find ALL big candles in today's data"""
        big_candles = []
        try:
            if df is None or len(df) < 4:
                return big_candles
                
            for i in range(3, len(df)):
                try:
                    candle_move = abs(float(df['Close'].iloc[i]) - float(df['Open'].iloc[i]))
                    if candle_move >= threshold:
                        analysis = self.analyze_big_candle_complete(df, i)
                        if analysis:
                            big_candles.append(analysis)
                except Exception as e:
                    continue
                    
            return big_candles
            
        except Exception as e:
            print(f"Error finding big candles: {e}")
            return []

# --------- TELEGRAM MESSAGE FORMATTING ---------
def format_complete_analysis(index, timeframe, analysis):
    """Format COMPLETE analysis with previous 3 candles"""
    
    # Format previous candles information
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

💡 **WHAT HAPPENED**:
{analysis['direction']} move of {analysis['points_moved']} points at {analysis['time_str']}
Volume surged {analysis['volume_surge_ratio']}x with {analysis['volatility_expansion']}% volatility expansion
Previous 3 candles showed {analysis['buying_pressure_ratio']} buying pressure

─────────────────────────────
"""
    return msg

# --------- MAIN ANALYSIS FUNCTION ---------
def analyze_todays_complete_moves():
    """Analyze TODAY'S complete moves for all indices"""
    
    analyzer = CompleteInstitutionalAnalyzer()
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    timeframes = ["1m", "5m"]
    
    startup_msg = f"""
📊 **TODAY'S COMPLETE INSTITUTIONAL ANALYSIS STARTED**
📅 Date: {datetime.now().strftime('%d %b %Y')}
🎯 Target: {BIG_CANDLE_THRESHOLD}+ points moves
📈 Indices: NIFTY, BANKNIFTY, SENSEX
⏰ Timeframes: 1min + 5min
🔍 Analyzing ALL big moves with previous 3 candles context

**ANALYZING TODAY'S COMPLETE DATA...**
"""
    send_telegram(startup_msg)
    print("Starting complete institutional analysis...")
    
    total_analysis_sent = 0
    
    for index in indices:
        index_big_moves = 0
        
        for timeframe in timeframes:
            try:
                print(f"🔍 Analyzing {index} {timeframe}...")
                df = fetch_todays_complete_data(index, timeframe)
                
                if df is not None and len(df) > 10:
                    big_candles = analyzer.find_all_big_candles_today(df, BIG_CANDLE_THRESHOLD)
                    
                    if big_candles:
                        # Send analysis for each big candle
                        for i, analysis in enumerate(big_candles):
                            message = format_complete_analysis(index, timeframe, analysis)
                            if send_telegram(message):
                                print(f"✅ Sent analysis for {index} {timeframe} at {analysis['time_str']}")
                                total_analysis_sent += 1
                                index_big_moves += 1
                            time.sleep(3)  # Avoid rate limiting
                    
                    # Send summary for this timeframe
                    timeframe_summary = f"""
📋 **{index} {timeframe} SUMMARY**
{'✅' if big_candles else '❌'} Found {len(big_candles)} big moves (≥{BIG_CANDLE_THRESHOLD} points)
🕒 Timeframe analysis completed
"""
                    send_telegram(timeframe_summary)
                    
                else:
                    no_data_msg = f"""
⚠️ **{index} {timeframe}**
📊 No data available for analysis
"""
                    send_telegram(no_data_msg)
                
                time.sleep(2)
                
            except Exception as e:
                error_msg = f"""
❌ **ERROR: {index} {timeframe}**
🔧 {str(e)}
"""
                send_telegram(error_msg)
                print(f"Error analyzing {index} {timeframe}: {e}")
                continue
        
        # Send index summary
        if index_big_moves > 0:
            index_summary = f"""
🏁 **{index} COMPLETE ANALYSIS FINISHED**
📈 Total big moves found: {index_big_moves}
✅ Analysis completed successfully
"""
            send_telegram(index_summary)
    
    # Final completion message
    completion_msg = f"""
🎉 **TODAY'S COMPLETE ANALYSIS FINISHED** 🎉

📅 Date: {datetime.now().strftime('%d %b %Y')}
🕒 Completed: {datetime.now().strftime('%H:%M:%S')}
📊 Total Analyses Sent: {total_analysis_sent}
📈 Indices Analyzed: NIFTY, BANKNIFTY, SENSEX
⏰ Timeframes: 1min + 5min

**ALL BIG MOVES ANALYZED WITH COMPLETE INSTITUTIONAL CONTEXT**
"""
    send_telegram(completion_msg)
    print(f"🎯 Analysis completed! Total analyses sent: {total_analysis_sent}")

# --------- RUN ANALYSIS ---------
if __name__ == "__main__":
    print("🚀 Starting Today's Complete Institutional Analysis...")
    analyze_todays_complete_moves()
