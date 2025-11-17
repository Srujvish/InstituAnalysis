# ULTIMATE NUMERICAL INSTITUTIONAL ANALYZER - 17 NOV 2025
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
        requests.post(url, data=payload, timeout=10)
        return True
    except:
        return False

# --------- DATA FETCHING ---------
def fetch_todays_data(index, interval="1m"):
    try:
        symbol_map = {
            "NIFTY": "^NSEI", 
            "BANKNIFTY": "^NSEBANK", 
            "SENSEX": "^BSESN"
        }
        today = datetime.now().strftime("%Y-%m-%d")
        df = yf.download(symbol_map[index], start=today, interval=interval, progress=False)
        return df if not df.empty else None
    except Exception:
        return None

# --------- NUMERICAL INSTITUTIONAL ANALYSIS ---------
class NumericalInstitutionalAnalyzer:
    def __init__(self):
        pass
    
    def analyze_big_candle_with_context(self, df, big_candle_idx):
        """Analyze BIG candle with 3 previous candles - ONLY NUMERICAL VALUES"""
        try:
            if len(df) < big_candle_idx + 1 or big_candle_idx < 3:
                return None
            
            # Get candles data
            big_candle = df.iloc[big_candle_idx]
            prev3 = df.iloc[big_candle_idx-3:big_candle_idx]
            prev2 = df.iloc[big_candle_idx-2]
            prev1 = df.iloc[big_candle_idx-1]
            
            # Calculate EXACT NUMERICAL VALUES
            analysis = {}
            
            # 1. PRICE MOVEMENT NUMBERS
            big_candle_move = abs(big_candle['Close'] - big_candle['Open'])
            analysis['big_candle_points'] = round(big_candle_move, 2)
            analysis['big_candle_direction'] = "UP" if big_candle['Close'] > big_candle['Open'] else "DOWN"
            analysis['big_candle_range'] = round(big_candle['High'] - big_candle['Low'], 2)
            
            # Previous 3 candles cumulative move
            prev3_move = abs(prev3['Close'].iloc[-1] - prev3['Open'].iloc[0])
            analysis['prev3_cumulative_points'] = round(prev3_move, 2)
            
            # Price acceleration ratio
            analysis['price_acceleration_ratio'] = round(big_candle_move / max(0.1, prev3_move), 2)
            
            # 2. VOLUME ANALYSIS NUMBERS
            big_volume = big_candle['Volume']
            prev3_avg_volume = prev3['Volume'].mean()
            analysis['big_candle_volume'] = int(big_volume)
            analysis['volume_surge_ratio'] = round(big_volume / max(1, prev3_avg_volume), 2)
            analysis['volume_change_percent'] = round(((big_volume - prev3_avg_volume) / max(1, prev3_avg_volume)) * 100, 2)
            
            # 3. VOLATILITY NUMBERS
            big_candle_range_pct = (big_candle['High'] - big_candle['Low']) / big_candle['Open'] * 100
            prev3_avg_range_pct = ((prev3['High'] - prev3['Low']) / prev3['Open'] * 100).mean()
            analysis['range_expansion_percent'] = round(((big_candle_range_pct - prev3_avg_range_pct) / max(0.1, prev3_avg_range_pct)) * 100, 2)
            
            # ATR calculations
            atr_5 = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 5).average_true_range().iloc[big_candle_idx]
            analysis['atr_value'] = round(atr_5, 2)
            analysis['move_vs_atr_ratio'] = round(big_candle_move / max(0.1, atr_5), 2)
            
            # 4. ORDER FLOW ESTIMATION NUMBERS
            # Price-Volume correlation in previous 3 candles
            prev3_price_changes = (prev3['Close'] - prev3['Open']).values
            prev3_volume_changes = (prev3['Volume'] / prev3['Volume'].mean()).values
            price_volume_corr = np.corrcoef(prev3_price_changes, prev3_volume_changes)[0,1] if len(prev3_price_changes) > 1 else 0
            analysis['price_volume_correlation'] = round(price_volume_corr, 3)
            
            # Bid-Ask imbalance estimation
            body_size = abs(big_candle['Close'] - big_candle['Open'])
            total_range = big_candle['High'] - big_candle['Low']
            if total_range > 0:
                wick_ratio = (total_range - body_size) / total_range
                # High wick ratio suggests rejection (selling pressure)
                analysis['wick_pressure_ratio'] = round(wick_ratio, 3)
            else:
                analysis['wick_pressure_ratio'] = 0.0
            
            # 5. MOMENTUM NUMBERS
            # RSI before big move
            rsi_before = ta.momentum.RSIIndicator(prev3['Close'], 3).rsi().iloc[-1]
            analysis['rsi_before_move'] = round(rsi_before, 2)
            
            # Momentum acceleration
            prev3_momentum = (prev3['Close'].iloc[-1] - prev3['Close'].iloc[0]) / prev3['Close'].iloc[0] * 100
            analysis['prev3_momentum_percent'] = round(prev3_momentum, 2)
            
            # 6. INSTITUTIONAL PRESSURE NUMBERS
            # Large move probability
            if big_candle_move > 30 and analysis['volume_surge_ratio'] > 2.0:
                analysis['institutional_probability'] = 85
            elif big_candle_move > 25 and analysis['volume_surge_ratio'] > 1.5:
                analysis['institutional_probability'] = 70
            elif big_candle_move > 20 and analysis['volume_surge_ratio'] > 1.2:
                analysis['institutional_probability'] = 60
            else:
                analysis['institutional_probability'] = 45
            
            # Aggressive order ratio
            if analysis['price_acceleration_ratio'] > 3.0:
                analysis['aggressive_order_ratio'] = 0.85
            elif analysis['price_acceleration_ratio'] > 2.0:
                analysis['aggressive_order_ratio'] = 0.70
            elif analysis['price_acceleration_ratio'] > 1.5:
                analysis['aggressive_order_ratio'] = 0.60
            else:
                analysis['aggressive_order_ratio'] = 0.40
            
            # 7. TIMING AND STRENGTH NUMBERS
            analysis['timestamp'] = df.index[big_candle_idx]
            analysis['strength_index'] = min(10, round((analysis['price_acceleration_ratio'] + analysis['volume_surge_ratio'] + analysis['move_vs_atr_ratio']) / 3, 2))
            
            return analysis
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return None
    
    def find_all_big_candles(self, df, threshold=20):
        """Find ALL big candles >= threshold points"""
        big_candles = []
        try:
            for i in range(3, len(df)):
                candle_move = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                if candle_move >= threshold:
                    analysis = self.analyze_big_candle_with_context(df, i)
                    if analysis:
                        big_candles.append(analysis)
        except Exception as e:
            print(f"Error finding big candles: {e}")
        return big_candles

# --------- TELEGRAM FORMATTING ---------
def format_numerical_analysis(index, timeframe, analysis):
    """Format analysis with ONLY NUMERICAL VALUES"""
    
    msg = f"""
🔢 <b>NUMERICAL INSTITUTIONAL ANALYSIS - {index} {timeframe}</b>
🕒 <b>Time</b>: {analysis['timestamp'].strftime('%H:%M:%S')}

📊 <b>BIG CANDLE NUMBERS</b>:
• Direction: {analysis['big_candle_direction']}
• Points Moved: {analysis['big_candle_points']}
• Range: {analysis['big_candle_range']} points
• Volume: {analysis['big_candle_volume']:,}

📈 <b>PRICE ACCELERATION</b>:
• Prev 3-Candle Move: {analysis['prev3_cumulative_points']} points
• Acceleration Ratio: {analysis['price_acceleration_ratio']}x
• Momentum Before: {analysis['prev3_momentum_percent']}%

📉 <b>VOLUME ANALYSIS</b>:
• Volume Surge: {analysis['volume_surge_ratio']}x
• Volume Change: {analysis['volume_change_percent']}%
• Price-Volume Corr: {analysis['price_volume_correlation']}

⚡ <b>VOLATILITY NUMBERS</b>:
• Range Expansion: {analysis['range_expansion_percent']}%
• ATR Value: {analysis['atr_value']}
• Move/ATR Ratio: {analysis['move_vs_atr_ratio']}

🎯 <b>ORDER FLOW METRICS</b>:
• Wick Pressure: {analysis['wick_pressure_ratio']}
• RSI Before: {analysis['rsi_before_move']}
• Aggressive Orders: {analysis['aggressive_order_ratio']}

🏛️ <b>INSTITUTIONAL SCORES</b>:
• Institutional Probability: {analysis['institutional_probability']}%
• Strength Index: {analysis['strength_index']}/10
• Aggressive Order Ratio: {analysis['aggressive_order_ratio']}

💡 <b>INTERPRETATION</b>:
Move caused by {analysis['big_candle_direction']} pressure with {analysis['volume_surge_ratio']}x volume surge
Price accelerated {analysis['price_acceleration_ratio']}x vs previous momentum
Institutional activity probability: {analysis['institutional_probability']}%

────────────────────
"""
    return msg

# --------- MAIN ANALYSIS FUNCTION ---------
def analyze_todays_big_moves():
    """Analyze today's big moves for all indices"""
    
    analyzer = NumericalInstitutionalAnalyzer()
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    timeframes = ["1m", "5m"]
    
    startup_msg = """
🔢 <b>NUMERICAL INSTITUTIONAL ANALYZER STARTED</b>
📅 <b>Date</b>: 17 NOV 2025
🎯 <b>Target</b>: 20+ points moves
📊 <b>Indices</b>: NIFTY, BANKNIFTY, SENSEX
⏰ <b>Timeframes</b>: 1min + 5min

<b>ANALYZING TODAY'S BIG MOVES...</b>
"""
    send_telegram(startup_msg)
    print("Starting numerical analysis...")
    
    for index in indices:
        for timeframe in timeframes:
            try:
                print(f"Analyzing {index} {timeframe}...")
                df = fetch_todays_data(index, timeframe)
                
                if df is not None and len(df) > 10:
                    big_candles = analyzer.find_all_big_candles(df, BIG_CANDLE_THRESHOLD)
                    
                    if big_candles:
                        for i, analysis in enumerate(big_candles):
                            message = format_numerical_analysis(index, timeframe, analysis)
                            send_telegram(message)
                            time.sleep(2)  # Avoid rate limiting
                        
                        summary_msg = f"""
✅ <b>{index} {timeframe} SUMMARY</b>
📈 Found {len(big_candles)} big moves (≥20 points)
🕒 Analysis completed: {datetime.now().strftime('%H:%M:%S')}
"""
                        send_telegram(summary_msg)
                    else:
                        no_moves_msg = f"""
❌ <b>{index} {timeframe}</b>
📊 No big moves ≥20 points detected
🕒 Checked {len(df)} candles
"""
                        send_telegram(no_moves_msg)
                
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"""
⚠️ <b>ERROR: {index} {timeframe}</b>
🔧 {str(e)}
"""
                send_telegram(error_msg)
                continue
    
    completion_msg = """
✅ <b>NUMERICAL ANALYSIS COMPLETED</b>
📅 Date: 17 NOV 2025
🕒 Finished: """ + datetime.now().strftime('%H:%M:%S') + """
📊 All indices analyzed for 20+ points moves
"""
    send_telegram(completion_msg)

# --------- RUN ANALYSIS ---------
if __name__ == "__main__":
    print("Starting Numerical Institutional Analyzer for 17 NOV 2025...")
    analyze_todays_big_moves()
