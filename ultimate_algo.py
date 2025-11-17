# ULTIMATE INSTITUTIONAL ANALYZER - FIXED VERSION
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

# --------- SMART DATA FETCHING ---------
def fetch_market_data(index, interval="5m"):
    """Fetch today's market data (9:15 AM to 3:30 PM)"""
    try:
        symbol_map = {
            "NIFTY": "^NSEI", 
            "BANKNIFTY": "^NSEBANK", 
            "SENSEX": "^BSESN"
        }
        
        # Get TODAY's date (whatever day you run the code)
        today = datetime.now().strftime("%Y-%m-%d")
        df = yf.download(symbol_map[index], start=today, interval=interval, progress=False)
        
        if df.empty:
            print(f"No data for {index} {interval}")
            return None
        
        # Filter ONLY market hours (9:15 AM to 3:30 PM IST)
        market_data = df.between_time('09:15', '15:30')
        
        if market_data.empty:
            print(f"⚠️ No market hours data for {index} {interval}")
            return None
            
        print(f"✅ Fetched {len(market_data)} MARKET candles for {index} {interval}")
        return market_data
        
    except Exception as e:
        print(f"Data error {index} {interval}: {e}")
        return None

# --------- VOLUME ESTIMATION ---------
def estimate_realistic_volume(df, candle_idx, points_moved, candle_range):
    """Estimate realistic volume since Yahoo gives zero for indices"""
    try:
        # Base volumes for different indices
        base_volumes = {
            "NIFTY": 1500000,
            "BANKNIFTY": 2000000, 
            "SENSEX": 800000
        }
        
        # Get index from dataframe context (approximate)
        if points_moved > 50 or candle_range > 40:
            volume_multiplier = 2.5
        elif points_moved > 30 or candle_range > 25:
            volume_multiplier = 2.0
        elif points_moved > 20 or candle_range > 15:
            volume_multiplier = 1.5
        else:
            volume_multiplier = 1.0
        
        # Use average base volume
        base_volume = 1500000
        estimated_volume = int(base_volume * volume_multiplier)
        
        return {
            'estimated_volume': estimated_volume,
            'volume_surge_ratio': round(volume_multiplier, 2),
            'volume_change_percent': int((volume_multiplier - 1) * 100)
        }
        
    except Exception as e:
        return {'estimated_volume': 1000000, 'volume_surge_ratio': 1.0, 'volume_change_percent': 0}

# --------- FIXED INSTITUTIONAL ANALYZER ---------
class FixedInstitutionalAnalyzer:
    def __init__(self):
        self.analyzed_candles = set()
    
    def analyze_big_candle_fixed(self, df, big_candle_idx, index_name):
        """FIXED analysis with selling pressure and proper timing"""
        try:
            if len(df) <= big_candle_idx or big_candle_idx < 3:
                return None
            
            # Get candle data
            current_candle = df.iloc[big_candle_idx]
            prev1_candle = df.iloc[big_candle_idx-1]
            prev2_candle = df.iloc[big_candle_idx-2]  
            prev3_candle = df.iloc[big_candle_idx-3]
            
            # Calculate metrics
            big_candle_move = abs(current_candle['Close'] - current_candle['Open'])
            direction = "GREEN" if current_candle['Close'] > current_candle['Open'] else "RED"
            
            # Estimate volume
            volume_data = estimate_realistic_volume(df, big_candle_idx, big_candle_move, 
                                                  current_candle['High'] - current_candle['Low'])
            
            # Convert timestamp to proper format with AM/PM
            timestamp = df.index[big_candle_idx]
            time_str_24hr = timestamp.strftime('%H:%M:%S')
            time_str_12hr = timestamp.strftime('%I:%M:%S %p')  # 07:25:00 AM/PM
            
            analysis = {
                'timestamp': timestamp,
                'time_str_24hr': time_str_24hr,
                'time_str_12hr': time_str_12hr,
                'direction': direction,
                'points_moved': round(float(big_candle_move), 2),
                'candle_range': round(float(current_candle['High'] - current_candle['Low']), 2),
                'volume': volume_data['estimated_volume'],
                'volume_surge_ratio': volume_data['volume_surge_ratio'],
                'volume_change_percent': volume_data['volume_change_percent'],
                'prev_candles': []
            }
            
            # Analyze previous 3 candles with proper timing
            prev_candles = [prev3_candle, prev2_candle, prev1_candle]
            for i, candle in enumerate(prev_candles):
                prev_timestamp = df.index[big_candle_idx-3+i]
                prev_time_12hr = prev_timestamp.strftime('%I:%M:%S %p')
                prev_volume = estimate_realistic_volume(df, big_candle_idx-3+i, 
                                                       abs(candle['Close'] - candle['Open']),
                                                       candle['High'] - candle['Low'])
                
                candle_data = {
                    'time_24hr': prev_timestamp.strftime('%H:%M:%S'),
                    'time_12hr': prev_time_12hr,
                    'open': round(float(candle['Open']), 2),
                    'high': round(float(candle['High']), 2), 
                    'low': round(float(candle['Low']), 2),
                    'close': round(float(candle['Close']), 2),
                    'points_move': round(abs(float(candle['Close']) - float(candle['Open'])), 2),
                    'direction': "GREEN" if candle['Close'] > candle['Open'] else "RED",
                    'volume': prev_volume['estimated_volume'],
                    'range': round(float(candle['High'] - candle['Low']), 2)
                }
                analysis['prev_candles'].append(candle_data)
            
            # Calculate FIXED institutional metrics with SELLING PRESSURE
            analysis.update(self.calculate_fixed_metrics(df, big_candle_idx, volume_data, index_name))
            
            return analysis
            
        except Exception as e:
            print(f"Fixed analysis error: {e}")
            return None
    
    def calculate_fixed_metrics(self, df, big_candle_idx, volume_data, index_name):
        """Calculate FIXED metrics with SELLING PRESSURE"""
        try:
            current_candle = df.iloc[big_candle_idx]
            current_move = abs(float(current_candle['Close']) - float(current_candle['Open']))
            current_range = float(current_candle['High']) - float(current_candle['Low'])
            current_direction = 1 if current_candle['Close'] > current_candle['Open'] else -1
            
            # Get previous 3 candles
            prev_candles = []
            for i in range(1, 4):
                if big_candle_idx - i >= 0:
                    prev_candles.append(df.iloc[big_candle_idx-i])
            
            # Calculate momentum
            if len(prev_candles) >= 3:
                start_price = float(prev_candles[0]['Close'])
                end_price = float(prev_candles[-1]['Close'])
                price_momentum = (end_price - start_price) / start_price * 100
            else:
                price_momentum = 0
            
            # Calculate volatility expansion
            prev_ranges = [float(c['High']) - float(c['Low']) for c in prev_candles]
            avg_prev_range = np.mean(prev_ranges) if prev_ranges else current_range
            volatility_expansion = ((current_range - avg_prev_range) / max(0.1, avg_prev_range)) * 100
            
            # Calculate BOTH BUYING AND SELLING PRESSURE
            green_candles = 0
            red_candles = 0
            
            for candle in prev_candles:
                if candle['Close'] > candle['Open']:
                    green_candles += 1
                else:
                    red_candles += 1
            
            buying_pressure = round(green_candles / 3, 2)
            selling_pressure = round(red_candles / 3, 2)
            
            # Calculate net pressure based on current direction
            if current_direction == 1:  # Green candle
                net_pressure = buying_pressure - selling_pressure
                pressure_direction = "BULLISH" if net_pressure > 0 else "BEARISH"
            else:  # Red candle  
                net_pressure = selling_pressure - buying_pressure
                pressure_direction = "BEARISH" if net_pressure > 0 else "BULLISH"
            
            # FIXED Institutional Score (more realistic)
            score = 0
            
            # Volume factor (30 points)
            if volume_data['volume_surge_ratio'] > 2.5: score += 30
            elif volume_data['volume_surge_ratio'] > 2.0: score += 25
            elif volume_data['volume_surge_ratio'] > 1.5: score += 20
            
            # Move strength factor (25 points)
            if current_move > 40: score += 25
            elif current_move > 30: score += 20
            elif current_move > 25: score += 15
            elif current_move > 20: score += 10
            
            # Volatility factor (25 points)
            if volatility_expansion > 80: score += 25
            elif volatility_expansion > 60: score += 20
            elif volatility_expansion > 40: score += 15
            
            # Pressure factor (20 points)
            if abs(net_pressure) > 0.5: score += 20
            elif abs(net_pressure) > 0.3: score += 15
            elif abs(net_pressure) > 0.1: score += 10
            
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
                'prev_momentum_percent': round(price_momentum, 2),
                'volatility_expansion': round(volatility_expansion, 2),
                'buying_pressure': buying_pressure,
                'selling_pressure': selling_pressure,
                'net_pressure': round(net_pressure, 2),
                'pressure_direction': pressure_direction,
                'institutional_score': institutional_score,
                'institutional_confidence': confidence,
                'institutional_activity': activity,
                'move_strength': "VERY_STRONG" if current_move > 35 else "STRONG" if current_move > 25 else "MODERATE"
            }
            
        except Exception as e:
            print(f"Fixed metrics error: {e}")
            return {}
    
    def find_big_candles_fixed(self, df, threshold=20, index_name=""):
        """Find big candles with FIXED logic"""
        big_candles = []
        try:
            if df is None or len(df) < 4:
                return big_candles
                
            for i in range(3, len(df)):
                candle_move = abs(float(df['Close'].iloc[i]) - float(df['Open'].iloc[i]))
                if candle_move >= threshold:
                    analysis = self.analyze_big_candle_fixed(df, i, index_name)
                    if analysis:
                        big_candles.append(analysis)
                        
            return big_candles
            
        except Exception as e:
            print(f"Find candles error: {e}")
            return []

# --------- FIXED TELEGRAM MESSAGE ---------
def format_fixed_analysis_message(index, timeframe, analysis):
    """FIXED message format with selling pressure and proper timing"""
    
    prev_candles_text = ""
    for i, candle in enumerate(analysis['prev_candles'], 1):
        prev_candles_text += f"""
    {i}. {candle['time_12hr']} - {candle['direction']} {candle['points_move']} points
       O: {candle['open']} | H: {candle['high']} | L: {candle['low']} | C: {candle['close']}
       Range: {candle['range']} pts | Volume: {candle['volume']:,}"""
    
    msg = f"""
🔴🟢 **BIG CANDLE DETECTED - {index} {timeframe}** 🔴🟢

⏰ **TIME**: {analysis['time_str_12hr']} ({analysis['time_str_24hr']})
🎯 **DIRECTION**: {analysis['direction']} 
📈 **POINTS MOVED**: {analysis['points_moved']} points ({analysis['move_strength']})
📊 **CANDLE RANGE**: {analysis['candle_range']} points  
📦 **EST. VOLUME**: {analysis['volume']:,} ({analysis['volume_surge_ratio']}x)

📋 **PREVIOUS 3 CANDLES**:{prev_candles_text}

📊 **INSTITUTIONAL METRICS**:
• Volume Surge: {analysis['volume_surge_ratio']}x (+{analysis['volume_change_percent']}%)
• Previous Momentum: {analysis['prev_momentum_percent']}%
• Volatility Expansion: {analysis['volatility_expansion']}%
• Buying Pressure: {analysis['buying_pressure']}
• Selling Pressure: {analysis['selling_pressure']}
• Net Pressure: {analysis['net_pressure']} ({analysis['pressure_direction']})

🏛️ **INSTITUTIONAL ASSESSMENT**:
• Institutional Score: {analysis['institutional_score']}/100
• Confidence: {analysis['institutional_confidence']}
• Activity: {analysis['institutional_activity']}

💡 **INTERPRETATION**:
{analysis['direction']} move with {analysis['volume_surge_ratio']}x volume surge
{analysis['pressure_direction']} pressure from previous candles

🎯 **TRADING VIEW**:
Consider {analysis['direction']} positions | {analysis['institutional_confidence']} confidence

─────────────────────────────
"""
    return msg

# --------- MAIN FIXED ANALYSIS ---------
def analyze_fixed_market_data():
    """Analyze TODAY's market data with FIXED logic"""
    
    analyzer = FixedInstitutionalAnalyzer()
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    timeframes = ["5m", "15m"]  # Using practical timeframes
    
    current_date = datetime.now().strftime('%d %b %Y')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    startup_msg = f"""
📊 **FIXED INSTITUTIONAL ANALYSIS STARTED**
📅 Date: {current_date}
🕒 Analysis Time: {current_time}
🎯 Target: {BIG_CANDLE_THRESHOLD}+ points moves
⏰ Market Hours: 9:15 AM - 3:30 PM
📈 Analyzing: NIFTY, BANKNIFTY, SENSEX

**PROCESSING TODAY'S MARKET DATA...**
"""
    send_telegram(startup_msg)
    print("Starting FIXED institutional analysis...")
    
    total_big_moves = 0
    
    for index in indices:
        index_moves = 0
        
        for timeframe in timeframes:
            try:
                print(f"🔍 Analyzing {index} {timeframe}...")
                
                # Fetch MARKET HOURS data
                df = fetch_market_data(index, timeframe)
                
                if df is not None and len(df) > 10:
                    big_candles = analyzer.find_big_candles_fixed(df, BIG_CANDLE_THRESHOLD, index)
                    
                    if big_candles:
                        for analysis in big_candles:
                            candle_id = f"{index}_{timeframe}_{analysis['time_str_24hr']}"
                            
                            if candle_id not in analyzer.analyzed_candles:
                                message = format_fixed_analysis_message(index, timeframe, analysis)
                                if send_telegram(message):
                                    print(f"✅ Sent {index} {timeframe} at {analysis['time_str_12hr']}")
                                    total_big_moves += 1
                                    index_moves += 1
                                    analyzer.analyzed_candles.add(candle_id)
                                time.sleep(3)
                    
                    # Send timeframe summary
                    summary_msg = f"""
📋 **{index} {timeframe} SUMMARY**
{'✅' if big_candles else '❌'} Found {len(big_candles)} big moves (≥{BIG_CANDLE_THRESHOLD} points)
🕒 Market Hours: 9:15 AM - 3:30 PM
"""
                    send_telegram(summary_msg)
                    
                else:
                    no_data_msg = f"""
⚠️ **{index} {timeframe}**
📊 No market hours data available
🕒 Check if market was open today
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
🏁 **{index} ANALYSIS COMPLETED**
📈 Found {index_moves} big moves in market hours
✅ Ready for chart analysis
"""
            send_telegram(index_msg)
    
    # Final completion
    completion_msg = f"""
🎉 **FIXED ANALYSIS COMPLETED** 🎉

📅 Market Date: {current_date}
🕒 Analysis Time: {current_time}
📊 Total Big Moves: {total_big_moves}
⏰ Market Hours: 9:15 AM - 3:30 PM
✅ All indices processed

**CHECK YOUR CHARTS AT THE SPECIFIED TIMES**
"""
    send_telegram(completion_msg)
    print(f"✅ FIXED analysis completed! Found {total_big_moves} big moves")

# --------- RUN FIXED ANALYSIS ---------
if __name__ == "__main__":
    print("🚀 Starting FIXED Institutional Analysis...")
    analyze_fixed_market_data()
