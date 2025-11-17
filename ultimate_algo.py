# ULTIMATE ANGEL ONE INSTITUTIONAL ANALYZER - COMPLETE ANALYSIS
import os
import time
import requests
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import pyotp

warnings.filterwarnings("ignore")

# --------- ANGEL ONE CONFIGURATION ---------
ANGEL_API_KEY = os.getenv("ANGEL_API_KEY")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID") 
ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD")
ANGEL_TOTP = os.getenv("ANGEL_TOTP")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# --------- SYMBOL TOKEN MAPPING ---------
SYMBOL_TOKENS = {
    "NIFTY": 99926000,      # NIFTY 50 Index
    "BANKNIFTY": 99926009,  # BANK NIFTY Index  
    "SENSEX": 99919000      # SENSEX Index
}

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        response = requests.post(url, data=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# --------- ANGEL ONE DATA FETCHING ---------
class AngelOneData:
    def __init__(self):
        self.smart_api = None
        self.login()
    
    def login(self):
        try:
            self.smart_api = SmartConnect(api_key=ANGEL_API_KEY)
            totp = pyotp.TOTP(ANGEL_TOTP).now()
            data = self.smart_api.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
            if data['status']:
                print("✅ Angel One Login Successful")
                # Refresh token every 30 minutes
                self.refreshToken = data['data']['refreshToken']
            else:
                print("❌ Angel One Login Failed")
        except Exception as e:
            print(f"Login error: {e}")
    
    def get_historical_data(self, symbol, interval="ONE_MINUTE", days=1):
        try:
            token = SYMBOL_TOKENS[symbol]
            exchange = "NSE" if symbol in ["NIFTY", "BANKNIFTY"] else "BSE"
            
            # Calculate from_date (today morning 9:00 AM)
            today = datetime.now()
            from_date = today.replace(hour=9, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")
            to_date = today.strftime("%Y-%m-%d %H:%M")
            
            historicParam = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            data = self.smart_api.getCandleData(historicParam)
            
            if data['status'] and data['data']:
                # Convert to DataFrame
                df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S')
                df.set_index('timestamp', inplace=True)
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                print(f"✅ Fetched {len(df)} candles for {symbol} {interval}")
                return df
            else:
                print(f"❌ No data for {symbol} {interval}")
                return None
                
        except Exception as e:
            print(f"Data fetch error for {symbol}: {e}")
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
            
            # Get current and previous candles - USING .iloc PROPERLY
            current_candle = df.iloc[big_candle_idx]
            prev1_candle = df.iloc[big_candle_idx-1]
            prev2_candle = df.iloc[big_candle_idx-2]  
            prev3_candle = df.iloc[big_candle_idx-3]
            
            # Calculate big candle move - USING .item() to get scalar values
            big_candle_move = abs(current_candle['close'] - current_candle['open'])
            direction = "GREEN" if current_candle['close'] > current_candle['open'] else "RED"
            
            analysis = {
                # Basic candle information
                'timestamp': df.index[big_candle_idx],
                'time_str': df.index[big_candle_idx].strftime('%H:%M:%S'),
                'direction': direction,
                'points_moved': round(float(big_candle_move), 2),
                'candle_range': round(float(current_candle['high'] - current_candle['low']), 2),
                'volume': int(current_candle['volume']),
                
                # Previous 3 candles detailed information
                'prev_candles': []
            }
            
            # Analyze previous 3 candles in detail
            prev_candles = [prev3_candle, prev2_candle, prev1_candle]
            for i, candle in enumerate(prev_candles):
                candle_data = {
                    'time': df.index[big_candle_idx-3+i].strftime('%H:%M:%S'),
                    'open': round(float(candle['open']), 2),
                    'high': round(float(candle['high']), 2), 
                    'low': round(float(candle['low']), 2),
                    'close': round(float(candle['close']), 2),
                    'points_move': round(abs(float(candle['close']) - float(candle['open'])), 2),
                    'direction': "GREEN" if candle['close'] > candle['open'] else "RED",
                    'volume': int(candle['volume']),
                    'range': round(float(candle['high'] - candle['low']), 2)
                }
                analysis['prev_candles'].append(candle_data)
            
            # Calculate institutional metrics
            # 1. Volume Analysis
            current_volume = float(current_candle['volume'])
            prev_volumes = [float(c['volume']) for c in prev_candles]
            avg_prev_volume = np.mean(prev_volumes)
            
            analysis['volume_surge_ratio'] = round(current_volume / max(1, avg_prev_volume), 2)
            analysis['volume_change_percent'] = round(((current_volume - avg_prev_volume) / max(1, avg_prev_volume)) * 100, 2)
            
            # 2. Price Momentum
            prev_closes = [float(c['close']) for c in prev_candles]
            price_momentum = (prev_closes[-1] - prev_closes[0]) / prev_closes[0] * 100
            analysis['prev_momentum_percent'] = round(price_momentum, 2)
            
            # 3. Volatility Analysis
            current_range_pct = (float(current_candle['high']) - float(current_candle['low'])) / float(current_candle['open']) * 100
            prev_ranges = []
            for candle in prev_candles:
                range_pct = (float(candle['high']) - float(candle['low'])) / float(candle['open']) * 100
                prev_ranges.append(range_pct)
            
            avg_prev_range = np.mean(prev_ranges)
            analysis['volatility_expansion'] = round(((current_range_pct - avg_prev_range) / max(0.1, avg_prev_range)) * 100, 2)
            
            # 4. Order Flow Pressure
            green_candles = sum(1 for c in prev_candles if c['close'] > c['open'])
            analysis['buying_pressure_ratio'] = round(green_candles / 3, 2)
            
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
            print(f"Complete analysis error at index {big_candle_idx}: {e}")
            return None
    
    def find_all_big_candles_today(self, df, threshold=20):
        """Find ALL big candles in today's data"""
        big_candles = []
        try:
            if df is None or len(df) < 4:
                return big_candles
                
            for i in range(3, len(df)):
                candle_move = abs(float(df['close'].iloc[i]) - float(df['open'].iloc[i]))
                if candle_move >= threshold:
                    analysis = self.analyze_big_candle_complete(df, i)
                    if analysis:
                        big_candles.append(analysis)
                        
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
def analyze_with_angel_one():
    """Analyze TODAY'S complete moves using Angel One data"""
    
    # Initialize Angel One connection
    angel_data = AngelOneData()
    analyzer = CompleteInstitutionalAnalyzer()
    
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    timeframes = {
        "ONE_MINUTE": "1min",
        "FIVE_MINUTE": "5min"
    }
    
    startup_msg = f"""
📊 **ANGEL ONE INSTITUTIONAL ANALYSIS STARTED**
📅 Date: {datetime.now().strftime('%d %b %Y')}
🎯 Target: 20+ points moves  
📈 Data Source: Angel One Live Data
🔍 Analyzing ALL big moves with previous 3 candles

**FETCHING TODAY'S DATA...**
"""
    send_telegram(startup_msg)
    print("Starting Angel One institutional analysis...")
    
    total_analysis_sent = 0
    
    for index in indices:
        index_big_moves = 0
        
        for interval, timeframe_name in timeframes.items():
            try:
                print(f"🔍 Analyzing {index} {timeframe_name}...")
                
                # Fetch data from Angel One
                df = angel_data.get_historical_data(index, interval)
                
                if df is not None and len(df) > 10:
                    big_candles = analyzer.find_all_big_candles_today(df, 20)
                    
                    if big_candles:
                        # Send analysis for each big candle
                        for analysis in big_candles:
                            message = format_complete_analysis(index, timeframe_name, analysis)
                            if send_telegram(message):
                                print(f"✅ Sent analysis for {index} {timeframe_name} at {analysis['time_str']}")
                                total_analysis_sent += 1
                                index_big_moves += 1
                            time.sleep(3)
                    
                    # Send summary for this timeframe
                    summary_msg = f"""
📋 **{index} {timeframe_name} SUMMARY**
{'✅' if big_candles else '❌'} Found {len(big_candles)} big moves (≥20 points)
🕒 Last Candle: {df.index[-1].strftime('%H:%M:%S') if len(df) > 0 else 'N/A'}
"""
                    send_telegram(summary_msg)
                    
                else:
                    no_data_msg = f"""
⚠️ **{index} {timeframe_name}**
📊 No data available from Angel One
"""
                    send_telegram(no_data_msg)
                
                time.sleep(2)
                
            except Exception as e:
                error_msg = f"""
❌ **ERROR: {index} {timeframe_name}**
🔧 {str(e)}
"""
                send_telegram(error_msg)
                print(f"Error analyzing {index} {timeframe_name}: {e}")
                continue
        
        # Send index summary
        if index_big_moves > 0:
            index_summary = f"""
🏁 **{index} COMPLETE ANALYSIS FINISHED**
📈 Total big moves found: {index_big_moves}
✅ Analysis completed successfully
"""
            send_telegram(index_summary)
        else:
            no_moves_msg = f"""
📭 **{index} ANALYSIS COMPLETED**
❌ No big moves (≥20 points) found today
"""
            send_telegram(no_moves_msg)
    
    # Final completion message
    completion_msg = f"""
🎉 **ANGEL ONE ANALYSIS COMPLETED** 🎉

📅 Date: {datetime.now().strftime('%d %b %Y')}
🕒 Completed: {datetime.now().strftime('%H:%M:%S')}
📊 Total Big Moves Analyzed: {total_analysis_sent}
📈 Data Source: Angel One
✅ All indices processed successfully

**READY FOR TRADING INSIGHTS**
"""
    send_telegram(completion_msg)
    print(f"🎯 Angel One analysis completed! Total analyses sent: {total_analysis_sent}")

# --------- RUN ANALYSIS ---------
if __name__ == "__main__":
    print("🚀 Starting Angel One Institutional Analysis...")
    analyze_with_angel_one()
