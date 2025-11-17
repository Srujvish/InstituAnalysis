# ULTIMATE HISTORICAL INSTITUTIONAL ANALYZER - TODAY'S COMPLETE DATA
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

# --------- HISTORICAL DATA FETCHING ---------
def fetch_todays_historical_data(index, interval="1m"):
    try:
        symbol_map = {
            "NIFTY": "^NSEI", 
            "BANKNIFTY": "^NSEBANK", 
            "SENSEX": "^BSESN"
        }
        
        # Get today's complete data
        today = datetime.now().strftime("%Y-%m-%d")
        df = yf.download(symbol_map[index], start=today, interval=interval, progress=False)
        
        if df.empty:
            print(f"No data found for {index} {interval}")
            return None
            
        print(f"✅ Fetched {len(df)} candles for {index} {interval}")
        return df
        
    except Exception as e:
        print(f"Data fetch error for {index} {interval}: {e}")
        return None

# --------- COMPLETE 3-CANDLE ANALYSIS ---------
class HistoricalInstitutionalAnalyzer:
    def __init__(self):
        self.analyzed_candles = set()
    
    def analyze_big_candle_complete(self, df, big_candle_idx):
        """COMPLETE ANALYSIS of big candle with previous 3 candles"""
        try:
            if len(df) <= big_candle_idx or big_candle_idx < 3:
                return None
            
            # Get current and previous candles
            current_candle = df.iloc[big_candle_idx]
            prev1_candle = df.iloc[big_candle_idx-1]
            prev2_candle = df.iloc[big_candle_idx-2]  
            prev3_candle = df.iloc[big_candle_idx-3]
            
            # Calculate big candle details
            big_candle_move = abs(current_candle['Close'] - current_candle['Open'])
            direction = "GREEN" if current_candle['Close'] > current_candle['Open'] else "RED"
            
            analysis = {
                'timestamp': df.index[big_candle_idx],
                'time_str': df.index[big_candle_idx].strftime('%H:%M:%S'),
                'direction': direction,
                'points_moved': round(float(big_candle_move), 2),
                'candle_range': round(float(current_candle['High'] - current_candle['Low']), 2),
                'volume': int(current_candle['Volume']),
                
                # Previous 3 candles COMPLETE information
                'prev_candles': []
            }
            
            # Analyze previous 3 candles in extreme detail
            prev_candles = [prev3_candle, prev2_candle, prev1_candle]
            for i, candle in enumerate(prev_candles):
                candle_data = {
                    'time': df.index[big_candle_idx-3+i].strftime('%H:%M:%S'),
                    'open': round(float(candle['Open']), 2),
                    'high': round(float(candle['High']), 2), 
                    'low': round(float(candle['Low']), 2),
                    'close': round(float(candle['Close']), 2),
                    'points_move': round(abs(float(candle['Close']) - float(candle['Open'])), 2),
                    'direction': "GREEN" if candle['Close'] > candle['Open'] else "RED",
                    'volume': int(candle['Volume']),
                    'range': round(float(candle['High'] - candle['Low']), 2),
                    'body_ratio': self.calculate_body_ratio(candle),
                    'wick_analysis': self.analyze_wicks(candle)
                }
                analysis['prev_candles'].append(candle_data)
            
            # Institutional metrics
            analysis.update(self.calculate_institutional_metrics(current_candle, prev_candles))
            
            return analysis
            
        except Exception as e:
            print(f"Analysis error at index {big_candle_idx}: {e}")
            return None
    
    def calculate_body_ratio(self, candle):
        """Calculate candle body to range ratio"""
        body_size = abs(float(candle['Close']) - float(candle['Open']))
        total_range = float(candle['High']) - float(candle['Low'])
        return round(body_size / total_range, 3) if total_range > 0 else 0
    
    def analyze_wicks(self, candle):
        """Analyze upper and lower wicks"""
        body_high = max(float(candle['Open']), float(candle['Close']))
        body_low = min(float(candle['Open']), float(candle['Close']))
        
        upper_wick = float(candle['High']) - body_high
        lower_wick = body_low - float(candle['Low'])
        total_range = float(candle['High']) - float(candle['Low'])
        
        upper_ratio = round(upper_wick / total_range, 3) if total_range > 0 else 0
        lower_ratio = round(lower_wick / total_range, 3) if total_range > 0 else 0
        
        if upper_ratio > 0.4:
            pressure = "STRONG_SELLING"
        elif lower_ratio > 0.4:
            pressure = "STRONG_BUYING"
        elif upper_ratio > lower_ratio:
            pressure = "SELLING_PRESSURE"
        elif lower_ratio > upper_ratio:
            pressure = "BUYING_PRESSURE"
        else:
            pressure = "BALANCED"
            
        return {
            'upper_wick_ratio': upper_ratio,
            'lower_wick_ratio': lower_ratio,
            'pressure': pressure
        }
    
    def calculate_institutional_metrics(self, current_candle, prev_candles):
        """Calculate institutional trading metrics"""
        try:
            # Volume Analysis
            current_volume = float(current_candle['Volume'])
            prev_volumes = [float(c['Volume']) for c in prev_candles]
            avg_prev_volume = np.mean(prev_volumes)
            
            volume_surge_ratio = round(current_volume / max(1, avg_prev_volume), 2)
            volume_change_percent = round(((current_volume - avg_prev_volume) / max(1, avg_prev_volume)) * 100, 2)
            
            # Price Momentum Analysis
            prev_closes = [float(c['Close']) for c in prev_candles]
            price_momentum = (prev_closes[-1] - prev_closes[0]) / prev_closes[0] * 100
            
            # Volatility Analysis
            current_range_pct = (float(current_candle['High']) - float(current_candle['Low'])) / float(current_candle['Open']) * 100
            prev_ranges = []
            for candle in prev_candles:
                range_pct = (float(candle['High']) - float(candle['Low'])) / float(candle['Open']) * 100
                prev_ranges.append(range_pct)
            
            avg_prev_range = np.mean(prev_ranges)
            volatility_expansion = round(((current_range_pct - avg_prev_range) / max(0.1, avg_prev_range)) * 100, 2)
            
            # Order Flow Pressure
            green_candles = sum(1 for c in prev_candles if c['Close'] > c['Open'])
            buying_pressure_ratio = round(green_candles / 3, 2)
            
            # Institutional Probability Score
            score = 0
            if volume_surge_ratio > 2.0: score += 35
            elif volume_surge_ratio > 1.5: score += 25
            if volatility_expansion > 75: score += 30
            elif volatility_expansion > 50: score += 20
            if abs(price_momentum) > 0.15: score += 20
            elif abs(price_momentum) > 0.08: score += 15
            if abs(current_candle['Close'] - current_candle['Open']) > 30: score += 15
            
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
            
            # Aggressive Trading Detection
            aggressive_trading = self.detect_aggressive_trading(prev_candles, current_candle)
            
            return {
                'volume_surge_ratio': volume_surge_ratio,
                'volume_change_percent': volume_change_percent,
                'prev_momentum_percent': round(price_momentum, 2),
                'volatility_expansion': volatility_expansion,
                'buying_pressure_ratio': buying_pressure_ratio,
                'institutional_score': institutional_score,
                'institutional_confidence': confidence,
                'institutional_activity': activity,
                'aggressive_trading': aggressive_trading,
                'what_happened': self.explain_what_happened(current_candle, prev_candles, volume_surge_ratio, volatility_expansion, institutional_score)
            }
            
        except Exception as e:
            print(f"Institutional metrics error: {e}")
            return {}
    
    def detect_aggressive_trading(self, prev_candles, current_candle):
        """Detect aggressive institutional trading patterns"""
        try:
            current_direction = 1 if current_candle['Close'] > current_candle['Open'] else -1
            prev_directions = [1 if c['Close'] > c['Open'] else -1 for c in prev_candles]
            
            # Check for momentum buildup
            if all(d == current_direction for d in prev_directions):
                return "AGGRESSIVE_MOMENTUM_CONTINUATION"
            elif sum(prev_directions) == -3 and current_direction == 1:
                return "AGGRESSIVE_REVERSAL_BUYING"
            elif sum(prev_directions) == 3 and current_direction == -1:
                return "AGGRESSIVE_REVERSAL_SELLING"
            else:
                return "MIXED_SENTIMENT"
                
        except:
            return "PATTERN_UNKNOWN"
    
    def explain_what_happened(self, current_candle, prev_candles, volume_surge, volatility_expansion, inst_score):
        """Detailed explanation of what caused the move"""
        direction = "GREEN" if current_candle['Close'] > current_candle['Open'] else "RED"
        points = abs(float(current_candle['Close']) - float(current_candle['Open']))
        
        explanation_parts = []
        
        # Volume analysis
        if volume_surge > 2.5:
            explanation_parts.append("MASSIVE_VOLUME_SURGE")
        elif volume_surge > 2.0:
            explanation_parts.append("HUGE_VOLUME_INCREASE")
        elif volume_surge > 1.5:
            explanation_parts.append("HIGH_VOLUME_PARTICIPATION")
        
        # Volatility analysis
        if volatility_expansion > 100:
            explanation_parts.append("EXTREME_VOLATILITY_EXPANSION")
        elif volatility_expansion > 75:
            explanation_parts.append("HIGH_VOLATILITY_BREAKOUT")
        elif volatility_expansion > 50:
            explanation_parts.append("MODERATE_VOLATILITY_INCREASE")
        
        # Institutional activity
        if inst_score >= 70:
            explanation_parts.append("STRONG_INSTITUTIONAL_ORDER_FLOW")
        elif inst_score >= 50:
            explanation_parts.append("MODERATE_INSTITUTIONAL_INVOLVEMENT")
        
        # Previous context
        prev_directions = ["GREEN" if c['Close'] > c['Open'] else "RED" for c in prev_candles]
        green_count = sum(1 for d in prev_directions if d == "GREEN")
        
        if green_count == 3 and direction == "GREEN":
            explanation_parts.append("CONTINUING_BULLISH_MOMENTUM")
        elif green_count == 0 and direction == "RED":
            explanation_parts.append("CONTINUING_BEARISH_PRESSURE")
        elif green_count >= 2 and direction == "RED":
            explanation_parts.append("REVERSING_BULLISH_SENTIMENT")
        elif green_count <= 1 and direction == "GREEN":
            explanation_parts.append("REVERSING_BEARISH_SENTIMENT")
        
        explanation = f"{direction} {points} points move | " + " | ".join(explanation_parts)
        return explanation
    
    def find_all_big_candles_today(self, df, threshold=20):
        """Find ALL big candles in today's data"""
        big_candles = []
        try:
            if df is None or len(df) < 4:
                return big_candles
                
            for i in range(3, len(df)):
                candle_move = abs(float(df['Close'].iloc[i]) - float(df['Open'].iloc[i]))
                if candle_move >= threshold:
                    analysis = self.analyze_big_candle_complete(df, i)
                    if analysis:
                        big_candles.append(analysis)
                        
            return big_candles
            
        except Exception as e:
            print(f"Error finding big candles: {e}")
            return []

# --------- TELEGRAM MESSAGE FORMATTING ---------
def format_complete_analysis_message(index, timeframe, analysis):
    """Format COMPLETE analysis message for Telegram"""
    
    # Format previous candles information
    prev_candles_text = ""
    for i, candle in enumerate(analysis['prev_candles'], 1):
        wick_info = candle['wick_analysis']
        prev_candles_text += f"""
    {i}. {candle['time']} - {candle['direction']} {candle['points_move']} points
       O: {candle['open']} | H: {candle['high']} | L: {candle['low']} | C: {candle['close']}
       Range: {candle['range']} pts | Volume: {candle['volume']:,}
       Body: {candle['body_ratio']} | Wick Pressure: {wick_info['pressure']}"""
    
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
• Aggressive Trading: {analysis['aggressive_trading']}

🏛️ **INSTITUTIONAL ASSESSMENT**:
• Institutional Score: {analysis['institutional_score']}/100
• Confidence: {analysis['institutional_confidence']}
• Activity Type: {analysis['institutional_activity']}

💡 **WHAT HAPPENED**:
{analysis['what_happened']}

🎯 **TRADING IMPLICATION**:
Consider {analysis['direction']} positions | {analysis['institutional_confidence']} confidence
Institutional probability: {analysis['institutional_score']}%

─────────────────────────────
"""
    return msg

# --------- MAIN HISTORICAL ANALYSIS FUNCTION ---------
def analyze_todays_historical_data():
    """Analyze TODAY'S complete historical data for all indices"""
    
    analyzer = HistoricalInstitutionalAnalyzer()
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    timeframes = ["1m", "5m"]
    
    startup_msg = f"""
📊 **HISTORICAL INSTITUTIONAL ANALYSIS STARTED**
📅 Date: {datetime.now().strftime('%d %b %Y')}
🎯 Target: {BIG_CANDLE_THRESHOLD}+ points moves
📈 Indices: NIFTY, BANKNIFTY, SENSEX
⏰ Timeframes: 1min + 5min
🔍 Analyzing TODAY'S complete data with 3-candle context

**PROCESSING HISTORICAL DATA...**
"""
    send_telegram(startup_msg)
    print("Starting historical institutional analysis...")
    
    total_big_moves_found = 0
    
    for index in indices:
        index_moves = 0
        
        for timeframe in timeframes:
            try:
                print(f"🔍 Analyzing {index} {timeframe}...")
                
                # Fetch historical data for today
                df = fetch_todays_historical_data(index, timeframe)
                
                if df is not None and len(df) > 10:
                    # Find all big candles in today's data
                    big_candles = analyzer.find_all_big_candles_today(df, BIG_CANDLE_THRESHOLD)
                    
                    if big_candles:
                        # Send analysis for each big candle
                        for analysis in big_candles:
                            message = format_complete_analysis_message(index, timeframe, analysis)
                            if send_telegram(message):
                                print(f"✅ Sent analysis for {index} {timeframe} at {analysis['time_str']}")
                                total_big_moves_found += 1
                                index_moves += 1
                            time.sleep(3)  # Avoid rate limiting
                    
                    # Send timeframe summary
                    summary_msg = f"""
📋 **{index} {timeframe} SUMMARY**
{'✅' if big_candles else '❌'} Found {len(big_candles)} big moves (≥{BIG_CANDLE_THRESHOLD} points)
🕒 Total Candles Analyzed: {len(df)}
"""
                    send_telegram(summary_msg)
                    
                else:
                    no_data_msg = f"""
⚠️ **{index} {timeframe}**
📊 No historical data available for analysis
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
        
        # Send index completion message
        if index_moves > 0:
            index_complete_msg = f"""
🏁 **{index} ANALYSIS COMPLETED**
📈 Total big moves found: {index_moves}
✅ Historical analysis finished
"""
            send_telegram(index_complete_msg)
        else:
            no_moves_msg = f"""
📭 **{index} ANALYSIS COMPLETED**
❌ No big moves (≥{BIG_CANDLE_THRESHOLD} points) found today
"""
            send_telegram(no_moves_msg)
    
    # Final completion message
    completion_msg = f"""
🎉 **HISTORICAL ANALYSIS COMPLETED** 🎉

📅 Date: {datetime.now().strftime('%d %b %Y')}
🕒 Finished: {datetime.now().strftime('%H:%M:%S')}
📊 Total Big Moves Found: {total_big_moves_found}
📈 All indices processed successfully

**TODAY'S INSTITUTIONAL ACTIVITY ANALYSIS COMPLETE**
"""
    send_telegram(completion_msg)
    print(f"🎯 Historical analysis completed! Total big moves found: {total_big_moves_found}")

# --------- RUN HISTORICAL ANALYSIS ---------
if __name__ == "__main__":
    print("🚀 Starting Today's Historical Institutional Analysis...")
    analyze_todays_historical_data()
