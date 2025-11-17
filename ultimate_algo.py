# ULTIMATE ERROR-FREE HISTORATIONAL INSTITUTIONAL ANALYZER - ENHANCED VERSION
import os
import time
import requests
import pandas as pd
import yfinance as yf
import warnings
import numpy as np
from datetime import datetime, timedelta
import pytz

warnings.filterwarnings("ignore")

# --------- CONFIGURATION ---------
BIG_CANDLE_THRESHOLD = 20
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Timezone setup
IST = pytz.timezone('Asia/Kolkata')

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        response = requests.post(url, data=payload, timeout=10)
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# --------- SMART DATE SELECTION ---------
def get_analysis_date():
    """Get the correct date for analysis - today if market open, else last trading day"""
    now_ist = datetime.now(IST)
    
    # Check if market is open today (Monday to Friday, 9:15 AM - 3:30 PM IST)
    if now_ist.weekday() < 5:  # Monday to Friday
        market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # If current time is after market open, analyze today
        if now_ist >= market_open:
            analysis_date = now_ist.date()
            market_status = "LIVE_MARKET" if now_ist <= market_close else "TODAYS_CLOSED_MARKET"
        else:
            # Before market open, analyze previous trading day
            analysis_date = (now_ist - timedelta(days=1)).date()
            market_status = "PREVIOUS_DAY_ANALYSIS"
    else:
        # Weekend - analyze last Friday
        days_back = 1 if now_ist.weekday() == 5 else 2  # Saturday or Sunday
        analysis_date = (now_ist - timedelta(days=days_back)).date()
        market_status = "WEEKEND_ANALYSIS"
    
    return analysis_date, market_status

# --------- SAFE DATA FETCHING ---------
def fetch_historical_data_safe(index, analysis_date, interval="1m"):
    try:
        symbol_map = {
            "NIFTY": "^NSEI", 
            "BANKNIFTY": "^NSEBANK", 
            "SENSEX": "^BSESN"
        }
        
        # Convert to string for yfinance
        date_str = analysis_date.strftime("%Y-%m-%d")
        
        # For 1-minute data, we need to handle market hours
        if interval == "1m":
            # Fetch data for the specific date
            df = yf.download(
                symbol_map[index], 
                start=date_str, 
                end=(analysis_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval, 
                progress=False
            )
        else:
            df = yf.download(symbol_map[index], start=date_str, interval=interval, progress=False)
        
        if df.empty:
            print(f"No data for {index} {interval} on {date_str}")
            return None
            
        print(f"✅ Fetched {len(df)} candles for {index} {interval} on {date_str}")
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

# --------- ENHANCED INSTITUTIONAL ANALYZER ---------
class EnhancedInstitutionalAnalyzer:
    def __init__(self):
        self.analyzed_candles = set()
    
    def analyze_big_candle_enhanced(self, df, big_candle_idx):
        """ENHANCED ANALYSIS - With selling pressure and IST timing"""
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
            
            # Convert timestamp to IST
            candle_timestamp = df.index[big_candle_idx]
            if candle_timestamp.tzinfo is None:
                candle_timestamp = pytz.UTC.localize(candle_timestamp)
            ist_time = candle_timestamp.astimezone(IST)
            
            analysis = {
                'timestamp': ist_time,
                'time_str': ist_time.strftime('%H:%M:%S'),
                'date_str': ist_time.strftime('%d %b %Y'),
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
                
                # Convert previous candle time to IST
                prev_timestamp = df.index[big_candle_idx-3+i]
                if prev_timestamp.tzinfo is None:
                    prev_timestamp = pytz.UTC.localize(prev_timestamp)
                prev_ist_time = prev_timestamp.astimezone(IST)
                
                candle_data = {
                    'time': prev_ist_time.strftime('%H:%M:%S'),
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
            
            # Calculate enhanced institutional metrics
            analysis.update(self.calculate_enhanced_metrics(
                current_open, current_high, current_low, current_close, current_volume,
                prev_rows, direction
            ))
            
            return analysis
            
        except Exception as e:
            print(f"Enhanced analysis error at index {big_candle_idx}: {e}")
            return None
    
    def calculate_enhanced_metrics(self, curr_open, curr_high, curr_low, curr_close, curr_volume, prev_rows, direction):
        """ENHANCED metric calculations with selling pressure"""
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
            avg_prev_volume = np.mean(prev_volumes) if prev_volumes else max(1, curr_volume)
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
            
            # ENHANCED: Buying & Selling Pressure
            green_candles = 0
            red_candles = 0
            for row in prev_rows:
                prev_open = safe_float(row['Open'])
                prev_close = safe_float(row['Close'])
                if prev_close > prev_open:
                    green_candles += 1
                else:
                    red_candles += 1
            
            buying_pressure_ratio = round(green_candles / 3, 2)
            selling_pressure_ratio = round(red_candles / 3, 2)
            
            # Enhanced Institutional Score
            score = 0
            
            # Volume scoring
            if volume_surge_ratio > 2.0: score += 35
            elif volume_surge_ratio > 1.5: score += 25
            elif volume_surge_ratio > 1.2: score += 15
            
            # Volatility scoring
            if volatility_expansion > 75: score += 30
            elif volatility_expansion > 50: score += 20
            elif volatility_expansion > 25: score += 10
            
            # Momentum scoring
            if abs(price_momentum) > 0.15: score += 20
            elif abs(price_momentum) > 0.08: score += 15
            elif abs(price_momentum) > 0.03: score += 10
            
            # Candle size scoring
            candle_size = abs(curr_close - curr_open)
            if candle_size > 40: score += 15
            elif candle_size > 30: score += 10
            elif candle_size > 20: score += 5
            
            # Pressure scoring based on direction
            if direction == "GREEN" and buying_pressure_ratio > 0.6:
                score += 10
            elif direction == "RED" and selling_pressure_ratio > 0.6:
                score += 10
            
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
                'selling_pressure_ratio': selling_pressure_ratio,
                'institutional_score': institutional_score,
                'institutional_confidence': confidence,
                'institutional_activity': activity,
                'pressure_direction': "BUYING" if direction == "GREEN" else "SELLING"
            }
            
        except Exception as e:
            print(f"Enhanced metrics error: {e}")
            return {}
    
    def find_all_big_candles_enhanced(self, df, threshold=20):
        """ENHANCED method to find all big candles"""
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
                        analysis = self.analyze_big_candle_enhanced(df, i)
                        if analysis:
                            big_candles.append(analysis)
                except Exception as e:
                    continue  # Skip problematic candles
                        
            return big_candles
            
        except Exception as e:
            print(f"Find big candles error: {e}")
            return []

# --------- ENHANCED TELEGRAM MESSAGE FORMATTING ---------
def format_enhanced_analysis_message(index, timeframe, analysis, market_status):
    """Format enhanced analysis for Telegram"""
    
    # Format previous candles
    prev_candles_text = ""
    for i, candle in enumerate(analysis['prev_candles'], 1):
        prev_candles_text += f"""
    {i}. {candle['time']} - {candle['direction']} {candle['points_move']} points
       O: {candle['open']} | H: {candle['high']} | L: {candle['low']} | C: {candle['close']}
       Range: {candle['range']} pts | Volume: {candle['volume']:,}"""
    
    # Market status indicator
    status_emoji = "🟢" if "LIVE" in market_status else "🟡"
    
    # Pressure display based on candle direction
    if analysis['direction'] == "GREEN":
        pressure_display = f"• Buying Pressure: {analysis['buying_pressure_ratio']}"
    else:
        pressure_display = f"• Selling Pressure: {analysis['selling_pressure_ratio']}"
    
    msg = f"""
{status_emoji} **BIG CANDLE DETECTED - {index} {timeframe}** {status_emoji}

📅 **DATE**: {analysis['date_str']}
⏰ **TIME**: {analysis['time_str']} IST
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
{pressure_display}
• Overall Buying Pressure: {analysis['buying_pressure_ratio']}
• Overall Selling Pressure: {analysis['selling_pressure_ratio']}

🏛️ **INSTITUTIONAL ASSESSMENT**:
• Institutional Score: {analysis['institutional_score']}/100
• Confidence: {analysis['institutional_confidence']}
• Activity Type: {analysis['institutional_activity']}

🎯 **TRADING IMPLICATION**:
Consider {analysis['direction']} positions | {analysis['institutional_confidence']} confidence
📊 Market Status: {market_status}

─────────────────────────────
"""
    return msg

# --------- ENHANCED MAIN ANALYSIS FUNCTION ---------
def analyze_enhanced_historical_data():
    """ENHANCED analysis with smart date selection"""
    
    analyzer = EnhancedInstitutionalAnalyzer()
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    timeframes = ["1m", "5m"]
    
    # Get smart analysis date
    analysis_date, market_status = get_analysis_date()
    
    startup_msg = f"""
📊 **ENHANCED INSTITUTIONAL ANALYSIS STARTED**
📅 Analysis Date: {analysis_date.strftime('%d %b %Y')}
🎯 Target: {BIG_CANDLE_THRESHOLD}+ points moves
📈 Analyzing: NIFTY, BANKNIFTY, SENSEX
⏰ Timeframes: 1min + 5min
📊 Market Status: {market_status}

**PROCESSING ENHANCED ANALYSIS...**
"""
    send_telegram(startup_msg)
    print(f"Starting enhanced analysis for {analysis_date}...")
    
    total_big_moves = 0
    
    for index in indices:
        index_moves = 0
        
        for timeframe in timeframes:
            try:
                print(f"🔍 Analyzing {index} {timeframe} for {analysis_date}...")
                
                # Fetch historical data
                df = fetch_historical_data_safe(index, analysis_date, timeframe)
                
                if df is not None and len(df) > 10:
                    # Find big candles with enhanced analysis
                    big_candles = analyzer.find_all_big_candles_enhanced(df, BIG_CANDLE_THRESHOLD)
                    
                    if big_candles:
                        # Send each analysis
                        for analysis in big_candles:
                            message = format_enhanced_analysis_message(index, timeframe, analysis, market_status)
                            if send_telegram(message):
                                print(f"✅ Sent {index} {timeframe} at {analysis['time_str']} IST")
                                total_big_moves += 1
                                index_moves += 1
                            time.sleep(3)
                    
                    # Send timeframe summary
                    summary_msg = f"""
📋 **{index} {timeframe} SUMMARY**
{'✅' if big_candles else '❌'} Found {len(big_candles)} big moves (≥{BIG_CANDLE_THRESHOLD} points)
📅 Date: {analysis_date.strftime('%d %b %Y')}
"""
                    send_telegram(summary_msg)
                    
                else:
                    no_data_msg = f"""
⚠️ **{index} {timeframe}**
📊 No data available for {analysis_date.strftime('%d %b %Y')}
"""
                    send_telegram(no_data_msg)
                
                time.sleep(2)
                
            except Exception as e:
                error_msg = f"""
❌ **ERROR: {index} {timeframe}**
🔧 {str(e)}
📅 Date: {analysis_date.strftime('%d %b %Y')}
"""
                send_telegram(error_msg)
                continue
        
        # Index completion message
        if index_moves > 0:
            index_msg = f"""
🏁 **{index} COMPLETED**
📈 Found {index_moves} big moves
📅 Date: {analysis_date.strftime('%d %b %Y')}
"""
            send_telegram(index_msg)
    
    # Final completion message
    completion_msg = f"""
🎉 **ENHANCED ANALYSIS COMPLETED** 🎉

📅 Analysis Date: {analysis_date.strftime('%d %b %Y')}
🕒 Finished: {datetime.now(IST).strftime('%H:%M:%S IST')}
📊 Total Big Moves: {total_big_moves}
📈 Market Status: {market_status}
✅ All indices processed

**READY FOR NEXT ANALYSIS**
"""
    send_telegram(completion_msg)
    print(f"✅ Enhanced analysis completed! Found {total_big_moves} big moves for {analysis_date}")

# --------- RUN ENHANCED ANALYSIS ---------
if __name__ == "__main__":
    print("🚀 Starting Enhanced Institutional Analysis...")
    analyze_enhanced_historical_data()
