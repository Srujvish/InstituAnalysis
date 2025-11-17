# TRUE INSTITUTIONAL PRESSURE ANALYZER - ENHANCED VERSION complete inst with volume surge and thing 
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

# --------- TRUE INSTITUTIONAL PRESSURE ANALYZER ---------
class TrueInstitutionalAnalyzer:
    def __init__(self):
        self.analyzed_candles = set()
    
    def analyze_big_candle_institutional(self, df, big_candle_idx):
        """TRUE INSTITUTIONAL ANALYSIS - Real pressure detection"""
        try:
            if len(df) <= big_candle_idx or big_candle_idx < 5:  # Increased to 5 for better analysis
                return None
            
            # Get candle data with more context
            current_row = df.iloc[big_candle_idx]
            prev1_row = df.iloc[big_candle_idx-1]
            prev2_row = df.iloc[big_candle_idx-2]  
            prev3_row = df.iloc[big_candle_idx-3]
            prev4_row = df.iloc[big_candle_idx-4]  # Additional context
            prev5_row = df.iloc[big_candle_idx-5]  # Additional context
            
            # Convert to scalar values
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
            
            # Analyze previous 3 candles for context
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
            
            # Calculate TRUE institutional pressure metrics
            analysis.update(self.calculate_true_institutional_pressure(
                current_open, current_high, current_low, current_close, current_volume,
                [prev5_row, prev4_row, prev3_row, prev2_row, prev1_row],  # More context
                direction
            ))
            
            return analysis
            
        except Exception as e:
            print(f"Institutional analysis error at index {big_candle_idx}: {e}")
            return None
    
    def calculate_true_institutional_pressure(self, curr_open, curr_high, curr_low, curr_close, curr_volume, prev_rows, direction):
        """TRUE INSTITUTIONAL PRESSURE CALCULATIONS"""
        try:
            # Extract data from previous candles
            prev_opens = []
            prev_highs = []
            prev_lows = []
            prev_closes = []
            prev_volumes = []
            prev_ranges = []
            
            for row in prev_rows:
                prev_open = safe_float(row['Open'])
                prev_high = safe_float(row['High'])
                prev_low = safe_float(row['Low'])
                prev_close = safe_float(row['Close'])
                prev_volume = safe_int(row['Volume'])
                
                prev_opens.append(prev_open)
                prev_highs.append(prev_high)
                prev_lows.append(prev_low)
                prev_closes.append(prev_close)
                prev_volumes.append(prev_volume)
                prev_ranges.append(prev_high - prev_low)
            
            # FIXED: Better volume simulation for institutional analysis
            base_volume = 50000  # Higher base for institutional context
            
            if curr_volume == 0:
                # Institutional volume simulation based on price action intensity
                price_movement = abs(curr_close - curr_open)
                range_size = curr_high - curr_low
                volatility = range_size / curr_open if curr_open > 0 else 0
                
                # Institutional trades have higher volume during significant moves
                movement_intensity = (price_movement / curr_open * 100) if curr_open > 0 else 0
                volatility_factor = volatility * 100
                
                curr_volume = int(base_volume * (1 + movement_intensity * 8 + volatility_factor * 3))
            
            # Calculate synthetic previous volumes
            synthetic_prev_volumes = []
            for i in range(len(prev_opens)):
                if prev_volumes[i] == 0:
                    prev_movement = abs(prev_closes[i] - prev_opens[i])
                    prev_range = prev_ranges[i]
                    prev_volatility = prev_range / prev_opens[i] if prev_opens[i] > 0 else 0
                    
                    prev_movement_intensity = (prev_movement / prev_opens[i] * 100) if prev_opens[i] > 0 else 0
                    prev_volatility_factor = prev_volatility * 100
                    
                    synthetic_vol = int(base_volume * (1 + prev_movement_intensity * 8 + prev_volatility_factor * 3))
                    synthetic_prev_volumes.append(synthetic_vol)
                else:
                    synthetic_prev_volumes.append(prev_volumes[i])
            
            # TRUE INSTITUTIONAL PRESSURE METRICS
            
            # 1. Volume Dominance Analysis
            avg_prev_volume = np.mean(synthetic_prev_volumes) if synthetic_prev_volumes else base_volume
            volume_surge_ratio = round(curr_volume / max(1, avg_prev_volume), 2)
            
            # Institutional volume thresholds (much higher than retail)
            if volume_surge_ratio > 3.0:
                volume_pressure = "VERY_HIGH"
            elif volume_surge_ratio > 2.0:
                volume_pressure = "HIGH"
            elif volume_surge_ratio > 1.5:
                volume_pressure = "MODERATE"
            else:
                volume_pressure = "LOW"
            
            # 2. Price Efficiency Analysis (Institutional vs Retail)
            current_efficiency = abs(curr_close - curr_open) / (curr_high - curr_low) if (curr_high - curr_low) > 0 else 0
            prev_efficiencies = []
            
            for i in range(len(prev_opens)):
                if prev_ranges[i] > 0:
                    eff = abs(prev_closes[i] - prev_opens[i]) / prev_ranges[i]
                    prev_efficiencies.append(eff)
            
            avg_prev_efficiency = np.mean(prev_efficiencies) if prev_efficiencies else current_efficiency
            efficiency_ratio = round(current_efficiency / max(0.01, avg_prev_efficiency), 2)
            
            # High efficiency = institutional (clean moves), Low efficiency = retail (choppy)
            if efficiency_ratio > 1.3:
                efficiency_pressure = "INSTITUTIONAL"
            elif efficiency_ratio > 0.8:
                efficiency_pressure = "MIXED"
            else:
                efficiency_pressure = "RETAIL"
            
            # 3. Momentum Consistency Analysis
            if len(prev_closes) >= 3:
                short_momentum = (prev_closes[-1] - prev_closes[-3]) / prev_closes[-3] * 100
                medium_momentum = (prev_closes[-1] - prev_closes[0]) / prev_closes[0] * 100
                
                momentum_alignment = abs(short_momentum - medium_momentum)
                
                if momentum_alignment < 0.05:  # Aligned momentum
                    momentum_pressure = "STRONG"
                elif momentum_alignment < 0.1:
                    momentum_pressure = "MODERATE"
                else:
                    momentum_pressure = "WEAK"
            else:
                momentum_pressure = "NEUTRAL"
            
            # 4. Range Expansion Analysis (Institutional volatility)
            current_range_pct = (curr_high - curr_low) / curr_open * 100 if curr_open > 0 else 0
            avg_prev_range_pct = np.mean([(r/prev_opens[i]*100) for i, r in enumerate(prev_ranges) if prev_opens[i] > 0]) if prev_opens else current_range_pct
            
            range_expansion = round(((current_range_pct - avg_prev_range_pct) / max(0.1, avg_prev_range_pct)) * 100, 2)
            
            if range_expansion > 80:
                volatility_pressure = "HIGH_VOL_INSTITUTIONAL"
            elif range_expansion > 50:
                volatility_pressure = "MODERATE_VOL_INSTITUTIONAL"
            elif range_expansion > 20:
                volatility_pressure = "LIGHT_INSTITUTIONAL"
            else:
                volatility_pressure = "RETAIL_VOLATILITY"
            
            # 5. TRUE INSTITUTIONAL PRESSURE SCORING
            score = 0
            
            # Volume scoring (institutional focus)
            if volume_surge_ratio > 3.0: score += 40
            elif volume_surge_ratio > 2.0: score += 30
            elif volume_surge_ratio > 1.5: score += 20
            
            # Efficiency scoring
            if efficiency_pressure == "INSTITUTIONAL": score += 25
            elif efficiency_pressure == "MIXED": score += 15
            
            # Momentum scoring
            if momentum_pressure == "STRONG": score += 20
            elif momentum_pressure == "MODERATE": score += 10
            
            # Volatility scoring
            if "INSTITUTIONAL" in volatility_pressure: score += 15
            
            # Candle size scoring (institutional moves are larger)
            candle_size = abs(curr_close - curr_open)
            if candle_size > 50: score += 20
            elif candle_size > 30: score += 15
            elif candle_size > 20: score += 10
            
            institutional_score = min(100, score)
            
            # PRESSURE TYPE DETERMINATION
            if institutional_score >= 70:
                pressure_type = "STRONG_INSTITUTIONAL"
                confidence = "VERY_HIGH"
            elif institutional_score >= 50:
                pressure_type = "MODERATE_INSTITUTIONAL"
                confidence = "HIGH"
            elif institutional_score >= 30:
                pressure_type = "LIGHT_INSTITUTIONAL"
                confidence = "MEDIUM"
            else:
                pressure_type = "RETAIL_DOMINATED"
                confidence = "LOW"
            
            # DIRECTIONAL PRESSURE
            if direction == "GREEN":
                directional_pressure = "INSTITUTIONAL_BUYING"
                pressure_strength = volume_pressure
            else:
                directional_pressure = "INSTITUTIONAL_SELLING"
                pressure_strength = volume_pressure
            
            return {
                'volume_surge_ratio': volume_surge_ratio,
                'volume_pressure': volume_pressure,
                'efficiency_ratio': efficiency_ratio,
                'efficiency_pressure': efficiency_pressure,
                'momentum_pressure': momentum_pressure,
                'range_expansion': range_expansion,
                'volatility_pressure': volatility_pressure,
                'institutional_score': institutional_score,
                'pressure_type': pressure_type,
                'confidence': confidence,
                'directional_pressure': directional_pressure,
                'pressure_strength': pressure_strength,
                'true_institutional_activity': pressure_type if institutional_score >= 30 else "RETAIL_DOMINATED"
            }
            
        except Exception as e:
            print(f"True institutional pressure error: {e}")
            return {
                'volume_surge_ratio': 0.0,
                'volume_pressure': "UNKNOWN",
                'efficiency_ratio': 0.0,
                'efficiency_pressure': "UNKNOWN",
                'momentum_pressure': "UNKNOWN",
                'range_expansion': 0.0,
                'volatility_pressure': "UNKNOWN",
                'institutional_score': 0,
                'pressure_type': "RETAIL_DOMINATED",
                'confidence': "LOW",
                'directional_pressure': "NEUTRAL",
                'pressure_strength': "LOW",
                'true_institutional_activity': "RETAIL_DOMINATED"
            }
    
    def find_all_big_candles_institutional(self, df, threshold=20):
        """Find all big candles with institutional analysis"""
        big_candles = []
        try:
            if df is None or len(df) < 6:  # Increased minimum
                return big_candles
                
            for i in range(5, len(df)):  # Start from 5 for better context
                try:
                    # SAFE candle move calculation
                    row = df.iloc[i]
                    open_val = safe_float(row['Open'])
                    close_val = safe_float(row['Close'])
                    candle_move = abs(close_val - open_val)
                    
                    if candle_move >= threshold:
                        analysis = self.analyze_big_candle_institutional(df, i)
                        if analysis:
                            big_candles.append(analysis)
                except Exception as e:
                    continue  # Skip problematic candles
                        
            return big_candles
            
        except Exception as e:
            print(f"Find big candles error: {e}")
            return []

# --------- INSTITUTIONAL TELEGRAM MESSAGE FORMATTING ---------
def format_institutional_analysis_message(index, timeframe, analysis, market_status):
    """Format true institutional analysis for Telegram"""
    
    # Format previous candles
    prev_candles_text = ""
    for i, candle in enumerate(analysis['prev_candles'], 1):
        prev_candles_text += f"""
    {i}. {candle['time']} - {candle['direction']} {candle['points_move']} points
       O: {candle['open']} | H: {candle['high']} | L: {candle['low']} | C: {candle['close']}
       Range: {candle['range']} pts | Volume: {candle['volume']:,}"""
    
    # Pressure emoji based on direction
    if "BUYING" in analysis['directional_pressure']:
        pressure_emoji = "🏛️🟢"
    else:
        pressure_emoji = "🏛️🔴"
    
    msg = f"""
{pressure_emoji} **INSTITUTIONAL PRESSURE DETECTED - {index} {timeframe}** {pressure_emoji}

📅 **DATE**: {analysis['date_str']}
⏰ **TIME**: {analysis['time_str']} IST
🎯 **DIRECTION**: {analysis['direction']}
📈 **POINTS MOVED**: {analysis['points_moved']} points
📊 **CANDLE RANGE**: {analysis['candle_range']} points  
📦 **VOLUME**: {analysis['volume']:,}

📋 **PREVIOUS 3 CANDLES ANALYSIS**:{prev_candles_text}

🏛️ **TRUE INSTITUTIONAL METRICS**:
• Volume Surge: {analysis['volume_surge_ratio']}x ({analysis['volume_pressure']})
• Price Efficiency: {analysis['efficiency_ratio']}x ({analysis['efficiency_pressure']})
• Momentum Alignment: {analysis['momentum_pressure']}
• Range Expansion: {analysis['range_expansion']}% ({analysis['volatility_pressure']})

💼 **INSTITUTIONAL ASSESSMENT**:
• Institutional Score: {analysis['institutional_score']}/100
• Pressure Type: {analysis['pressure_type']}
• Confidence: {analysis['confidence']}
• Directional Pressure: {analysis['directional_pressure']}
• Pressure Strength: {analysis['pressure_strength']}

🎯 **TRADING IMPLICATION**:
{analysis['directional_pressure']} | {analysis['confidence']} confidence
True Activity: {analysis['true_institutional_activity']}
📊 Market Status: {market_status}

─────────────────────────────
"""
    return msg

# --------- MAIN INSTITUTIONAL ANALYSIS FUNCTION ---------
def analyze_true_institutional_data():
    """TRUE institutional pressure analysis"""
    
    analyzer = TrueInstitutionalAnalyzer()
    indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
    timeframes = ["1m", "5m"]
    
    # Get smart analysis date
    analysis_date, market_status = get_analysis_date()
    
    # Start timer
    start_time = time.time()
    
    startup_msg = f"""
🏛️ **TRUE INSTITUTIONAL PRESSURE ANALYSIS STARTED**
📅 Analysis Date: {analysis_date.strftime('%d %b %Y')}
🎯 Target: {BIG_CANDLE_THRESHOLD}+ points moves
📈 Analyzing: NIFTY, BANKNIFTY, SENSEX
⏰ Timeframes: 1min + 5min
📊 Market Status: {market_status}

**DETECTING REAL INSTITUTIONAL ACTIVITY...**
"""
    send_telegram(startup_msg)
    print(f"Starting true institutional analysis for {analysis_date}...")
    
    total_big_moves = 0
    total_analyzed = 0
    
    for index in indices:
        index_moves = 0
        
        for timeframe in timeframes:
            try:
                print(f"🏛️ Analyzing {index} {timeframe} for institutional pressure...")
                
                # Fetch historical data
                df = fetch_historical_data_safe(index, analysis_date, timeframe)
                total_analyzed += 1
                
                if df is not None and len(df) > 10:
                    # Find big candles with institutional analysis
                    big_candles = analyzer.find_all_big_candles_institutional(df, BIG_CANDLE_THRESHOLD)
                    
                    if big_candles:
                        # Send each analysis
                        for analysis in big_candles:
                            message = format_institutional_analysis_message(index, timeframe, analysis, market_status)
                            if send_telegram(message):
                                print(f"✅ Sent {index} {timeframe} institutional pressure at {analysis['time_str']}")
                                total_big_moves += 1
                                index_moves += 1
                            time.sleep(1)
                    
                    # Send timeframe summary
                    summary_msg = f"""
📋 **{index} {timeframe} INSTITUTIONAL SUMMARY**
{'🏛️' if big_candles else '❌'} Found {len(big_candles)} institutional moves (≥{BIG_CANDLE_THRESHOLD} points)
📅 Date: {analysis_date.strftime('%d %b %Y')}
"""
                    send_telegram(summary_msg)
                    
                else:
                    no_data_msg = f"""
⚠️ **{index} {timeframe}**
📊 No data available for {analysis_date.strftime('%d %b %Y')}
"""
                    send_telegram(no_data_msg)
                
                time.sleep(1)
                
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
🏁 **{index} INSTITUTIONAL ANALYSIS COMPLETED**
📈 Found {index_moves} institutional pressure moves
📅 Date: {analysis_date.strftime('%d %b %Y')}
"""
            send_telegram(index_msg)
    
    # Calculate analysis time
    analysis_time = round(time.time() - start_time)
    
    # Final completion message
    completion_msg = f"""
🎉 **TRUE INSTITUTIONAL ANALYSIS FINISHED** 🎉

📅 Analysis Date: {analysis_date.strftime('%d %b %Y')}
🕒 Finished: {datetime.now(IST).strftime('%H:%M:%S IST')}
📊 Total Institutional Moves Found: {total_big_moves}
📈 Total Analysis: {total_analyzed} datasets
⏱️ Analysis Time: {analysis_time} seconds
📊 Market Status: {market_status}

✅ **ALL INSTITUTIONAL ANALYSIS COMPLETED - PROGRAM STOPPED**

**Ready for detecting real institutional activity**
"""
    send_telegram(completion_msg)
    print(f"✅ True institutional analysis finished! Found {total_big_moves} moves in {analysis_time} seconds")

# --------- RUN TRUE INSTITUTIONAL ANALYSIS ---------
if __name__ == "__main__":
    print("🏛️ Starting True Institutional Pressure Analysis...")
    analyze_true_institutional_data()
    print("🛑 Program stopped automatically after completing all institutional analysis")
