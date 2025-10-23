import pandas as pd
import numpy as np

def sma(close, length=50):
    """Simple Moving Average (SMA)"""
    return close.rolling(window=length).mean()

def ema(close, length=20):
    """Exponential Moving Average (EMA)"""
    return close.ewm(span=length, adjust=False).mean()

def rsi(close, length=14):
    """Relative Strength Index (RSI)"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(high, low, close, length=14):
    """Average True Range (ATR)"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close}).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def macd(close, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence (MACD)"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def adx(high, low, close, length=14):
    """Average Directional Movement Index (ADX)"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.rolling(length).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha = 1/length).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/length).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (length - 1)) + dx) / length
    adx_smooth = adx.ewm(alpha = 1/length).mean()
    return adx_smooth, plus_di, minus_di
