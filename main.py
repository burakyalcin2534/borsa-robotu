import os
import random
import warnings
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests

# YENİ KÜTÜPHANE: ta
import ta

# ML Kütüphaneleri
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Gelişmiş Modeller (Varsa çalışır, yoksa hata vermez)
try:
    import yfinance as yf
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    _HAS_XGB = True
    _HAS_LGB = True
    _HAS_CAT = True
except ImportError:
    _HAS_XGB = False
    _HAS_LGB = False
    _HAS_CAT = False

# ---------- TELEGRAM FONKSİYONU ----------
def send_telegram(message):
    try:
        token = os.environ.get("TELEGRAM_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id: return

        if len(message) > 4000:
            parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
            for part in parts:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                requests.post(url, data={"chat_id": chat_id, "text": part, "parse_mode": "Markdown"})
                time.sleep(1)
        else:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
    except: pass

# ---------- HİSSE LİSTESİ ----------
BIST_100_LISTESI = [
    "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALTNY.IS", "ANSGR.IS", "ARCLK.IS", "ASELS.IS",
    "ASTOR.IS", "BALSU.IS", "BIMAS.IS", "BINHO.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS", "BTCIM.IS", "CANTE.IS", "CCOLA.IS",
    "CLEBI.IS", "CWENE.IS", "DAPGM.IS", "DOAS.IS", "DOHOL.IS", "DSTKF.IS", "ECILC.IS", "EFOR.IS", "EGEEN.IS", "EKGYO.IS",
    "ENERY.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS", "EUPWR.IS", "FENER.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS",
    "GLRMK.IS", "GRSEL.IS", "GRTHO.IS", "GSRAY.IS", "GUBRF.IS", "HALKB.IS", "HEKTS.IS", "IEYHO.IS", "IPEKE.IS", "ISCTR.IS",
    "ISMEN.IS", "KCAER.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "KTLEV.IS", "KUYAS.IS", "MAGEN.IS",
    "MAVI.IS", "MGROS.IS", "MIATK.IS", "MPARK.IS", "OBAMS.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PATEK.IS", "PASEU.IS",
    "PETKM.IS", "PGSUS.IS", "RALYH.IS", "REEDR.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "SKBNK.IS", "SOKM.IS", "TABGD.IS",
    "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TSPOR.IS", "TUKAS.IS", "TTKOM.IS", "TTRAK.IS",
    "TUPRS.IS", "TUREX.IS", "TURSG.IS", "ULKER.IS", "VAKBN.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "ZOREN.IS"
]

@dataclass
class ModelConfig:
    target_horizon: int = 1

CONFIG = ModelConfig()

# ---------- VERİ YÖNETİMİ ----------
class AdvancedDataManager:
    def download_data(self, ticker: str) -> pd.DataFrame:
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower().strip() for c in df.columns]
            if 'adj close' in df.columns: df['close'] = df['adj close']
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in required): return pd.DataFrame()
            if 'volume' in df.columns: df['volume'] = df['volume'].replace(0, np.nan).fillna(method='ffill')
            return df
        except: return pd.DataFrame()

# ---------- ÖZELLİK MÜHENDİSLİĞİ (ta Kütüphanesi ile) ----------
class OmniFeatureEngineer:
    def calculate_supertrend(self, df, period=10, multiplier=3):
        # SuperTrend ta kütüphanesinde olmadığı için manuel hesaplıyoruz
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ATR hesapla
        atr = ta.volatility.average_true_range(high, low, close, window=period)
        
        hl2 = (high + low) / 2
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        # Basitleştirilmiş yön
        direction = np.where(close > upperband.shift(1), 1, 0)
        direction = np.where(close < lowerband.shift(1), -1, direction)
        return pd.Series(direction, index=df.index)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()
        
        # 1. Trend İndikatörleri (ta library)
        f['sma_10'] = ta.trend.sma_indicator(f['close'], window=10)
        f['sma_50'] = ta.trend.sma_indicator(f['close'], window=50)
        f['ema_20'] = ta.trend.ema_indicator(f['close'], window=20)
        
        # MACD
        f['macd'] = ta.trend.macd(f['close'])
        f['macd_signal'] = ta.trend.macd_signal(f['close'])
        
        # ADX (Geri Geldi!)
        f['adx'] = ta.trend.adx(f['high'], f['low'], f['close'], window=14)
        
        # Ichimoku (Geri Geldi!)
        ichi = ta.trend.IchimokuIndicator(f['high'], f['low'], window1=9, window2=26, window3=52)
        f['ichi_a'] = ichi.ichimoku_a()
        f['ichi_b'] = ichi.ichimoku_b()
        f['ichi_cloud_dir'] = np.where(f['ichi_a'] > f['ichi_b'], 1, -1)

        # 2. Volatilite
        f['atr'] = ta.volatility.average_true_range(f['high'], f['low'], f['close'], window=14)
        
        # SuperTrend (Özel Fonksiyon)
        f['supertrend_dir'] = self.calculate_supertrend(f)

        # 3. Momentum
        f['rsi'] = ta.momentum.rsi(f['close'], window=14)
        f['cci'] = ta.trend.cci(f['high'], f['low'], f['close'], window=14)
        f['roc'] = ta.momentum.roc(f['close'], window=10) # Geri Geldi!

        # 4. Hacim
        f['obv'] = ta.volume.on_balance_volume(f['close'], f['volume']) # Geri Geldi!
        f['mfi'] = ta.volume.money_flow_index(f['high'], f['low'], f['close'], f['volume'], window=14) # Geri Geldi!
        f['cmf'] = ta.volume.chaikin_money_flow(f['high'], f['low'], f['close'], f['volume'], window=20) # Geri Geldi!

        # Hedef Belirleme
        horizon = 1
        future_return = f['close'].pct_change(horizon).shift(-horizon)
        f['target_dir'] = (future_return > 0).astype(int)
        f['target_ret'] = future_return
        
        return f

# ---------- EĞİTİM VE TAHMİN ----------
class HybridModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.imputers = {}
        self.scalers = {}
        self.feature_cols = []

    def train_hybrid(self, X: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series):
        cols_to_drop = ['target_dir', 'target_ret', 'open', 'high', 'low', 'close', 'volume', 'adj close']
        self.feature_cols = [c for c in X.columns if c not in cols_to_drop]
        X_data = X[self.feature_cols]
        
        imp = SimpleImputer(strategy='median')
        scl = RobustScaler()
        X_proc = scl.fit_transform(imp.fit_transform(X_data))
        self.imputers['all'] = imp
        self.scalers['all'] = scl
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
        rf.fit(X_proc, y_cls)
        self.models['tree'] = rf
        
        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
        rf_reg.fit(X_proc, y_reg)
        self.models['reg'] = rf_reg

    def predict(self, X_raw: pd.DataFrame):
        X_ready = X_raw[self.feature_cols]
        X_proc = self.scalers['all'].transform(self.imputers['all'].transform(X_ready))
        p_prob = self.models['tree'].predict_proba(X_proc)[0][1]
        p_ret = self.models['reg'].predict(X_proc)[0]
        return p_prob, p_ret

# ---------- ANA TARAMA ----------
def omni_scan():
    tg_report = "🚀 *BIST 100 AI ANALİZ RAPORU (Full İndikatör)* 🚀\n\n"
    print("Analiz Başlıyor...")
    
    # --- ENDEKS ANALİZİ ---
    try:
        dm = AdvancedDataManager()
        df_index = dm.download_data("XU100.IS")
        if not df_index.empty:
            fe = OmniFeatureEngineer()
            df_f = fe.create_features(df_index)
            f_cols = [c for c in df_f.columns if 'target' not in c]
            train_df = df_f.dropna(subset=f_cols + ['target_dir', 'target_ret'])
            
            trainer = HybridModelTrainer(CONFIG)
            trainer.train_hybrid(train_df[f_cols], train_df['target_dir'], train_df['target_ret'])
            
            pred_row = df_f.iloc[[-1]]
            p_prob, p_ret = trainer.predict(pred_row)
            
            status = "YÜKSELİŞ" if p_prob > 0.5 else "DÜŞÜŞ"
            icon = "🟢" if p_prob > 0.5 else "🔴"
            tg_report += f"🌍 *XU100 Endeks*: {df_index['close'].iloc[-1]:.2f}\n"
            tg_report += f"Yön: {icon} {status} (Güven: %{p_prob*100:.1f})\n\n"
    except Exception as e: print(f"Endeks hatası: {e}")

    # --- HİSSE TARAMASI ---
    results = []
    print("Hisseler taranıyor...")
    for ticker in BIST_100_LISTESI:
        try:
            df = dm.download_data(ticker)
            if df.empty or len(df) < 60: continue
            
            fe = OmniFeatureEngineer()
            df_features = fe.create_features(df)
            f_cols = [c for c in df_features.columns if 'target' not in c]
            train_df = df_features.dropna(subset=f_cols + ['target_dir', 'target_ret'])
            if len(train_df) < 50: continue
            
            trainer = HybridModelTrainer(CONFIG)
            trainer.train_hybrid(train_df[f_cols], train_df['target_dir'], train_df['target_ret'])
            
            pred_row = df_features.iloc[[-1]]
            p_prob, p_ret = trainer.predict(pred_row)
            
            if p_prob > 0.55:
                price = df['close'].iloc[-1]
                atr = pred_row['atr'].values[0]
                stop = price - (2 * atr)
                tp = price + (3 * atr)
                
                results.append({
                    "Hisse": ticker.replace('.IS', ''),
                    "Fiyat": price,
                    "Güven": p_prob,
                    "Hedef": p_ret,
                    "Stop": stop,
                    "KarAl": tp
                })
        except: continue

    if results:
        results = sorted(results, key=lambda x: x['Güven'], reverse=True)
        tg_report += "💎 *GÜÇLÜ AL SİNYALLERİ* 💎\n"
        tg_report += "Hisse | Fiyat | Güven | Hedef\n"
        tg_report += "-"*30 + "\n"
        for row in results[:15]:
            tg_report += f"*{row['Hisse']}*: {row['Fiyat']:.2f} | %{row['Güven']*100:.0f} | %{row['Hedef']*100:.1f}\n"
        tg_report += "\n_Yatırım tavsiyesi değildir._"
    else:
        tg_report += "⚠️ Bugün güçlü sinyal bulunamadı."

    send_telegram(tg_report)
    print("Bitti.")

if __name__ == "__main__":
    omni_scan()
