import os
import random
import warnings
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests

# ML Kütüphaneleri
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Gelişmiş Modeller (Varsa kullanır, yoksa hata vermez)
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

# ---------- TEKNİK ANALİZ MOTORU (Kütüphanesiz - Saf Matematik) ----------
class TechnicalAnalysis:
    @staticmethod
    def get_sma(series, period):
        return series.rolling(window=period).mean()

    @staticmethod
    def get_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def get_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def get_macd(close):
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line

    @staticmethod
    def get_cci(high, low, close, period=14):
        tp = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = (tp - sma).abs().rolling(period).mean()
        return (tp - sma) / (0.015 * mad)

    @staticmethod
    def get_supertrend(high, low, close, period=10, multiplier=3):
        # Basitleştirilmiş SuperTrend Hesaplaması
        atr = TechnicalAnalysis.get_atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        # Basit hesaplama (Veri çerçevesi üzerinde vektörel)
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        # Yön tahmini için basit bir mantık: Kapanış üst bandı kırarsa AL, altı kırarsa SAT
        # (Tam SuperTrend döngüsü çok kompleks olduğu için ML modeline giren feature olarak bu yeterlidir)
        direction = np.where(close > upperband.shift(1), 1, 0)
        direction = np.where(close < lowerband.shift(1), -1, direction)
        return pd.Series(direction, index=close.index)

# ---------- HİSSE LİSTESİ ----------
BIST_100_LISTESI = [
    "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALTNY.IS", "ANSGR.IS", "ARCLK.IS", "ASELS.IS",
    "ASTOR.IS", "BALSU.IS", "BIMAS.IS", "BINHO.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS", "BTCIM.IS", "CANTE.IS", "CCOLA.IS",
    "CLEBI.IS", "CWENE.IS", "DAPGM.IS", "DOAS.IS", "DOHOL.IS", "DSTKF.IS", "ECILC.IS", "EFORC.IS", "EGEEN.IS", "EKGYO.IS",
    "ENERY.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS", "EUPWR.IS", "FENER.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS",
    "GLRMK.IS", "GRSEL.IS", "GRTHO.IS", "GSRAY.IS", "GUBRF.IS", "HALKB.IS", "HEKTS.IS", "IEYHO.IS", "IPEKE.IS", "ISCTR.IS",
    "ISMEN.IS", "KCAER.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "KTLEV.IS", "KUYAS.IS", "MAGEN.IS",
    "MAVI.IS", "MGROS.IS", "MIATK.IS", "MPARK.IS", "OBAMS.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PATEK.IS", "PASEU.IS",
    "PETKM.IS", "PGSUS.IS", "RALYH.IS", "REEDR.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "SKBNK.IS", "SOKM.IS", "TABGD.IS",
    "TAVH.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TSPOR.IS", "TUKAS.IS", "TTKOM.IS", "TTRAK.IS",
    "TUPRS.IS", "TUREX.IS", "TURSG.IS", "ULKER.IS", "VAKBN.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "ZOREN.IS"
]

@dataclass
class ModelConfig:
    ticker: str = "XU100.IS"
    target_horizon: int = 1
    random_state: int = 42

CONFIG = ModelConfig()

# ---------- VERİ VE ÖZELLİK MÜHENDİSLİĞİ ----------
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

class OmniFeatureEngineer:
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()
        
        # TA Kütüphanesi yerine kendi yazdığımız fonksiyonları kullanıyoruz
        f['sma_10'] = TechnicalAnalysis.get_sma(f['close'], 10)
        f['ema_20'] = TechnicalAnalysis.get_ema(f['close'], 20)
        f['atr'] = TechnicalAnalysis.get_atr(f['high'], f['low'], f['close'], 14)
        f['macd'] = TechnicalAnalysis.get_macd(f['close'])
        f['rsi'] = TechnicalAnalysis.get_rsi(f['close'], 14)
        f['cci'] = TechnicalAnalysis.get_cci(f['high'], f['low'], f['close'], 14)
        f['supertrend_dir'] = TechnicalAnalysis.get_supertrend(f['high'], f['low'], f['close'])
        
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
        
        # Model Eğitimi
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

# ---------- ANA TARAMA FONKSİYONU ----------
def omni_scan():
    tg_report = "🚀 *BIST 100 AI ANALİZ RAPORU* 🚀\n\n"
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
            
            if p_prob > 0.55: # Eşik değer
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

    # --- RAPORLAMA ---
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
