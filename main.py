import os
import random
import warnings
import logging
import time
from datetime import datetime, timedelta
import pytz
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests # Telegram için eklendi

# ML Libs
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# Gelişmiş Modeller (Yüklü değilse hata vermesin diye try-except blokları)
try:
    import yfinance as yf
    import pandas_ta as ta
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_XGB = True
    _HAS_LGB = True
    _HAS_CAT = True
except ImportError:
    _HAS_XGB = False
    _HAS_LGB = False
    _HAS_CAT = False

# ---------- TELEGRAM AYARLARI ----------
def send_telegram(message):
    try:
        token = os.environ.get("TELEGRAM_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        if not token or not chat_id:
            print("Telegram Token veya Chat ID bulunamadı!")
            return

        # Mesaj çok uzunsa parça parça gönder
        if len(message) > 4000:
            parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
            for part in parts:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = {"chat_id": chat_id, "text": part, "parse_mode": "Markdown"}
                requests.post(url, data=data)
                time.sleep(1)
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
            requests.post(url, data=data)
            
    except Exception as e:
        print(f"Telegram gönderme hatası: {e}")

# ---------- 0. HİSSE LİSTESİ ----------
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

# ---------- 1. KESİN DETERMINISM ----------
def set_global_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

@dataclass
class ModelConfig:
    ticker: str = "XU100.IS"
    period: str = "4y" 
    target_horizon: int = 1
    random_state: int = 42
    rf_depth: int = 6
    rf_estimators: int = 120

CONFIG = ModelConfig()
set_global_determinism(CONFIG.random_state)

# ---------- 2. VERİ YÖNETİMİ ----------
class AdvancedDataManager:
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def download_data(self, ticker: str) -> pd.DataFrame:
        try:
            # Basit veri indirme (Hata riskini azaltmak için sadeleştirildi)
            df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
            
            # Sütun isimlerini düzelt
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Eğer 'adj close' varsa onu 'close' yap, yoksa normal 'close' kullan
            if 'adj close' in df.columns:
                df['close'] = df['adj close']
            
            required = ['open', 'high', 'low', 'close', 'volume']
            # Eksik sütun kontrolü
            if not all(col in df.columns for col in required):
                return pd.DataFrame()
            
            df = df[required] # Sadece gerekli sütunları al
            
            if len(df) < 100: return pd.DataFrame() 
            
            if 'volume' in df.columns: 
                df['volume'] = df['volume'].replace(0, np.nan).fillna(method='ffill')
            return df
        except Exception as e:
            return pd.DataFrame()

# ---------- 3. OMNI-FEATURE ENGINE ----------
class OmniFeatureEngineer:
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()
        # Temel İndikatörler
        f['sma_10'] = ta.sma(f['close'], length=10)
        f['ema_20'] = ta.ema(f['close'], length=20)
        f['atr'] = ta.atr(f['high'], f['low'], f['close'], length=14)
        
        # MACD
        try:
            macd = ta.macd(f['close'])
            if macd is not None:
                f['macd'] = macd['MACD_12_26_9']
        except: pass
        
        # SuperTrend
        try:
            st = ta.supertrend(f['high'], f['low'], f['close'], length=10, multiplier=3)
            if st is not None:
                st_dir_col = [c for c in st.columns if 'SUPERTd' in c][0]
                f['supertrend_dir'] = st[st_dir_col]
        except: pass

        f['rsi'] = ta.rsi(f['close'], length=14)
        f['cci'] = ta.cci(f['high'], f['low'], f['close'], length=14)
        
        # Hedef Belirleme
        horizon = self.config.target_horizon
        future_return = f['close'].pct_change(horizon).shift(-horizon)
        f['target_dir'] = (future_return > 0).astype(int)
        f['target_ret'] = future_return
        
        return f

# ---------- 4. MODEL AUTO-TUNER ----------
class ModelAutoTuner:
    def tune_params(self, X_train, y_train):
        # Basit parametreler (Hız için sabitlendi)
        return {'depth': 6, 'est': 120}

# ---------- 5. MODEL EĞİTİCİ ----------
class HybridModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.imputers = {}
        self.scalers = {}
        self.feature_cols = []
        self.metrics = {"accuracy": 0.0}

    def train_hybrid(self, X: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series):
        # Gereksiz sütunları at
        cols_to_drop = ['target_dir', 'target_ret', 'open', 'high', 'low', 'close', 'volume', 'adj close']
        train_cols = [c for c in X.columns if c not in cols_to_drop]
        self.feature_cols = train_cols
        
        X_data = X[train_cols]
        
        # Veri Temizleme
        imp = SimpleImputer(strategy='median')
        scl = RobustScaler()
        X_proc = scl.fit_transform(imp.fit_transform(X_data))
        
        self.imputers['all'] = imp
        self.scalers['all'] = scl
        
        # Eğitim
        seed = self.config.random_state
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed, n_jobs=1)
        rf.fit(X_proc, y_cls)
        self.models['tree'] = rf
        
        # Doğruluk
        y_pred = rf.predict(X_proc)
        self.metrics['accuracy'] = accuracy_score(y_cls, y_pred)
        
        # Regresyon
        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed, n_jobs=1)
        rf_reg.fit(X_proc, y_reg)
        self.models['reg'] = rf_reg

    def predict(self, X_raw: pd.DataFrame):
        X_ready = X_raw[self.feature_cols]
        X_proc = self.scalers['all'].transform(self.imputers['all'].transform(X_ready))
        
        p_tree = self.models['tree'].predict_proba(X_proc)[0][1]
        pred_ret = self.models['reg'].predict(X_proc)[0]
        
        return 1 if p_tree > 0.5 else 0, p_tree, pred_ret, {"TREE": "AL"}

# ---------- 6. RISK MANAGER ----------
def calculate_strategy(price, atr, conf, status):
    if status != "AL": return 0, 0, 0
    stop_loss = price - (2.0 * atr)
    take_profit = price + (3.0 * atr)
    base_alloc = 5.0
    if conf > 0.7: base_alloc = 10.0
    return stop_loss, take_profit, base_alloc

# ---------- 7. MAIN SCANNER ----------
def omni_scan():
    # Mesaj metnini biriktireceğimiz değişken
    tg_report = "🚀 *BIST 100 AI ANALİZ RAPORU* 🚀\n\n"
    
    print("Analiz Başlıyor...")
    
    # --- ENDEKS ---
    try:
        index_config = ModelConfig(ticker="XU100.IS")
        dm = AdvancedDataManager(index_config)
        df_index = dm.download_data("XU100.IS")
        
        if not df_index.empty:
            fe = OmniFeatureEngineer(index_config)
            df_f = fe.create_features(df_index)
            f_cols = [c for c in df_f.columns if 'target' not in c]
            train_df = df_f.dropna(subset=f_cols + ['target_dir', 'target_ret'])
            
            trainer = HybridModelTrainer(index_config)
            trainer.train_hybrid(train_df[f_cols], train_df['target_dir'], train_df['target_ret'])
            
            pred_row = df_f.iloc[[-1]]
            p_dir, p_prob, p_ret, _ = trainer.predict(pred_row)
            
            last_price = df_index['close'].iloc[-1]
            status = "YÜKSELİŞ" if p_prob > 0.5 else "DÜŞÜŞ"
            icon = "🟢" if p_prob > 0.5 else "🔴"
            
            tg_report += f"🌍 *XU100 Endeks Durumu*\n"
            tg_report += f"Fiyat: {last_price:.2f}\n"
            tg_report += f"Yön: {icon} {status} (Güven: %{p_prob*100:.1f})\n"
            tg_report += f"Hedef Getiri: %{p_ret*100:+.2f}\n\n"
            print("Endeks analizi bitti.")
    except Exception as e:
        print(f"Endeks hatası: {e}")

    # --- HİSSELER ---
    results = []
    print("Hisseler taranıyor (Bu işlem 3-5 dk sürebilir)...")
    
    for ticker in BIST_100_LISTESI:
        try:
            CONFIG.ticker = ticker
            dm = AdvancedDataManager(CONFIG)
            df = dm.download_data(ticker)
            if df.empty: continue
            
            fe = OmniFeatureEngineer(CONFIG)
            df_features = fe.create_features(df)
            
            feature_cols = [c for c in df_features.columns if 'target' not in c]
            train_df = df_features.dropna(subset=feature_cols + ['target_dir', 'target_ret'])
            if len(train_df) < 50: continue
            
            pred_row = df_features.iloc[[-1]]
            price = pred_row['close'].values[0]
            atr = pred_row['atr'].values[0] if 'atr' in pred_row else (price * 0.05)
            
            trainer = HybridModelTrainer(CONFIG)
            trainer.train_hybrid(train_df[feature_cols], train_df['target_dir'], train_df['target_ret'])
            
            pred_dir, prob, pred_ret, _ = trainer.predict(pred_row)
            
            status = "AL" if prob > 0.55 else "SAT" # Eşik değeri 0.55
            
            stop, tp, alloc = calculate_strategy(price, atr, prob, status)
            
            if status == "AL":
                results.append({
                    "Hisse": ticker.replace('.IS', ''),
                    "Fiyat": price,
                    "Güven": prob,
                    "Hedef": pred_ret,
                    "Stop": stop,
                    "KarAl": tp
                })
        except: continue

    # --- RAPORLAMA ---
    if results:
        # Güvene göre sırala
        results = sorted(results, key=lambda x: x['Güven'], reverse=True)
        
        tg_report += "💎 *EN GÜÇLÜ AL SİNYALLERİ* 💎\n"
        tg_report += "Hisse | Fiyat | Güven | Hedef\n"
        tg_report += "-"*30 + "\n"
        
        for row in results[:15]: # İlk 15 hisseyi gönder
            guven_str = f"%{row['Güven']*100:.0f}"
            hedef_str = f"%{row['Hedef']*100:.1f}"
            satir = f"*{row['Hisse']}* : {row['Fiyat']:.2f}TL | {guven_str} | {hedef_str}\n"
            tg_report += satir
            
        tg_report += "\n_Not: Stop ve Kar-Al seviyeleri ATR tabanlıdır. YTD._"
    else:
        tg_report += "⚠️ Bugün güçlü bir AL sinyali bulunamadı."

    print("Telegram'a gönderiliyor...")
    send_telegram(tg_report)
    print("Tamamlandı.")

if __name__ == "__main__":
    omni_scan()