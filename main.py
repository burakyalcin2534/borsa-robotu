import os
import random
import warnings
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests

# Kütüphane Değişikliği: pandas_ta yerine ta kullanıldı
import ta

# ML Kütüphaneleri
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Gelişmiş Modeller (GitHub'da varsa kullanır)
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

# ---------- TELEGRAM AYARLARI ----------
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

# ---------- HİSSE LİSTESİ (Aynen Korundu) ----------
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
    target_horizon: int = 1
    random_state: int = 42

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

# ---------- ÖZELLİK MÜHENDİSLİĞİ (ta ile yeniden yazıldı) ----------
class OmniFeatureEngineer:
    def calculate_supertrend(self, df, period=10, multiplier=3):
        high, low, close = df['high'], df['low'], df['close']
        atr = ta.volatility.average_true_range(high, low, close, window=period)
        hl2 = (high + low) / 2
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        direction = np.where(close > upperband.shift(1), 1, 0)
        direction = np.where(close < lowerband.shift(1), -1, direction)
        return pd.Series(direction, index=df.index)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()
        # Trend
        f['sma_10'] = ta.trend.sma_indicator(f['close'], window=10)
        f['sma_50'] = ta.trend.sma_indicator(f['close'], window=50)
        f['ema_20'] = ta.trend.ema_indicator(f['close'], window=20)
        f['macd'] = ta.trend.macd(f['close'])
        f['adx'] = ta.trend.adx(f['high'], f['low'], f['close'], window=14)
        
        # Ichimoku
        ichi = ta.trend.IchimokuIndicator(f['high'], f['low'], window1=9, window2=26, window3=52)
        f['ichi_cloud_dir'] = np.where(ichi.ichimoku_a() > ichi.ichimoku_b(), 1, -1)

        # Volatilite & Momentum
        f['atr'] = ta.volatility.average_true_range(f['high'], f['low'], f['close'], window=14)
        f['supertrend_dir'] = self.calculate_supertrend(f)
        f['rsi'] = ta.momentum.rsi(f['close'], window=14)
        f['cci'] = ta.trend.cci(f['high'], f['low'], f['close'], window=14)
        f['roc'] = ta.momentum.roc(f['close'], window=10)

        # Hacim
        f['obv'] = ta.volume.on_balance_volume(f['close'], f['volume'])
        f['mfi'] = ta.volume.money_flow_index(f['high'], f['low'], f['close'], f['volume'], window=14)
        f['cmf'] = ta.volume.chaikin_money_flow(f['high'], f['low'], f['close'], f['volume'], window=20)

        # Hedef
        horizon = 1
        future_return = f['close'].pct_change(horizon).shift(-horizon)
        f['target_dir'] = (future_return > 0).astype(int)
        f['target_ret'] = future_return
        return f

# ---------- AUTO-TUNER (ORİJİNAL KODDAN GERİ GETİRİLDİ) ----------
class ModelAutoTuner:
    """
    Her hisse için en iyi Random Forest parametrelerini bulur.
    """
    def tune_params(self, X_train, y_train):
        best_score = -1
        best_params = {'depth': 6, 'est': 120} # Varsayılan
        
        # Orijinal koddaki grid
        params_grid = [
            {'depth': 4, 'est': 100},
            {'depth': 6, 'est': 120},
            {'depth': 8, 'est': 150}
        ]
        
        # Veriyi böl (Hız için basit split)
        split = int(len(X_train) * 0.8)
        X_t, X_v = X_train[:split], X_train[split:]
        y_t, y_v = y_train[:split], y_train[split:]
        
        for p in params_grid:
            model = RandomForestClassifier(n_estimators=p['est'], max_depth=p['depth'], random_state=42, n_jobs=1)
            model.fit(X_t, y_t)
            score = accuracy_score(y_v, model.predict(X_v))
            if score > best_score:
                best_score = score
                best_params = p
        return best_params

# ---------- STRATEGY MANAGER (ORİJİNAL KODDAN GERİ GETİRİLDİ) ----------
class StrategyManager:
    @staticmethod
    def calculate_strategy(price, atr, conf, status):
        """
        ATR tabanlı Stop/Kar seviyeleri ve Kasa Yönetimi hesaplar.
        """
        if status != "AL":
            return 0, 0, 0
        
        # Stop & Kar-Al
        stop_loss = price - (2.0 * atr)
        take_profit = price + (3.0 * atr)
        
        # Pozisyon Büyüklüğü (Orijinal Mantık)
        base_allocation = 5.0 # %5
        conf_multiplier = (conf / 0.60)
        
        # Volatilite Cezası
        volatility_pct = (atr / price) * 100
        vol_penalty = 1.0
        if volatility_pct > 3.0: vol_penalty = 0.7
        if volatility_pct > 5.0: vol_penalty = 0.5
        
        suggested_allocation = base_allocation * conf_multiplier * vol_penalty
        suggested_allocation = min(suggested_allocation, 15.0)
        suggested_allocation = max(suggested_allocation, 1.0)
        
        return stop_loss, take_profit, suggested_allocation

# ---------- HYBRID MODEL TRAINER (Tam Sürüm) ----------
class HybridModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.imputers = {}
        self.scalers = {}
        self.feature_cols = []

    def _get_neural_network(self):
        base_mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', 
                               solver='adam', alpha=0.001, max_iter=400, random_state=42)
        return BaggingClassifier(estimator=base_mlp, n_estimators=5, random_state=42, n_jobs=1)

    def _get_tree_models(self, best_params):
        models = []
        seed = 42
        d = best_params['depth']
        e = best_params['est']
        
        if _HAS_XGB: models.append(('xgb', xgb.XGBClassifier(n_estimators=e, max_depth=d-2, learning_rate=0.04, random_state=seed, n_jobs=1)))
        if _HAS_LGB: models.append(('lgb', lgb.LGBMClassifier(n_estimators=e, max_depth=d-2, learning_rate=0.04, random_state=seed, verbose=-1, n_jobs=1)))
        models.append(('rf', RandomForestClassifier(n_estimators=e, max_depth=d, random_state=seed, n_jobs=1)))
        return models
    
    def _get_tree_regressors(self, best_params):
        models = []
        seed = 42
        d = best_params['depth']
        e = best_params['est']
        if _HAS_XGB: models.append(('xgb', xgb.XGBRegressor(n_estimators=e, max_depth=d-2, learning_rate=0.04, random_state=seed, n_jobs=1)))
        models.append(('rf', RandomForestRegressor(n_estimators=e, max_depth=d, random_state=seed, n_jobs=1)))
        return models

    def train_hybrid(self, X: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series):
        cols_to_drop = ['target_dir', 'target_ret', 'open', 'high', 'low', 'close', 'volume', 'adj close']
        self.feature_cols = [c for c in X.columns if c not in cols_to_drop]
        X_data = X[self.feature_cols]
        
        imp = SimpleImputer(strategy='median')
        scl = RobustScaler()
        X_proc = scl.fit_transform(imp.fit_transform(X_data))
        self.imputers['all'] = imp
        self.scalers['all'] = scl
        
        # --- AUTO-TUNING (Geri Geldi) ---
        tuner = ModelAutoTuner()
        # Eğitim verisinin %80'i üzerinde tune et
        split_tune = int(len(X_proc) * 0.8)
        best_params = tuner.tune_params(X_proc[:split_tune], y_cls.iloc[:split_tune])
        
        # 1. KARAR AĞACI MODELLERİ
        tree_estimators = self._get_tree_models(best_params)
        tree_voting = VotingClassifier(estimators=tree_estimators, voting='soft', n_jobs=1)
        tree_voting.fit(X_proc, y_cls)
        self.models['tree'] = tree_voting
        
        # 2. NÖRAL AĞ
        neural_net = self._get_neural_network()
        neural_net.fit(X_proc, y_cls)
        self.models['nn'] = neural_net
        
        # 3. REGRESYON
        reg_estimators = self._get_tree_regressors(best_params)
        voting_reg = VotingRegressor(estimators=reg_estimators, n_jobs=1)
        voting_reg.fit(X_proc, y_reg)
        self.models['reg'] = voting_reg

    def predict(self, X_raw: pd.DataFrame):
        X_ready = X_raw[self.feature_cols]
        X_proc = self.scalers['all'].transform(self.imputers['all'].transform(X_ready))
        
        p_tree = self.models['tree'].predict_proba(X_proc)[0][1]
        p_nn = self.models['nn'].predict_proba(X_proc)[0][1]
        final_prob = (p_tree * 0.55) + (p_nn * 0.45)
        pred_ret = self.models['reg'].predict(X_proc)[0]
        
        votes = {"TREE": "AL" if p_tree > 0.5 else "SAT", "NEURAL": "AL" if p_nn > 0.5 else "SAT"}
        return final_prob, pred_ret, votes

# ---------- AKILLI ANALİZ VE TARAMA ----------
def analyze_signal(votes, prob):
    agreement = (votes['TREE'] == votes['NEURAL'])
    raw_conf = abs(prob - 0.5) * 2
    if agreement: raw_conf += 0.2
    else: raw_conf *= 0.6
    final_conf = min(raw_conf, 0.99)
    status = "AL" if prob > 0.5 else "SAT"
    if final_conf < 0.45: status = "NÖTR"
    return status, final_conf

def omni_scan():
    tg_report = "🚀 *BIST 100 STRATEGIST V14 (Orijinal Mantık)* 🚀\n\n"
    print("Analiz Başlıyor (Auto-Tuner ve Risk Yönetimi Aktif)...")
    
    # --- ENDEKS ---
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
            p_prob, p_ret, _ = trainer.predict(pred_row)
            
            status = "YÜKSELİŞ" if p_prob > 0.5 else "DÜŞÜŞ"
            icon = "🟢" if p_prob > 0.5 else "🔴"
            tg_report += f"🌍 *XU100 Endeks*: {df_index['close'].iloc[-1]:.2f}\n"
            tg_report += f"Yön: {icon} {status} (Güven: %{p_prob*100:.1f})\n\n"
    except Exception as e: print(f"Endeks hatası: {e}")

    # --- HİSSELER ---
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
            
            # Auto-Tuner devreye giriyor
            trainer = HybridModelTrainer(CONFIG)
            trainer.train_hybrid(train_df[f_cols], train_df['target_dir'], train_df['target_ret'])
            
            pred_row = df_features.iloc[[-1]]
            p_prob, p_ret, votes = trainer.predict(pred_row)
            
            status, conf = analyze_signal(votes, p_prob)
            
            if status == "AL" and conf > 0.55:
                price = df['close'].iloc[-1]
                atr = pred_row['atr'].values[0]
                
                # Orijinal Strategy Manager hesaplaması
                stop, tp, alloc = StrategyManager.calculate_strategy(price, atr, conf, status)
                
                results.append({
                    "Hisse": ticker.replace('.IS', ''),
                    "Fiyat": price,
                    "Güven": conf,
                    "Hedef": p_ret,
                    "Stop": stop,
                    "KarAl": tp,
                    "Kasa": alloc # Kasa yönetimi geri geldi
                })
        except: continue

    if results:
        results = sorted(results, key=lambda x: x['Güven'], reverse=True)
        tg_report += "💎 *GÜÇLÜ AL SİNYALLERİ* 💎\n"
        tg_report += f"{'HİSSE':<8} {'FİYAT':<8} {'GÜVEN':<6} {'HEDEF':<6} {'KASA%'}\n"
        tg_report += "-"*40 + "\n"
        
        for row in results[:15]:
            tg_report += f"*{row['Hisse']}*: {row['Fiyat']:.2f} | %{row['Güven']*100:.0f} | %{row['Hedef']*100:.1f} | %{row['Kasa']:.1f}\n"
        
        tg_report += "\nℹ️ *Kasa%*: Risk ve volatiliteye göre portföyden önerilen pay.\n"
        tg_report += "_Yatırım tavsiyesi değildir._"
    else:
        tg_report += "⚠️ Bugün güçlü sinyal bulunamadı."

    send_telegram(tg_report)
    print("Bitti.")

if __name__ == "__main__":
    omni_scan()
