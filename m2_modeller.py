"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi Modernizasyonu
Modül 2 – 4 ML Modeli: RF · XGBoost · LightGBM · CatBoost
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

- Multi-output: 6 aylık tahmini tek seferde üretir
- Optuna ile hiperparametre optimizasyonu
- Test setinde MAE · RMSE · MAPE karşılaştırması
- Geleneksel yöntemlerle de karşılaştırma
- A/B sınıfı → ML  |  C/Z sınıfı → Hareketli Ortalama
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import optuna
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from m1_veri import FEATURE_COLS, TARGET, PARCA, SPLIT, metrik, geleneksel_tahmin, parca_verisi

# Opsiyonel kütüphaneler
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False

try:
    from catboost import CatBoostRegressor
    CAT_OK = True
except ImportError:
    CAT_OK = False

N_AY = 6  # Tahmin ufku


# ══════════════════════════════════════════════════════════════════
# YARDIMCILAR
# ══════════════════════════════════════════════════════════════════

def _get_X(df, feat):
    X = np.zeros((len(df), len(feat)))
    for i, c in enumerate(feat):
        if c in df.columns:
            X[:, i] = df[c].fillna(0).values
    return X


def _cv_rmse(estimator, X, y, n_splits=4):
    """Zaman serisi CV RMSE."""
    n = len(X)
    fold = max(n // (n_splits + 1), 1)
    scores = []
    for i in range(1, n_splits + 1):
        tr_end  = i * fold
        val_end = min(tr_end + fold, n)
        if val_end <= tr_end or tr_end >= n:
            continue
        try:
            m = type(estimator)(**estimator.get_params())
            m.fit(X[:tr_end], y[:tr_end])
            pred = m.predict(X[tr_end:val_end])
            scores.append(float(np.sqrt(mean_squared_error(
                y[tr_end:val_end].flatten(), pred.flatten()
            ))))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 1e9


def _multi_hedef(df, n_ay=N_AY):
    """Multi-output için t+1 … t+n_ay hedef sütunları ekler."""
    parcalar = []
    for pid, grp in df.groupby(PARCA):
        grp  = grp.sort_values("Tarih").copy()
        talep = grp[TARGET].values
        for i in range(1, n_ay + 1):
            shifted = np.roll(talep, -i).astype(float)
            shifted[-i:] = np.nan
            grp[f"h_{i}"] = shifted
        parcalar.append(grp)
    result = pd.concat(parcalar)
    hedef_cols = [f"h_{i}" for i in range(1, n_ay + 1)]
    return result.dropna(subset=hedef_cols), hedef_cols


# ══════════════════════════════════════════════════════════════════
# MODEL EĞİTİMİ (SEGMENT BAZLI)
# ══════════════════════════════════════════════════════════════════

def segment_modelleri_egit(train_df, test_df, n_trials=30):
    """
    Her segment için 4 ML modeli eğitir.
    C/Z segmentleri geleneksel yöntemle işaretlenir.
    """
    train_mo, hedef_cols = _multi_hedef(train_df)
    feat = [c for c in FEATURE_COLS if c in train_mo.columns]

    segmentler = sorted(train_df["Segment"].dropna().unique().tolist()) \
                 if "Segment" in train_df.columns else ["ALL"]

    sonuclar = {}

    for seg in segmentler:
        abc_s = seg[0] if len(seg) > 0 else "C"
        xyz_s = seg[1] if len(seg) > 1 else "Z"

        # C/Z → geleneksel
        if abc_s == "C" or xyz_s == "Z":
            print(f"  [{seg}] → Geleneksel (Hareketli Ortalama)")
            sonuclar[seg] = {"tip": "geleneksel", "sampiyon": "Hareketli Ort.",
                              "feat": feat, "hedef_cols": hedef_cols}
            continue

        # Train verisi
        tr_mo = train_mo[train_mo["Segment"] == seg] if "Segment" in train_mo.columns else train_mo
        te    = test_df[test_df["Segment"] == seg]   if "Segment" in test_df.columns  else test_df

        if len(tr_mo) < 20:
            sonuclar[seg] = {"tip": "geleneksel", "sampiyon": "Hareketli Ort.",
                              "feat": feat, "hedef_cols": hedef_cols}
            continue

        X_tr = _get_X(tr_mo, feat)
        y_tr = tr_mo[hedef_cols].fillna(0).values
        X_te = _get_X(te, feat)
        y_te = te[TARGET].values

        modeller  = {}
        metrikler = {}

        # ── RF ─────────────────────────────────────────────────
        print(f"  [{seg}] RF...")
        def rf_obj(trial):
            p = {"n_estimators": trial.suggest_int("n_estimators", 50, 200),
                 "max_depth":    trial.suggest_int("max_depth", 3, 12),
                 "max_features": trial.suggest_float("max_features", 0.3, 1.0),
                 "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                 "random_state": 42, "n_jobs": -1}
            return _cv_rmse(RandomForestRegressor(**p), X_tr, y_tr[:, 0])

        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(rf_obj, n_trials=n_trials, show_progress_bar=False)
        p = {**s.best_params, "random_state": 42, "n_jobs": -1}
        m = MultiOutputRegressor(RandomForestRegressor(**p))
        m.fit(X_tr, y_tr)
        pred = np.maximum(m.predict(X_te), 0)
        modeller["RF"]  = m
        metrikler["RF"] = metrik(y_te, pred[:, 0] if pred.ndim > 1 else pred, "RF")

        # ── XGBoost ────────────────────────────────────────────
        print(f"  [{seg}] XGBoost...")
        def xgb_obj(trial):
            p = {"n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                 "max_depth":     trial.suggest_int("max_depth", 2, 8),
                 "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
                 "random_state":  42}
            base = xgb.XGBRegressor(**p, verbosity=0, n_jobs=-1) if XGB_OK \
                   else GradientBoostingRegressor(**{k: v for k, v in p.items()
                                                     if k != "subsample"})
            return _cv_rmse(base, X_tr, y_tr[:, 0])

        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=False)
        if XGB_OK:
            base = xgb.XGBRegressor(**s.best_params, random_state=42, verbosity=0, n_jobs=-1)
        else:
            p2 = {k: v for k, v in s.best_params.items()
                  if k in ["n_estimators", "max_depth", "learning_rate"]}
            base = GradientBoostingRegressor(**p2, random_state=42)
        m = MultiOutputRegressor(base)
        m.fit(X_tr, y_tr)
        pred = np.maximum(m.predict(X_te), 0)
        modeller["XGBoost"]  = m
        metrikler["XGBoost"] = metrik(y_te, pred[:, 0] if pred.ndim > 1 else pred, "XGBoost")

        # ── LightGBM ───────────────────────────────────────────
        print(f"  [{seg}] LightGBM...")
        def lgb_obj(trial):
            p = {"n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                 "max_depth":     trial.suggest_int("max_depth", 2, 10),
                 "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 "num_leaves":    trial.suggest_int("num_leaves", 15, 63),
                 "random_state":  42}
            base = lgb.LGBMRegressor(**p, verbose=-1, n_jobs=-1) if LGB_OK \
                   else GradientBoostingRegressor(
                       n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                       learning_rate=p["learning_rate"], random_state=42)
            return _cv_rmse(base, X_tr, y_tr[:, 0])

        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(lgb_obj, n_trials=n_trials, show_progress_bar=False)
        if LGB_OK:
            base = lgb.LGBMRegressor(**s.best_params, random_state=42, verbose=-1, n_jobs=-1)
        else:
            p2 = {k: v for k, v in s.best_params.items()
                  if k in ["n_estimators", "max_depth", "learning_rate"]}
            base = GradientBoostingRegressor(**p2, random_state=42)
        m = MultiOutputRegressor(base)
        m.fit(X_tr, y_tr)
        pred = np.maximum(m.predict(X_te), 0)
        modeller["LightGBM"]  = m
        metrikler["LightGBM"] = metrik(y_te, pred[:, 0] if pred.ndim > 1 else pred, "LightGBM")

        # ── CatBoost ───────────────────────────────────────────
        print(f"  [{seg}] CatBoost...")
        def cat_obj(trial):
            p = {"iterations":    trial.suggest_int("iterations", 50, 300),
                 "depth":         trial.suggest_int("depth", 2, 8),
                 "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 "random_seed":   42}
            base = CatBoostRegressor(**p, verbose=0) if CAT_OK \
                   else GradientBoostingRegressor(
                       n_estimators=p["iterations"], max_depth=p["depth"],
                       learning_rate=p["learning_rate"], random_state=42)
            return _cv_rmse(base, X_tr, y_tr[:, 0])

        s = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(cat_obj, n_trials=n_trials, show_progress_bar=False)
        if CAT_OK:
            base = CatBoostRegressor(**s.best_params, random_seed=42, verbose=0)
        else:
            p2 = {k: v for k, v in s.best_params.items()
                  if k in ["iterations", "depth", "learning_rate"]}
            base = GradientBoostingRegressor(
                n_estimators=p2.get("iterations", 100),
                max_depth=p2.get("depth", 3),
                learning_rate=p2.get("learning_rate", 0.1),
                random_state=42)
        m = MultiOutputRegressor(base)
        m.fit(X_tr, y_tr)
        pred = np.maximum(m.predict(X_te), 0)
        modeller["CatBoost"]  = m
        metrikler["CatBoost"] = metrik(y_te, pred[:, 0] if pred.ndim > 1 else pred, "CatBoost")

        # Şampiyon: en düşük RMSE
        sampiyon = min(metrikler, key=lambda k: metrikler[k]["RMSE"])
        rmse_str = " | ".join([f"{k}={metrikler[k]['RMSE']:,.0f}" for k in metrikler])
        print(f"  [{seg}] {rmse_str} → {sampiyon}")

        sonuclar[seg] = {
            "tip":         "ml",
            "modeller":    modeller,
            "metrikler":   metrikler,
            "sampiyon":    sampiyon,
            "feat":        feat,
            "hedef_cols":  hedef_cols,
        }

    return sonuclar


# ══════════════════════════════════════════════════════════════════
# PARÇA BAZLI TAHMİN
# ══════════════════════════════════════════════════════════════════

def parca_tahmin(parca_kodu: str, ml_df, seg_modelleri, n_ay=N_AY) -> dict:
    """
    Tek parça için:
    1. Segment şampiyonu ile test seti tahmini → MAE/RMSE/MAPE
    2. Tüm ML modelleri + geleneksel karşılaştırma
    3. Gelecek n_ay tahmini (multi-output)
    """
    pv        = parca_verisi(ml_df, parca_kodu)
    ts_train  = pv["ts_train"]
    ts_test   = pv["ts_test"]
    train_df  = pv["train"]
    test_df   = pv["test"]
    seg       = pv["segment"]
    kullan_ml = pv["kullan_ml"]

    if seg not in seg_modelleri:
        seg = list(seg_modelleri.keys())[0]

    seg_res   = seg_modelleri[seg]
    feat      = seg_res["feat"]
    hedef_cols= seg_res["hedef_cols"]
    sampiyon  = seg_res["sampiyon"]

    # Geleneksel tahminler (her zaman hesapla)
    gel = geleneksel_tahmin(ts_train, n=max(len(ts_test), n_ay))
    gel_metrikler = {
        k: metrik(ts_test, v[:len(ts_test)], k)
        for k, v in gel.items()
    }

    if not kullan_ml or seg_res["tip"] == "geleneksel":
        # C/Z → Hareketli Ortalama
        return {
            "tahminler":    gel["Hareketli Ort."][:n_ay],
            "y_test":       ts_test.tolist(),
            "y_pred_test":  gel["Hareketli Ort."][:len(ts_test)],
            "ts_train":     ts_train.tolist(),
            "sampiyon":     "Hareketli Ort.",
            "segment":      seg,
            "abc":          pv["abc"],
            "xyz":          pv["xyz"],
            "kullan_ml":    False,
            "ml_metrikler": {},
            "gel_metrikler":gel_metrikler,
            "tum_ml_pred":  {},
        }

    # ML tahminleri
    X_te = _get_X(test_df, feat)
    ml_metrikler = {}
    tum_ml_pred  = {}

    for m_adi, m_obj in seg_res.get("modeller", {}).items():
        try:
            pred = np.maximum(m_obj.predict(X_te), 0)
            p1   = pred[:, 0] if pred.ndim > 1 else pred
            ml_metrikler[m_adi] = metrik(ts_test, p1, m_adi)
            tum_ml_pred[m_adi]  = p1.tolist()
        except Exception as e:
            print(f"  [!] {m_adi}: {e}")

    # Gelecek n_ay tahmini (şampiyon, son test satırı)
    tahminler = gel["Hareketli Ort."][:n_ay]  # fallback
    samp_model = seg_res.get("modeller", {}).get(sampiyon)
    if samp_model is not None and len(test_df) > 0:
        try:
            X_son = _get_X(test_df.iloc[-1:], feat)
            pred  = np.maximum(samp_model.predict(X_son), 0)
            if pred.ndim > 1 and pred.shape[1] >= n_ay:
                tahminler = pred[0, :n_ay].tolist()
            else:
                tahminler = gel["Hareketli Ort."][:n_ay]
        except Exception:
            pass

    y_pred_test = tum_ml_pred.get(sampiyon, gel["Hareketli Ort."][:len(ts_test)])

    return {
        "tahminler":     tahminler,
        "y_test":        ts_test.tolist(),
        "y_pred_test":   y_pred_test if isinstance(y_pred_test, list) else list(y_pred_test),
        "ts_train":      ts_train.tolist(),
        "sampiyon":      sampiyon,
        "segment":       seg,
        "abc":           pv["abc"],
        "xyz":           pv["xyz"],
        "kullan_ml":     True,
        "ml_metrikler":  ml_metrikler,
        "gel_metrikler": gel_metrikler,
        "tum_ml_pred":   tum_ml_pred,
    }


# ══════════════════════════════════════════════════════════════════
# BATCH TAHMİN
# ══════════════════════════════════════════════════════════════════

def batch_tahmin(ml_df, seg_modelleri, parcalar, n_ay=N_AY):
    rows = []
    for pid in parcalar:
        try:
            res = parca_tahmin(pid, ml_df, seg_modelleri, n_ay)
            rec = {
                "Parça_Kodu": pid,
                "Sampiyon":   res["sampiyon"],
                "Segment":    res["segment"],
                "ABC":        res["abc"],
                "XYZ":        res["xyz"],
                "Kullan_ML":  res["kullan_ml"],
            }
            for i, t in enumerate(res["tahminler"], 1):
                rec[f"Tahmin_Ay_{i}"] = round(t, 1)

            # Şampiyon metriği (ML veya geleneksel)
            tum = {**res["ml_metrikler"], **res["gel_metrikler"]}
            sm  = tum.get(res["sampiyon"], {})
            rec["MAE"]  = round(sm.get("MAE",  0), 2)
            rec["RMSE"] = round(sm.get("RMSE", 0), 2)
            mape = sm.get("MAPE", float("nan"))
            rec["MAPE"] = round(mape, 2) if not np.isnan(mape) else None
            rows.append(rec)
        except Exception as e:
            print(f"  [!] {pid}: {e}")
    return pd.DataFrame(rows)
