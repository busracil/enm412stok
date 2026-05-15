"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi Modernizasyonu
Modül 1 – Veri Yükleme ve Geleneksel Tahmin Yöntemleri
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

Kural:
    A veya B sınıfı → ML (RF · XGBoost · LightGBM · CatBoost)
    C sınıfı veya Z grubu → Geleneksel (Hareketli Ortalama)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ML'e girecek özellik sütunları (Feature_Rehberi'nden, leakage yok)
FEATURE_COLS = [
    "lag_1", "lag_3", "lag_6", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_std_3", "roll_std_6", "roll_max_3",
    "Ay", "Yil", "Ceyrek", "sin_ay", "cos_ay",
    "MPS_Toplam_Arac", "MPS_lag_1",
    "MPS_LC12m", "MPS_LC18m", "MPS_Coach", "MPS_Coach2", "MPS_Skyliner",
    "ABC_enc", "XYZ_enc",
]

TARGET   = "Talep"
PARCA    = "Parça_Kodu"
TARIH    = "Tarih"
SPLIT    = "Split"


def ml_mi(abc: str, xyz: str) -> bool:
    """A veya B sınıfı VE Z grubu değilse ML kullan."""
    return (abc in ["A", "B"]) and (xyz != "Z")


# ══════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════

def veri_yukle(dosya_yolu: str) -> dict:
    xl = pd.ExcelFile(dosya_yolu)

    # ML hazır veri
    ml = pd.read_excel(xl, sheet_name="ML_Hazir_Veri")
    ml[PARCA] = ml[PARCA].astype(str).str.strip()
    ml[TARIH] = ml[TARIH].astype(str).str.strip()
    for c in FEATURE_COLS:
        if c in ml.columns:
            ml[c] = pd.to_numeric(ml[c], errors="coerce").fillna(0)
    ml[TARGET] = pd.to_numeric(ml[TARGET], errors="coerce").fillna(0)

    # ABC/XYZ segmentasyon
    abc = pd.read_excel(xl, sheet_name="ABC_XYZ_Segmentasyon")
    abc[PARCA] = abc[PARCA].astype(str).str.strip()

    # Optimizasyon parametreleri
    opt = pd.read_excel(xl, sheet_name="Optimizasyon_Parametreleri")
    opt[PARCA] = opt[PARCA].astype(str).str.strip()
    opt = opt.rename(columns={
        "Tedarik Süresi (gün)":     "LT_gun",
        "Birim Maliyet (TL)":       "Birim_Maliyet",
        "Sipariş Maliyeti (TL)":    "S",
        "Elde Tutma (TL/adet/ay)":  "h",
        "Stoksuz Maliyet (TL)":     "p",
        "Başlangıç Stok":           "Baslangic_Stok",
    })
    opt["LT_ay"] = opt["LT_gun"] / 30

    # Maliyet katsayılarını ML verisine ekle
    ml = ml.merge(
        opt[[PARCA, "LT_gun", "LT_ay", "Birim_Maliyet", "S", "h", "p", "Baslangic_Stok"]],
        on=PARCA, how="left"
    )

    # ABC/XYZ bilgilerini ekle
    ml = ml.merge(
        abc[[PARCA, "Ort_Aylik_Talep", "Std_Sapma", "CV"]],
        on=PARCA, how="left", suffixes=("", "_abc")
    )

    # ML mi geleneksel mi?
    ml["kullan_ml"] = ml.apply(
        lambda r: ml_mi(str(r.get("ABC", "C")), str(r.get("XYZ", "Z"))), axis=1
    )

    train = ml[ml[SPLIT] == "Train"].copy().reset_index(drop=True)
    test  = ml[ml[SPLIT] == "Test"].copy().reset_index(drop=True)
    parcalar = sorted(ml[PARCA].unique().tolist())

    ml_p  = sorted(ml[ml["kullan_ml"] == True][PARCA].unique().tolist())
    gel_p = sorted(ml[ml["kullan_ml"] == False][PARCA].unique().tolist())

    print(f"[Veri] {len(parcalar):,} parça | ML: {len(ml_p):,} | Geleneksel: {len(gel_p):,}")
    print(f"       Train: {len(train):,} satır | Test: {len(test):,} satır")

    return {
        "ml_df":       ml,
        "abc_df":      abc,
        "opt_df":      opt,
        "train_df":    train,
        "test_df":     test,
        "parcalar":    parcalar,
        "ml_parcalar": ml_p,
        "gel_parcalar":gel_p,
    }


def parca_verisi(ml_df: pd.DataFrame, parca_kodu: str) -> dict:
    """Tek parça için veriyi döner."""
    pdf   = ml_df[ml_df[PARCA] == parca_kodu].sort_values(TARIH)
    train = pdf[pdf[SPLIT] == "Train"]
    test  = pdf[pdf[SPLIT] == "Test"]
    return {
        "pdf":       pdf,
        "train":     train,
        "test":      test,
        "ts_train":  train[TARGET].values,
        "ts_test":   test[TARGET].values,
        "tarihler":  pdf[TARIH].tolist(),
        "kullan_ml": bool(pdf["kullan_ml"].iloc[0]) if "kullan_ml" in pdf.columns else True,
        "abc":       str(pdf["ABC"].iloc[0])     if "ABC"     in pdf.columns else "C",
        "xyz":       str(pdf["XYZ"].iloc[0])     if "XYZ"     in pdf.columns else "Z",
        "segment":   str(pdf["Segment"].iloc[0]) if "Segment" in pdf.columns else "CZ",
    }


# ══════════════════════════════════════════════════════════════════
# GELENEKSEL TAHMİN YÖNTEMLERİ
# ══════════════════════════════════════════════════════════════════

def geleneksel_tahmin(ts_train: np.ndarray, n: int = 6) -> dict:
    """
    Üç geleneksel yöntem:
    - Hareketli Ortalama (son 6 ay)
    - Üstel Düzeltme (α=0.3)
    - Naif (son değer)
    """
    ts = np.array(ts_train, dtype=float)
    k  = len(ts)

    # Hareketli Ortalama
    pencere = min(6, k)
    ho_val  = float(np.mean(ts[-pencere:])) if k > 0 else 0.0
    ho      = [ho_val] * n

    # Üstel Düzeltme
    alpha  = 0.3
    u_val  = float(ts[0]) if k > 0 else 0.0
    for v in ts:
        u_val = alpha * float(v) + (1 - alpha) * u_val
    ustel = [u_val] * n

    # Naif
    naif_val = float(ts[-1]) if k > 0 else 0.0
    naif     = [naif_val] * n

    return {"Hareketli Ort.": ho, "Üstel Düzeltme": ustel, "Naif": naif}


# ══════════════════════════════════════════════════════════════════
# METRİKLER
# ══════════════════════════════════════════════════════════════════

def metrik(y_true, y_pred, isim=""):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae_v  = float(mean_absolute_error(y_true, y_pred))
    rmse_v = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask   = y_true > 0
    mape_v = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
             if mask.sum() > 0 else float("nan")
    return {"model": isim, "MAE": mae_v, "RMSE": rmse_v, "MAPE": mape_v}
