"""
ENM412 – MAN Türkiye A.Ş. Stok Yönetimi Modernizasyonu
Modül 4 – Ana Pipeline
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

Çalıştırma:
    python m4_pipeline.py --dosya MAN_ML_Dataset_v3.xlsx --n_trials 30
"""

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from m1_veri    import veri_yukle
from m2_modeller import segment_modelleri_egit, batch_tahmin


def pipeline_calistir(dosya_yolu, n_trials=30, cache="enm412_cache.pkl"):
    t0 = time.time()
    print("=" * 65)
    print("  ENM412 – MAN TÜRKİYE A.Ş. STOK OPTİMİZASYON PİPELİNE")
    print("=" * 65)

    # 1. Veri
    print("\n[1/4] Veri yükleniyor...")
    veri = veri_yukle(dosya_yolu)

    # 2. Model eğitimi
    print(f"\n[2/4] Modeller eğitiliyor...")
    print(f"      A/B → RF · XGBoost · LightGBM · CatBoost ({n_trials} trial)")
    print(f"      C/Z → Hareketli Ortalama")
    seg_mod = segment_modelleri_egit(
        veri["train_df"], veri["test_df"], n_trials=n_trials
    )

    print("\n  Segment Özeti:")
    for seg, res in seg_mod.items():
        if res["tip"] == "ml":
            m = res["metrikler"]
            en_iyi = min(m, key=lambda k: m[k]["RMSE"])
            print(f"  [{seg}] Şampiyon: {en_iyi} "
                  f"(RMSE={m[en_iyi]['RMSE']:,.0f})")
        else:
            print(f"  [{seg}] Geleneksel yöntem")

    # 3. Batch tahmin
    print("\n[3/4] Tüm parçalar için tahmin üretiliyor...")
    batch_df = batch_tahmin(veri["ml_df"], seg_mod, veri["parcalar"], n_ay=6)
    print(f"      {len(batch_df):,} parça tamamlandı.")

    # 4. Cache
    sonuc = {
        "veri":          veri,
        "seg_modelleri": seg_mod,
        "batch_df":      batch_df,
    }
    print(f"\n[4/4] Cache kaydediliyor → {cache}")
    with open(cache, "wb") as f:
        pickle.dump(sonuc, f, protocol=4)

    print(f"\n✅ Pipeline tamamlandı! Süre: {(time.time()-t0)/60:.1f} dk")
    print("=" * 65)
    return sonuc


def cache_yukle(cache="enm412_cache.pkl"):
    with open(cache, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ENM412 MAN Türkiye Stok Optimizasyon Pipeline"
    )
    parser.add_argument("--dosya",    default="MAN_ML_Dataset_v3.xlsx")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--cache",    default="enm412_cache.pkl")
    args = parser.parse_args()
    pipeline_calistir(args.dosya, args.n_trials, args.cache)
