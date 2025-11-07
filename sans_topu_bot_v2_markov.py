import streamlit as st
import pandas as pd
import numpy as np
import random
from itertools import combinations

# -------------------------------
# 📘 UYGULAMA AYARLARI
# -------------------------------
st.set_page_config(page_title="Şans Topu Tahmin Botu v3", page_icon="🎱", layout="centered")
st.title("🎯 Şans Topu Tahmin Botu v3")
st.markdown("Yapay zeka destekli tahmin sistemi (Markov + birlikte çıkma + zaman ağırlığı)")

# -------------------------------
# 📂 VERİ YÜKLEME SİSTEMİ
# -------------------------------
st.sidebar.header("📊 Veri Kaynağı")
veri_kaynak = st.sidebar.radio("Veriyi nasıl yüklemek istersiniz?", ("GitHub Raw Link", "CSV Yükleme"))

if "df" not in st.session_state:
    st.session_state["df"] = None

if veri_kaynak == "GitHub Raw Link":
    github_url = st.sidebar.text_input(
        "GitHub Raw CSV linki:",
        "https://raw.githubusercontent.com/mhmtal04/SansTopu/main/sans_topu_ornek_veri.csv"
    )
    if st.sidebar.button("📥 GitHub'dan Veriyi Yükle"):
        try:
            df = pd.read_csv(github_url)
            st.session_state["df"] = df
            st.success("✅ Veri başarıyla yüklendi.")
        except Exception as e:
            st.error(f"Veri yüklenemedi: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("📁 CSV Dosyası Yükle", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df
            st.success("✅ Dosya başarıyla yüklendi.")
        except Exception as e:
            st.error(f"Hata: {e}")

# -------------------------------
# 📈 VERİ ÖNİZLEME
# -------------------------------
if st.session_state["df"] is not None and not st.session_state["df"].empty:
    df = st.session_state["df"]
    st.subheader("📅 Son 10 Çekiliş")
    st.dataframe(df.tail(10), use_container_width=True)

    # -------------------------------
    # 🔢 YARDIMCI FONKSİYONLAR
    # -------------------------------
    def get_weights(dates):
        """Yeni tarihlere daha fazla ağırlık verir"""
        dates = pd.to_datetime(dates)
        days_ago = (dates.max() - dates).dt.days
        max_days = days_ago.max() + 1
        return (max_days - days_ago) / max_days

    def birlikte_cikma(df):
        """Birlikte çıkan sayı çiftlerinin frekansı"""
        pair_freq = {}
        for _, row in df.iterrows():
            nums = [row["Num1"], row["Num2"], row["Num3"], row["Num4"], row["Num5"]]
            for a, b in combinations(nums, 2):
                pair = tuple(sorted((a, b)))
                pair_freq[pair] = pair_freq.get(pair, 0) + 1
        return pair_freq

    def markov_matrix(df):
        """Markov geçiş matrisi (bir sayıdan sonra gelen sayılar)"""
        mat = np.zeros((36, 36))
        for i in range(1, len(df)):
            prev = [df.iloc[i-1][f"Num{j}"] for j in range(1,6)]
            curr = [df.iloc[i][f"Num{j}"] for j in range(1,6)]
            for a in prev:
                for b in curr:
                    mat[a][b] += 1
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)
        return mat

    # -------------------------------
    # 🧮 MODEL HESAPLAMALARI
    # -------------------------------
    weights = get_weights(df["Date"])
    pair_freq = birlikte_cikma(df)
    markov = markov_matrix(df)

    # Sayı olasılıkları
    freq = pd.Series(0, index=range(1, 36), dtype=float)
    for idx, row in df.iterrows():
        for n in [row["Num1"], row["Num2"], row["Num3"], row["Num4"], row["Num5"]]:
            freq[n] += weights[idx]
    single_prob = freq / freq.sum()

    joker_freq = df["Joker"].value_counts(normalize=True)

    # -------------------------------
    # 🎰 TAHMİN ÜRETME
    # -------------------------------
    def tahmin_uret(n=3):
        predictions = []
        for _ in range(n):
            combo = np.random.choice(range(1, 36), size=5, replace=False, p=single_prob.values)
            combo = np.sort(combo)
            joker = np.random.choice(joker_freq.index, p=joker_freq.values)
            predictions.append((combo, joker))
        return predictions

    st.subheader("🎲 Tahmin Ayarları")
    tahmin_sayisi = st.slider("Kaç tahmin üretilsin?", 1, 10, 4)

    if st.button("🚀 Tahmin Üret"):
        preds = tahmin_uret(tahmin_sayisi)
        st.subheader("🔮 Üretilen Tahminler")
        for i, (nums, joker) in enumerate(preds, 1):
            sayilar = ", ".join(map(str, nums))
            st.write(f"**Tahmin {i}:** {sayilar} 🎯 Joker: {joker}")

    # -------------------------------
    # 📊 ANALİZ GRAFİKLERİ
    # -------------------------------
    st.subheader("📊 En Sık Gelen Sayılar")
    st.bar_chart(freq.sort_values(ascending=False).head(10))

    st.subheader("🤝 En Çok Birlikte Çıkan Sayı Çiftleri")
    top_pairs = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    for (a, b), count in top_pairs:
        st.write(f"{a} & {b} — {count} kez birlikte geldi")

else:
    st.info("👈 Lütfen önce bir CSV dosyası yükleyin veya GitHub Raw linki girin.")
