import streamlit as st
import pandas as pd
import numpy as np
import io
import requests

st.set_page_config(page_title="Şans Topu Tahmin Botu", page_icon="🎱", layout="centered")

st.title("🎱 Şans Topu Tahmin Botu")
st.write("Geçmiş çekiliş verilerini kullanarak Markov zinciri mantığında tahminler üretir.")

# --- Veri Kaynağı Seçimi ---
st.sidebar.header("Veri Kaynağı Seç")
veri_tipi = st.sidebar.radio("Veriyi nasıl yüklemek istersiniz?", ["📁 CSV Yükle", "🌐 GitHub Raw Link"])

df = None

if veri_tipi == "📁 CSV Yükle":
    uploaded_file = st.sidebar.file_uploader("CSV dosyanızı yükleyin", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
elif veri_tipi == "🌐 GitHub Raw Link":
    raw_url = st.sidebar.text_input("GitHub Raw CSV bağlantısını girin:")
    if st.sidebar.button("Veriyi Getir") and raw_url:
        try:
            response = requests.get(raw_url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            st.success("✅ Veriler başarıyla yüklendi!")
        except Exception as e:
            st.error(f"❌ Veri alınamadı: {e}")

# --- Veri Kontrolü ---
if df is not None:
    try:
        df.columns = ["Date", "Num1", "Num2", "Num3", "Num4", "Num5", "Joker"]
        df = df.dropna().reset_index(drop=True)
        st.subheader("📊 Son 10 Çekiliş")
        st.dataframe(df.tail(10))

        # --- Sayı Frekans Analizi ---
        all_numbers = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].values.flatten()
        freq = pd.Series(all_numbers).value_counts().sort_index()
        st.bar_chart(freq)

        # --- Birlikte Çıkma Analizi ---
        co_occurrence = {}
        for _, row in df.iterrows():
            numbers = row[["Num1", "Num2", "Num3", "Num4", "Num5"]].tolist()
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
        co_df = pd.DataFrame(
            [{"Sayı1": k[0], "Sayı2": k[1], "Birlikte Çıkma": v} for k, v in co_occurrence.items()]
        ).sort_values(by="Birlikte Çıkma", ascending=False)
        st.subheader("🤝 En Çok Birlikte Çıkan Sayılar")
        st.dataframe(co_df.head(10))

        # --- Markov Temelli Tahmin ---
        def generate_prediction(df, num_predictions=4):
            all_numbers = df[["Num1", "Num2", "Num3", "Num4", "Num5"]].values.flatten()
            freq = pd.Series(all_numbers).value_counts(normalize=True)
            predictions = []
            for _ in range(num_predictions):
                pick = np.random.choice(freq.index, size=5, replace=False, p=freq.values)
                pick.sort()
                joker = np.random.choice(df["Joker"])
                predictions.append({"Tahmin": list(pick), "Joker": joker})
            return predictions

        if st.button("🎯 Tahmin Üret"):
            preds = generate_prediction(df)
            st.success("Tahminler oluşturuldu!")
            for i, p in enumerate(preds, 1):
                st.write(f"**Tahmin {i}:** {p['Tahmin']} 🎰 Joker: {p['Joker']}")

    except Exception as e:
        st.error(f"Veri işlenirken hata oluştu: {e}")
else:
    st.info("Lütfen bir CSV yükleyin veya GitHub Raw bağlantısı girin.")

st.markdown("---")
st.caption("⚙️ Geliştirici: mhmtal04 | GPT-5 destekli Şans Topu Botu") 
