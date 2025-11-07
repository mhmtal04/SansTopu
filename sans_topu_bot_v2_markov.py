import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations

# =============================
# 🔧 Yardımcı Fonksiyonlar
# =============================

def get_weights(dates):
    """Tarihe göre ağırlıklandırma (yeniler daha yüksek)."""
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    return (max_days - days_ago) / max_days

def weighted_single_probabilities(df, number_col_range):
    """Her sayının ağırlıklı çıkma olasılığı."""
    weights = get_weights(df['Date'])
    total_weight = weights.sum()
    freq = pd.Series(0, index=range(1, number_col_range + 1), dtype=float)
    for idx, row in df.iterrows():
        for n in row['Numbers']:
            freq[n] += weights[idx]
    return freq / total_weight

def pair_frequencies(df, number_col_range):
    """Birlikte çıkan sayıların ağırlıklı frekans matrisi."""
    weights = get_weights(df['Date'])
    pair_freq = pd.DataFrame(0, index=range(1, number_col_range + 1),
                             columns=range(1, number_col_range + 1), dtype=float)
    for idx, row in df.iterrows():
        for a, b in combinations(row['Numbers'], 2):
            pair_freq.at[a, b] += weights[idx]
            pair_freq.at[b, a] += weights[idx]
    return pair_freq

def markov_chain(df, number_col_range):
    """Markov geçiş olasılık matrisi."""
    transitions = np.zeros((number_col_range + 1, number_col_range + 1))
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]['Numbers']
        curr = df.iloc[i]['Numbers']
        for a in prev:
            for b in curr:
                transitions[a][b] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    return np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums != 0)

def structured_pattern_score(combo, single_prob, pair_freq):
    """Seçilen sayı kombinasyonu için olasılık tabanlı skor."""
    single_product = np.prod([single_prob.get(n, 1e-6) for n in combo])
    pair_product = 1.0
    for a, b in combinations(combo, 2):
        pair_product *= pair_freq.at[a, b] if pair_freq.at[a, b] > 0 else 1e-6
    return single_product * pair_product

def generate_predictions(df, single_prob, pair_freq, markov_probs, n_preds=3, trials=10000):
    """Ana tahmin üretici."""
    predictions = []
    numbers_list = list(range(1, 35))
    single_probs_list = single_prob.values

    for _ in range(n_preds):
        best_combo = None
        best_score = -1
        for __ in range(trials):
            chosen = np.random.choice(numbers_list, size=5, replace=False,
                                      p=single_probs_list / single_probs_list.sum())
            chosen = np.sort(chosen)
            combo_score = structured_pattern_score(chosen, single_prob, pair_freq)
            markov_score = np.mean([markov_probs[a].mean() for a in chosen if a < markov_probs.shape[0]])
            final_score = combo_score * (1 + markov_score)
            if final_score > best_score:
                best_score = final_score
                best_combo = chosen

        # Joker (1–14 arası)
        joker = np.random.randint(1, 15)
        predictions.append((best_combo, joker, best_score))

    return predictions


# =============================
# 🎛️ Streamlit Arayüzü
# =============================

def main():
    st.set_page_config(page_title="🎯 Şans Topu Tahmin Botu v3", page_icon="🎲", layout="centered")
    st.title("🎯 Şans Topu Tahmin Botu v3 (Markov + Birlikte Çıkma Analizi)")

    st.write("Bu bot geçmiş çekiliş verilerini analiz eder, birlikte çıkan sayıları ve Markov geçiş olasılıklarını kullanarak olası tahminler üretir.")
    st.markdown("---")

    # Veri kaynağı seçimi
    st.subheader("📂 Veri Yükleme")
    data_option = st.radio("Veri kaynağını seç:", ["GitHub Linki", "CSV Dosyası Yükle"])

    df = None

    if data_option == "GitHub Linki":
        url = st.text_input("GitHub (Raw) CSV bağlantısını buraya girin:", 
                            placeholder="https://raw.githubusercontent.com/kullaniciadi/repo/main/sans_topu.csv")
        if st.button("🔗 Veriyi Getir"):
            try:
                df = pd.read_csv(url)
                st.success("✅ GitHub verisi başarıyla yüklendi!")
            except Exception as e:
                st.error(f"❌ Veri yüklenemedi: {e}")

    elif data_option == "CSV Dosyası Yükle":
        uploaded_file = st.file_uploader("📁 Şans Topu CSV dosyasını yükle", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dosya başarıyla yüklendi!")

    # Eğer veri yüklendiyse analiz et
    if df is not None:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']].values.tolist()
        except Exception:
            st.error("CSV formatı şu şekilde olmalı: Date, Num1, Num2, Num3, Num4, Num5, Joker")
            return

        st.write(f"Toplam çekiliş sayısı: {len(df)}")
        st.dataframe(df.tail(5))

        with st.spinner("🧠 Modeller eğitiliyor..."):
            single_prob = weighted_single_probabilities(df, 34)
            pair_freq = pair_frequencies(df, 34)
            markov_probs = markov_chain(df, 34)

        st.success("✅ Modeller hazır!")

        n_preds = st.number_input("🎲 Kaç tahmin üretmek istersiniz?", min_value=1, max_value=10, value=3, step=1)

        if st.button("🚀 Tahminleri Üret"):
            with st.spinner("Tahminler oluşturuluyor..."):
                preds = generate_predictions(df, single_prob, pair_freq, markov_probs, n_preds=n_preds)
            st.success("🎉 Tahminler hazır!")

            for i, (combo, joker, score) in enumerate(preds):
                st.write(f"**{i+1}. Tahmin:** {', '.join(map(str, combo))} + Joker: 🎯 {joker}")
                st.caption(f"Model Skoru: {score:.2e}")

            st.markdown("---")
            st.subheader("📈 Birlikte Çıkma Matrisi (İlk 10 Sayı)")
            st.dataframe(pair_freq.iloc[:10, :10].round(3))

if __name__ == "__main__":
    main() 
