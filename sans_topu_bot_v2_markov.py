import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------------------------------------
#  ŞANS TOPU TAHMİN BOTU v3 (Markov + Korelasyon + Streamlit)
# -------------------------------------------------------------
st.set_page_config(page_title="Şans Topu Tahmin Botu", page_icon="🎯", layout="wide")

# ----------------- Yardımcı Fonksiyonlar -----------------
def get_weights(dates):
    """Tarihlere göre ağırlık hesapla (yakın tarihler daha yüksek ağırlık alır)."""
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    return (max_days - days_ago) / max_days

def weighted_single_probabilities(df):
    """Tekil sayıların ağırlıklı olasılıklarını hesapla."""
    weights = get_weights(df['Tarih'])
    total_weight = weights.sum()
    freq = pd.Series(0, index=range(1, 35), dtype=float)
    for idx, row in df.iterrows():
        for n in row[['Sayi_1', 'Sayi_2', 'Sayi_3', 'Sayi_4', 'Sayi_5']]:
            freq[n] += weights[idx]
    return freq / total_weight

def pair_frequencies(df):
    """Birlikte çıkan sayı çiftlerinin frekanslarını hesapla."""
    weights = get_weights(df['Tarih'])
    pair_freq = pd.DataFrame(0, index=range(1, 35), columns=range(1, 35), dtype=float)
    for idx, row in df.iterrows():
        for a, b in combinations(row[['Sayi_1', 'Sayi_2', 'Sayi_3', 'Sayi_4', 'Sayi_5']], 2):
            pair_freq.at[a, b] += weights[idx]
            pair_freq.at[b, a] += weights[idx]
    return pair_freq

def markov_chain(df):
    """Markov geçiş matrisi (bir çekilişten diğerine geçiş olasılıkları)."""
    transitions = np.zeros((35, 35))
    for i in range(1, len(df)):
        prev = df.iloc[i - 1][['Sayi_1', 'Sayi_2', 'Sayi_3', 'Sayi_4', 'Sayi_5']].values
        curr = df.iloc[i][['Sayi_1', 'Sayi_2', 'Sayi_3', 'Sayi_4', 'Sayi_5']].values
        for a in prev:
            for b in curr:
                transitions[a][b] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    return np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums != 0)

def train_models(df):
    """Naive Bayes ve Gradient Boosting modellerini eğitir."""
    X = np.repeat(df.index.values.reshape(-1, 1), 5, axis=0)
    y = np.array([n for row in df[['Sayi_1', 'Sayi_2', 'Sayi_3', 'Sayi_4', 'Sayi_5']].values for n in row])
    nb = GaussianNB().fit(X, y)
    gb = GradientBoostingRegressor().fit(X, y)
    return nb, gb

def generate_predictions(df, single_prob, pair_freq, markov_probs, nb_model, gb_model, n_preds=4, trials=15000):
    """Markov + Olasılık + ML modelleri ile tahmin üretir."""
    preds = []
    numbers = list(range(1, 35))
    single_probs_list = single_prob.values
    for _ in range(n_preds):
        best_combo, best_score = None, -1
        for __ in range(trials):
            chosen = np.random.choice(numbers, size=5, replace=False, p=single_probs_list / single_probs_list.sum())
            chosen = np.sort(chosen)

            # Olasılık bazlı skor
            combo_score = np.prod([single_prob[n] for n in chosen])
            for a, b in combinations(chosen, 2):
                combo_score *= (pair_freq.at[a, b] + 1e-6)

            # Markov katkısı
            markov_score = np.mean([markov_probs[a].mean() for a in chosen])

            # ML modelleri
            X_test = np.array([[len(df) + 1]])
            gb_pred = gb_model.predict(X_test)[0]
            nb_score = np.mean(nb_model.predict_proba(X_test)[0]) if hasattr(nb_model, "predict_proba") else 0

            final_score = combo_score * (1 + gb_pred / 35) * (1 + nb_score) * (1 + markov_score)

            if final_score > best_score:
                best_combo, best_score = chosen, final_score

        joker = np.random.randint(1, 15)
        preds.append((best_combo, joker))
    return preds

# ----------------- Streamlit Arayüzü -----------------
def main():
    st.title("🎯 Şans Topu | Gelişmiş Tahmin Botu v3")
    st.caption("Markov Zinciri + Birlikte Çıkma Analizi + Makine Öğrenimi Destekli")

    uploaded_file = st.file_uploader("📂 CSV dosyanızı yükleyin (Tarih, Sayi_1~Sayi_5, Joker)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Tarih'] = pd.to_datetime(df['Tarih'])
        st.success(f"✅ Veriler yüklendi ({len(df)} çekiliş).")
        st.dataframe(df.tail(10))

        with st.spinner("🧠 Modeller eğitiliyor..."):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            markov_probs = markov_chain(df)
            nb_model, gb_model = train_models(df)

        n_preds = st.number_input("🎲 Kaç tahmin üretmek istersiniz?", min_value=1, max_value=10, value=4, step=1)

        if st.button("🚀 Tahminleri Üret"):
            with st.spinner("🔮 Tahminler hesaplanıyor..."):
                preds = generate_predictions(df, single_prob, pair_freq, markov_probs, nb_model, gb_model, n_preds=n_preds)
            st.success("🎉 Tahminler Hazır!")

            for i, (nums, joker) in enumerate(preds, 1):
                st.markdown(f"**{i}. Tahmin:** 🎱 {', '.join(map(str, nums))} + 🃏 Joker: {joker}")

        with st.expander("📊 Birlikte Çıkma Frekans Matrisi"):
            st.dataframe(pair_freq.style.background_gradient(cmap="Blues", axis=None))

if __name__ == "__main__":
    main()
