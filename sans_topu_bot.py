import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Şans Topu Tahmin Botu", page_icon="🎯", layout="centered")

# --- Yardımcı Fonksiyonlar ---
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    return (max_days - days_ago) / max_days

def weighted_single_probabilities(df):
    weights = get_weights(df['Tarih'])
    total_weight = weights.sum()
    freq = pd.Series(0, index=range(1, 35), dtype=float)
    for idx, row in df.iterrows():
        for n in row[['Sayi_1', 'Sayi_2', 'Sayi_3', 'Sayi_4', 'Sayi_5']]:
            freq[n] += weights[idx]
    return freq / total_weight

def pair_frequencies(df):
    weights = get_weights(df['Tarih'])
    pair_freq = pd.DataFrame(0, index=range(1, 35), columns=range(1, 35), dtype=float)
    for idx, row in df.iterrows():
        for a, b in combinations(row[['Sayi_1', 'Sayi_2', 'Sayi_3', 'Sayi_4', 'Sayi_5']], 2):
            pair_freq.at[a, b] += weights[idx]
            pair_freq.at[b, a] += weights[idx]
    return pair_freq

def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1, 35), columns=range(1, 35), dtype=float)
    for a in range(1, 35):
        if single_prob[a] > 0:
            cond_prob.loc[a] = pair_freq.loc[a] / single_prob[a]
    return cond_prob

def train_models(df):
    X = np.repeat(df.index.values.reshape(-1, 1), 5, axis=0)
    y = np.array([n for row in df[['Sayi_1', 'Sayi_2', 'Sayi_3', 'Sayi_4', 'Sayi_5']].values for n in row])
    nb_model = GaussianNB().fit(X, y)
    gb_model = GradientBoostingRegressor().fit(X, y)
    return nb_model, gb_model

def generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, n_preds=3, trials=20000):
    predictions = []
    numbers_list = list(range(1, 35))
    single_probs_list = single_prob.values
    for _ in range(n_preds):
        best_combo = None
        best_score = -1
        for __ in range(trials):
            chosen = np.random.choice(numbers_list, size=5, replace=False, p=single_probs_list / single_probs_list.sum())
            chosen = np.sort(chosen)
            combo_score = np.prod([single_prob[n] for n in chosen])
            for i in range(5):
                for j in range(i + 1, 5):
                    combo_score *= cond_prob.at[chosen[i], chosen[j]]
            X_test = np.array([[len(df) + 1]])
            nb_pred = np.mean(nb_model.predict_proba(X_test)[0]) if hasattr(nb_model, "predict_proba") else 0
            gb_pred = gb_model.predict(X_test)[0]
            final_score = combo_score * (1 + nb_pred) * (1 + gb_pred / 35)
            if final_score > best_score:
                best_score = final_score
                best_combo = chosen
        joker = np.random.randint(1, 15)
        predictions.append((best_combo, joker))
    return predictions

# --- Streamlit Arayüzü ---
def main():
    st.title("🎯 Şans Topu | Gelişmiş Tahmin Botu v3")

    uploaded_file = st.file_uploader("📂 CSV dosyanızı yükleyin (Tarih, Sayi_1~Sayi_5, Joker)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Veriler yüklendi! Toplam çekiliş: {len(df)}")
        st.dataframe(df.head())

        single_prob = weighted_single_probabilities(df)
        pair_freq = pair_frequencies(df)
        cond_prob = conditional_probabilities(single_prob, pair_freq)
        nb_model, gb_model = train_models(df)

        n_preds = st.number_input("🎲 Kaç tahmin üretmek istersiniz?", min_value=1, max_value=10, value=3, step=1)

        if st.button("🚀 Tahminleri Üret"):
            with st.spinner("🧠 Tahminler hesaplanıyor..."):
                preds = generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, n_preds=n_preds)
            st.success("🎉 Tahminler hazır!")
            for i, (nums, joker) in enumerate(preds, 1):
                st.write(f"{i}. Tahmin: {', '.join(map(str, nums))} + Joker: {joker}")

if __name__ == "__main__":
    main() 
