import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor

# ------------------- Yardımcı Fonksiyonlar -------------------

def get_weights(dates):
    """Tarihe göre ağırlık hesaplar (yakın tarih = yüksek ağırlık)"""
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    return (max_days - days_ago) / max_days


def weighted_single_probabilities(df):
    """Her sayının ağırlıklı olasılığını hesaplar"""
    weights = get_weights(df['Date'])
    total_weight = weights.sum()
    freq = pd.Series(0, index=range(1, 35), dtype=float)  # 1-34 arası ana sayılar
    for idx, row in df.iterrows():
        for n in row['Numbers']:
            freq[n] += weights[idx]
    return freq / total_weight


def pair_frequencies(df):
    """Sayısal ikili frekanslarını hesaplar"""
    weights = get_weights(df['Date'])
    pair_freq = pd.DataFrame(0, index=range(1, 35), columns=range(1, 35), dtype=float)
    for idx, row in df.iterrows():
        for a, b in combinations(row['Numbers'], 2):
            pair_freq.at[a, b] += weights[idx]
            pair_freq.at[b, a] += weights[idx]
    return pair_freq


def conditional_probabilities(single_prob, pair_freq):
    """Koşullu olasılıklar (ikili ilişkiler)"""
    cond_prob = pd.DataFrame(0, index=range(1, 35), columns=range(1, 35), dtype=float)
    for a in range(1, 35):
        if single_prob[a] > 0:
            cond_prob.loc[a] = pair_freq.loc[a] / single_prob[a]
    return cond_prob


def markov_chain(df):
    """Markov geçiş olasılıklarını hesaplar"""
    transitions = np.zeros((35, 35))
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]['Numbers']
        curr = df.iloc[i]['Numbers']
        for a in prev:
            for b in curr:
                transitions[a][b] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    return np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums != 0)


# ------------------- Model Eğitimi -------------------

def train_naive_bayes(df):
    """Naive Bayes modeli"""
    X = np.repeat(df.index.values.reshape(-1, 1), 5, axis=0)
    y = np.array([n for row in df['Numbers'] for n in row])
    model = GaussianNB()
    model.fit(X, y)
    return model


def train_gradient_boost(df):
    """Gradient Boosting modeli"""
    X = np.repeat(df.index.values.reshape(-1, 1), 5, axis=0)
    y = np.array([n for row in df['Numbers'] for n in row])
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model


# ------------------- Tahmin Fonksiyonu -------------------

def generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, markov_probs, pair_freq, n_preds=1, trials=1000000):
    """Ana tahmin üretme fonksiyonu"""
    predictions = []
    numbers_list = list(range(1, 35))
    single_probs_list = single_prob.values
    theoretical_odds = 1 / 324632  # Şans Topu için örnek oran

    for _ in range(n_preds):
        best_combo = None
        best_score = -1
        for __ in range(trials):
            chosen = np.random.choice(numbers_list, size=5, replace=False, p=single_probs_list / single_probs_list.sum())
            chosen = np.sort(chosen)

            combo_score = 1.0
            for i in range(5):
                combo_score *= single_prob[chosen[i]]
                for j in range(i + 1, 5):
                    combo_score *= cond_prob.at[chosen[i], chosen[j]]

            X_test = np.array([[len(df) + 1]])
            gb_pred = gb_model.predict(X_test)[0]
            markov_score = np.mean([markov_probs[a].mean() if a < markov_probs.shape[0] else 0 for a in chosen])

            final_score = combo_score * (1 + gb_pred / 35) * (1 + markov_score)

            if final_score > best_score:
                best_score = final_score
                best_combo = chosen

        if best_combo is not None:
            joker = np.random.randint(1, 15)
            advantage = best_score / theoretical_odds
            predictions.append((best_combo, joker, best_score, theoretical_odds, advantage))

    return predictions


# ------------------- Streamlit Arayüz -------------------

def main():
    st.title("🎯 Şans Topu Tahmin Botu | Markov + Naive Bayes + GB Modeli")

    st.write("📊 Bu bot geçmiş Şans Topu sonuçlarını analiz eder, Markov zinciri ve istatistiksel modellerle yeni tahminler üretir.")

    option = st.radio("Veri kaynağını seçin:", ["📁 CSV yükle", "🌐 GitHub Raw linki gir"])

    df = None

    if option == "📁 CSV yükle":
        uploaded_file = st.file_uploader("📂 CSV dosyanızı yükleyin (Date, Num1~Num5, Joker)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

    elif option == "🌐 GitHub Raw linki gir":
        raw_url = st.text_input("🔗 GitHub Raw CSV linkini girin:")
        if raw_url:
            try:
                df = pd.read_csv(raw_url)
                st.success("✅ Veri başarıyla yüklendi.")
            except Exception as e:
                st.error(f"CSV okunamadı: {e}")

    if df is not None:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Numbers'] = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']].values.tolist()

        with st.spinner("🧠 Modeller eğitiliyor..."):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)
            nb_model = train_naive_bayes(df)
            gb_model = train_gradient_boost(df)
            markov_probs = markov_chain(df)

        n_preds = st.number_input("🎲 Kaç tahmin üretmek istersiniz?", min_value=1, max_value=10, value=3, step=1)

        if st.button("🚀 Tahmin Üret"):
            with st.spinner("🔮 Tahminler oluşturuluyor..."):
                preds = generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, markov_probs, pair_freq, n_preds=n_preds)

            st.success("🎉 Tahminler hazır!")

            for i, (combo, joker, score, theo, adv) in enumerate(preds):
                st.write(f"**{i+1}. Tahmin:** {', '.join(map(str, combo))} | 🃏 Joker: {joker}")
                st.caption(f"📈 Model Skoru: {score:.2e} | 🎯 Teorik Olasılık: 1 / {int(1/theo)} | Avantaj: {adv:.1f} kat")

if __name__ == "__main__":
    main()
