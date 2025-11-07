import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB

# ---------------------- Yardımcı Fonksiyonlar ----------------------
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    return (max_days - days_ago) / max_days

def weighted_single_probabilities(df):
    # Ana sayılar 1..34
    weights = get_weights(df['Tarih'])
    total_weight = weights.sum()
    freq = pd.Series(0.0, index=range(1, 35))
    for idx, row in df.iterrows():
        for n in row['Ana']:
            freq.at[n] += weights.iloc[idx]
    return freq / (total_weight + 1e-12)

def pair_frequencies(df):
    weights = get_weights(df['Tarih'])
    pair_freq = pd.DataFrame(0.0, index=range(1, 35), columns=range(1, 35))
    for idx, row in df.iterrows():
        nums = row['Ana']
        for a, b in combinations(nums, 2):
            pair_freq.at[a, b] += weights.iloc[idx]
            pair_freq.at[b, a] += weights.iloc[idx]
    return pair_freq

def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0.0, index=range(1, 35), columns=range(1, 35))
    for a in range(1, 35):
        if single_prob.at[a] > 0:
            cond_prob.loc[a] = pair_freq.loc[a] / (single_prob.at[a] + 1e-12)
    return cond_prob

def cooccurrence_matrix(df):
    mat = np.zeros((34,34), dtype=float)
    for _, row in df.iterrows():
        nums = row['Ana']
        for a, b in combinations(nums, 2):
            mat[a-1, b-1] += 1.0
            mat[b-1, a-1] += 1.0
    return pd.DataFrame(mat, index=range(1,35), columns=range(1,35))

def build_markov_order2(df):
    transitions = {}
    for i in range(2, len(df)):
        prev_pair = tuple(sorted(df.iloc[i-2]['Ana'] + df.iloc[i-1]['Ana']))
        curr = df.iloc[i]['Ana']
        if prev_pair not in transitions:
            transitions[prev_pair] = np.zeros(34, dtype=float)
        for n in curr:
            transitions[prev_pair][n-1] += 1.0
    for k in list(transitions.keys()):
        s = transitions[k].sum()
        if s > 0:
            transitions[k] = transitions[k] / s
    return transitions

def markov_score_for_seed(transitions, recent_two):
    key = tuple(sorted(recent_two[0] + recent_two[1]))
    if key in transitions:
        return pd.Series(transitions[key], index=range(1,35))
    else:
        return pd.Series(1/34, index=range(1,35))

def train_naive_bayes(df):
    X = np.repeat(df.index.values.reshape(-1,1), 5, axis=0)
    y = np.array([n for row in df['Ana'] for n in row])
    model = GaussianNB()
    model.fit(X, y)
    return model

def train_gradient_boost(df):
    X = np.repeat(df.index.values.reshape(-1,1), 5, axis=0)
    y = np.array([n for row in df['Ana'] for n in row])
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

def train_joker_models(df):
    joker_freq = df['Joker'].value_counts().reindex(range(1,15), fill_value=0).astype(float)
    total = joker_freq.sum()
    if total > 0:
        joker_freq = joker_freq / total
    else:
        joker_freq = pd.Series(1/14, index=range(1,15))
    X = df.index.values.reshape(-1,1)
    y = df['Joker'].values
    nb = GaussianNB()
    try:
        nb.fit(X, y)
    except Exception:
        pass
    return joker_freq, nb

def structured_pattern_score(combo):
    ranges = {"0s": 0, "10s": 0, "20s": 0, "30s": 0, "40s": 0}
    for n in combo:
        if n < 10: ranges["0s"] += 1
        elif n < 20: ranges["10s"] += 1
        elif n < 30: ranges["20s"] += 1
        elif n < 40: ranges["30s"] += 1
        else: ranges["40s"] += 1
    pattern = [ranges[k] for k in ["0s","10s","20s","30s","40s"]]
    if max(pattern) <= 2:
        return 1.2
    return 1.0

def generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, markov_trans, pair_freq, joker_freq, joker_nb, n_sets=4, trials=8000, alpha=0.35, beta=0.25, gamma=0.25, delta=0.15):
    results = []
    numbers = list(range(1,35))
    single_probs = single_prob.reindex(numbers).values
    single_probs = np.clip(single_probs, 1e-9, None)
    single_probs = single_probs / single_probs.sum()

    if len(df) >= 2:
        recent_two = [df.iloc[-2]['Ana'], df.iloc[-1]['Ana']]
    elif len(df) == 1:
        recent_two = [df.iloc[-1]['Ana'], df.iloc[-1]['Ana']]
    else:
        recent_two = [[1,2,3,4,5],[1,2,3,4,5]]

    for s in range(n_sets):
        best_combo = None
        best_score = -np.inf
        for t in range(trials):
            chosen = np.random.choice(numbers, size=5, replace=False, p=single_probs)
            chosen = np.sort(chosen)
            freq_component = np.prod([single_prob.at[n] for n in chosen])
            cond_component = 1.0
            for a, b in combinations(chosen, 2):
                cond_component *= (cond_prob.at[a, b] if cond_prob.at[a,b] > 0 else 1e-6)
            markov_s = markov_score_for_seed(markov_trans, recent_two)
            markov_component = np.prod([markov_s.at[n] for n in chosen])
            coocc_boost = 1.0
            for a, b in combinations(chosen, 2):
                coocc_boost += (pair_freq.at[a, b] / (pair_freq.values.max() + 1e-12)) * delta
            model_pattern = structured_pattern_score(chosen)
            X_test = np.array([[len(df) + 1]])
            try:
                gb_pred = gb_model.predict(X_test)[0] / 34.0
            except Exception:
                gb_pred = 0.5
            try:
                probs = nb_model.predict_proba(X_test)[0]
                classes = nb_model.classes_
                nb_score = np.mean([probs[np.where(classes == n)[0][0]] if n in classes else 0.0 for n in chosen])
            except Exception:
                nb_score = 0.0

            final_score = (alpha * freq_component) + (beta * cond_component) + (gamma * markov_component) + (delta * coocc_boost)
            final_score = final_score * (1.0 + nb_score) * (1.0 + gb_pred) * model_pattern

            if final_score > best_score:
                best_score = final_score
                best_combo = chosen.copy()

        try:
            jk_probs = joker_freq.values.copy()
            jk_probs = jk_probs / jk_probs.sum()
            joker_pick = np.random.choice(range(1,15), p=jk_probs)
            jk_nb_probs = joker_nb.predict_proba(np.array([[len(df)+1]]))[0] if hasattr(joker_nb, 'predict_proba') else np.ones(14)/14
            jk_classes = joker_nb.classes_ if hasattr(joker_nb, 'classes_') else np.arange(1,15)
            nb_dist = np.zeros(14)
            if hasattr(joker_nb, 'classes_'):
                for idx, c in enumerate(jk_classes):
                    if 1 <= int(c) <= 14:
                        nb_dist[int(c)-1] = jk_nb_probs[idx]
                nb_dist = nb_dist / (nb_dist.sum() + 1e-12)
            else:
                nb_dist = np.ones(14)/14
            combined = 0.6 * jk_probs + 0.4 * nb_dist
            combined = combined / combined.sum()
            joker_pick = np.random.choice(range(1,15), p=combined)
        except Exception:
            joker_pick = np.random.randint(1,15)

        results.append({'Ana': sorted([int(x) for x in best_combo]), 'Joker': int(joker_pick), 'Score': float(best_score)})

    return results

# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title='Şans Topu Tahmin Botu v2', layout='wide')
st.title('🎯 Şans Topu — Tahmin Botu v2 (CSV yükleme mevcut)')
st.markdown('Bu uygulama geçmiş çekiliş verilerini analiz ederek 5 ana sayı + 1 joker tahminleri üretir. Sonuçlar sadece arayüzde gösterilir.')

with st.sidebar:
    st.header('Ayarlar & Veri')
    uploaded = st.file_uploader('Geçmiş çekiliş CSV dosyası (Tarih, Sayi_1..Sayi_5, Joker)', type=['csv'])
    sample_btn = st.button('Örnek CSV indir')
    n_sets = st.number_input('Kaç tahmin seti üretilsin?', min_value=1, max_value=10, value=4)
    trials = st.number_input('Her set için deneme sayısı (daha çok = daha uzun)', min_value=500, max_value=20000, value=5000, step=500)
    alpha = st.slider('Frekans ağırlığı (alpha)', 0.0, 1.0, 0.35)
    beta = st.slider('Koşullu olasılık ağırlığı (beta)', 0.0, 1.0, 0.25)
    gamma = st.slider('Markov ağırlığı (gamma)', 0.0, 1.0, 0.25)
    delta = st.slider('Co-Occurrence ağırlığı (delta)', 0.0, 1.0, 0.15)

if sample_btn:
    sample_csv = 'Tarih,Sayi_1,Sayi_2,Sayi_3,Sayi_4,Sayi_5,Joker\\n2025-10-30,3,8,12,24,27,9\\n2025-10-23,5,11,16,22,33,2\\n2025-10-16,1,6,13,19,29,7\\n'
    st.download_button('Örnek CSV indir', data=sample_csv, file_name='sans_topu_ornek.csv', mime='text/csv')

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        cols_lower = [c.strip().lower() for c in df_raw.columns]
        date_col = None
        for c in df_raw.columns:
            if 'tarih' in c.lower() or 'date' in c.lower():
                date_col = c
                break
        if date_col is None:
            st.warning('Tarih sütunu bulunamadı, varsayılan olarak ilk sütunu tarih kabul ediliyor.')
            date_col = df_raw.columns[0]

        joker_col = None
        for c in df_raw.columns:
            if 'joker' in c.lower():
                joker_col = c
                break
        num_cols = [c for c in df_raw.columns if c != date_col and c != joker_col and df_raw[c].dtype != 'O']

        if len(num_cols) < 5:
            possible = list(df_raw.columns)
            try_cols = [c for c in possible if c != date_col]
            if len(try_cols) >= 6:
                num_cols = try_cols[:5]
                joker_col = try_cols[5] if joker_col is None else joker_col
            else:
                st.error('CSV içinde sayı sütunları bulunamadı. Lütfen (Tarih,Sayi_1..Sayi_5,Joker) formatında dosya yükleyin.')
                st.stop()

        df = pd.DataFrame()
        df['Tarih'] = pd.to_datetime(df_raw[date_col], errors='coerce')
        df['Sayi_1'] = pd.to_numeric(df_raw[num_cols[0]], errors='coerce').astype('Int64')
        df['Sayi_2'] = pd.to_numeric(df_raw[num_cols[1]], errors='coerce').astype('Int64')
        df['Sayi_3'] = pd.to_numeric(df_raw[num_cols[2]], errors='coerce').astype('Int64')
        df['Sayi_4'] = pd.to_numeric(df_raw[num_cols[3]], errors='coerce').astype('Int64')
        df['Sayi_5'] = pd.to_numeric(df_raw[num_cols[4]], errors='coerce').astype('Int64')
        if joker_col is not None:
            df['Joker'] = pd.to_numeric(df_raw[joker_col], errors='coerce').astype('Int64')
        else:
            df['Joker'] = pd.Series([np.nan]*len(df_raw), dtype='Int64')

        df = df.dropna(subset=['Tarih']).reset_index(drop=True)
        df['Ana'] = df[['Sayi_1','Sayi_2','Sayi_3','Sayi_4','Sayi_5']].values.tolist()
        df['Ana'] = df['Ana'].apply(lambda row: [int(x) for x in row])

        st.success(f'✅ Veri yüklendi — toplam çekiliş: {len(df)}')
        st.dataframe(df.head(10))

        with st.spinner('Modeller eğitiliyor ve analizler hesaplanıyor...'):
            single_prob = weighted_single_probabilities(df)
            pair_freq = pair_frequencies(df)
            cond_prob = conditional_probabilities(single_prob, pair_freq)
            co_mat = cooccurrence_matrix(df)
            markov_trans = build_markov_order2(df)
            nb_model = train_naive_bayes(df)
            gb_model = train_gradient_boost(df)
            joker_freq, joker_nb = train_joker_models(df)

        st.subheader('Temel İstatistikler')
        c1, c2 = st.columns(2)
        with c1:
            st.write('Ana sayı frekansları (zaman ağırlıklı)')
            st.bar_chart(single_prob)
            st.write('En sık gelen 10 ana sayı:')
            st.write(single_prob.sort_values(ascending=False).head(10))
        with c2:
            st.write('Joker frekansları')
            st.bar_chart(pd.Series(joker_freq.values, index=range(1,15)))
            st.write('En sık gelen jokerler:')
            st.write(pd.Series(joker_freq.values, index=range(1,15)).sort_values(ascending=False).head(5))

        st.subheader('Birlikte Çıkma (Co-occurrence) - En sık çiftler')
        pairs = []
        for a in range(1,35):
            for b in range(a+1,35):
                pairs.append(((a,b), co_mat.at[a,b]))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:30]
        top_pairs_df = pd.DataFrame([{'Çift': f'{p[0][0]} & {p[0][1]}', 'Sıklık': p[1]} for p in pairs_sorted])
        st.table(top_pairs_df)

        st.markdown('---')
        if st.button('Tahmin Üret'):
            with st.spinner('Tahminler üretiliyor... (birkaç saniye)'):
                preds = generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, markov_trans, co_mat, joker_freq, joker_nb, n_sets=n_sets, trials=int(trials), alpha=alpha, beta=beta, gamma=gamma, delta=delta)
            st.success('Tahminler hazır — sadece arayüzde gösteriliyor.')
            for i, p in enumerate(preds, start=1):
                st.subheader(f'{i}. Tahmin Seti')
                st.write('Ana sayılar:', ', '.join(f'{x:02d}' for x in p['Ana']))
                st.write('Joker:', f'{p["Joker"]:02d}')
                st.caption(f'Model skoru: {p["Score"]:.3e}')

    except Exception as e:
        st.error(f'Bir hata oluştu: {e}')

else:
    st.info('Lütfen sol menüden geçmiş çekiliş CSV dosyanızı yükleyin. (Tarih,Sayi_1..Sayi_5,Joker)')
