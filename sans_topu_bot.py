import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Şans Topu Tahmin Botu v2", layout="wide")

# ---------------------- Yardımcı Fonksiyonlar ----------------------
def get_weights(dates, halflife_days=180):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    # exponential decay weights
    weights = 0.5 ** (days_ago / halflife_days)
    return weights

def normalize_series(s):
    s = s.copy().astype(float)
    if s.sum() == 0:
        return pd.Series(1.0 / len(s), index=s.index)
    return s / s.sum()

def compute_single_freq(df):
    # Ana sayılar: 1..34
    weights = get_weights(df['Tarih'])
    freq = pd.Series(0.0, index=range(1,35))
    for idx, row in df.iterrows():
        w = weights.iloc[idx]
        for n in row['Ana']:
            freq.at[n] += w
    return normalize_series(freq)

def compute_pair_freq(df):
    weights = get_weights(df['Tarih'])
    mat = pd.DataFrame(0.0, index=range(1,35), columns=range(1,35))
    for idx, row in df.iterrows():
        w = weights.iloc[idx]
        nums = row['Ana']
        for a,b in combinations(nums, 2):
            mat.at[a,b] += w
            mat.at[b,a] += w
    return mat

def compute_conditional(pair_freq, single_freq):
    cond = pair_freq.copy()
    for a in cond.index:
        if single_freq.at[a] > 0:
            cond.loc[a] = cond.loc[a] / single_freq.at[a]
        else:
            cond.loc[a] = 0.0
    return cond

def build_markov_order2(df):
    # key: sorted tuple of previous two draws concatenated
    transitions = {}
    for i in range(2, len(df)):
        prev_key = tuple(sorted(df.iloc[i-2]['Ana'] + df.iloc[i-1]['Ana']))
        curr = df.iloc[i]['Ana']
        if prev_key not in transitions:
            transitions[prev_key] = np.zeros(34, dtype=float)
        for n in curr:
            transitions[prev_key][n-1] += 1.0
    # normalize
    for k in list(transitions.keys()):
        s = transitions[k].sum()
        if s > 0:
            transitions[k] = transitions[k] / s
    return transitions

def markov_score(transitions, recent_two):
    key = tuple(sorted(recent_two[0] + recent_two[1]))
    if key in transitions:
        return pd.Series(transitions[key], index=range(1,35))
    else:
        # fallback to uniform or aggregated distribution
        # aggregate all transitions
        if len(transitions) == 0:
            return pd.Series(1/34, index=range(1,35))
        agg = np.zeros(34, dtype=float)
        for v in transitions.values():
            agg += v
        agg = agg / agg.sum()
        return pd.Series(agg, index=range(1,35))

def train_models(df):
    # For ML helpers we use draw index as single feature and target = numbers (flattened)
    if len(df)==0:
        return None, None, None, None
    X = np.repeat(df.index.values.reshape(-1,1), 5, axis=0)
    y = np.array([n for row in df['Ana'] for n in row])
    try:
        nb = GaussianNB().fit(X, y)
    except Exception:
        nb = None
    try:
        gb = GradientBoostingRegressor().fit(X, y)
    except Exception:
        gb = None
    # Joker models: freq + NB for joker 1..14
    joker_freq = df['Joker'].value_counts().reindex(range(1,15), fill_value=0).astype(float)
    if joker_freq.sum() > 0:
        joker_freq = joker_freq / joker_freq.sum()
    else:
        joker_freq = pd.Series(1/14, index=range(1,15))
    try:
        Xj = df.index.values.reshape(-1,1)
        yj = df['Joker'].values
        joker_nb = GaussianNB().fit(Xj, yj)
    except Exception:
        joker_nb = None
    return nb, gb, joker_freq, joker_nb

def generate_predictions(df, single_freq, cond_prob, pair_freq, transitions, nb, gb, joker_freq, joker_nb,
                         n_sets=4, trials=8000, alpha=0.35, beta=0.25, gamma=0.25, delta=0.15):
    results = []
    nums = list(range(1,35))
    single_p = single_freq.reindex(nums).values
    single_p = np.clip(single_p, 1e-9, None)
    single_p = single_p / single_p.sum()

    # prepare recent seed for markov
    if len(df) >= 2:
        recent_two = [df.iloc[-2]['Ana'], df.iloc[-1]['Ana']]
    elif len(df) == 1:
        recent_two = [df.iloc[-1]['Ana'], df.iloc[-1]['Ana']]
    else:
        recent_two = [[1,2,3,4,5],[1,2,3,4,5]]

    for s in range(n_sets):
        best = None
        best_score = -np.inf
        for t in range(trials):
            chosen = np.random.choice(nums, size=5, replace=False, p=single_p)
            chosen = np.sort(chosen)
            # components
            freq_comp = np.prod([single_freq.at[n] for n in chosen])
            cond_comp = 1.0
            for a,b in combinations(chosen,2):
                val = cond_prob.at[a,b] if cond_prob.at[a,b] > 0 else 1e-6
                cond_comp *= val
            markov_s = markov_score(transitions, recent_two)
            markov_comp = np.prod([markov_s.at[n] for n in chosen])
            coocc_boost = 1.0
            max_pair = pair_freq.values.max() if pair_freq.values.size>0 else 1.0
            for a,b in combinations(chosen,2):
                coocc_boost += (pair_freq.at[a,b] / (max_pair + 1e-12)) * delta
            # ML influences
            X_test = np.array([[len(df)+1]])
            try:
                gb_pred = gb.predict(X_test)[0] / 34.0 if gb is not None else 0.5
            except Exception:
                gb_pred = 0.5
            try:
                nb_score = 0.0
                if nb is not None and hasattr(nb, 'predict_proba'):
                    probs = nb.predict_proba(X_test)[0]
                    classes = nb.classes_
                    nb_score = np.mean([probs[np.where(classes==n)[0][0]] if n in classes else 0.0 for n in chosen])
            except Exception:
                nb_score = 0.0

            # final score combines weighted components
            final = (alpha * freq_comp) + (beta * cond_comp) + (gamma * markov_comp) + (delta * (coocc_boost - 1.0))
            final = final * (1.0 + nb_score) * (1.0 + gb_pred)
            # pattern bonus (diversified ranges) - small boost
            ranges = [0,0,0,0]
            for n in chosen:
                if n<=8: ranges[0]+=1
                elif n<=16: ranges[1]+=1
                elif n<=24: ranges[2]+=1
                else: ranges[3]+=1
            if max(ranges) <= 2:
                final *= 1.05

            if final > best_score:
                best_score = final
                best = chosen.copy()

        # Joker pick: combine freq + NB
        try:
            jk_probs = joker_freq.values.copy()
            jk_probs = jk_probs / jk_probs.sum()
            if joker_nb is not None and hasattr(joker_nb, 'predict_proba'):
                jk_nb_probs = joker_nb.predict_proba(np.array([[len(df)+1]]))[0]
                classes = joker_nb.classes_
                nb_dist = np.zeros(14)
                for idx, c in enumerate(classes):
                    if 1 <= int(c) <= 14:
                        nb_dist[int(c)-1] = jk_nb_probs[idx]
                nb_dist = nb_dist / (nb_dist.sum() + 1e-12)
            else:
                nb_dist = np.ones(14)/14
            combined = 0.6 * jk_probs + 0.4 * nb_dist
            combined = combined / combined.sum()
            joker_pick = int(np.random.choice(range(1,15), p=combined))
        except Exception:
            joker_pick = int(np.random.randint(1,15))

        results.append({'Ana': sorted([int(x) for x in best]), 'Joker': joker_pick, 'Score': float(best_score)})
    return results

# --------------------------- Streamlit UI ---------------------------
st.title("🎯 Şans Topu Tahmin Botu v2 — Markov + Co-occurrence")
st.markdown("CSV yükleyin (Tarih, Sayi_1..Sayi_5, Joker). Uygulama geçmiş veriler üzerinden 4 tahmin seti üretir.")

with st.sidebar:
    st.header("Ayarlar")
    uploaded = st.file_uploader("Geçmiş çekiliş CSV dosyası (Tarih, Sayi_1..Sayi_5, Joker)", type=['csv'])
    n_sets = st.number_input("Kaç tahmin seti üretilsin?", min_value=1, max_value=10, value=4)
    trials = st.number_input("Her set için deneme sayısı (daha çok = daha uzun)", min_value=500, max_value=30000, value=8000, step=500)
    halflife = st.number_input("Zaman ağırlığı yarılanma günü (halflife)", min_value=30, max_value=1000, value=180)
    alpha = st.slider("Frekans ağırlığı (alpha)", 0.0, 1.0, 0.35)
    beta = st.slider("Koşullu olasılık ağırlığı (beta)", 0.0, 1.0, 0.25)
    gamma = st.slider("Markov ağırlığı (gamma)", 0.0, 1.0, 0.25)
    delta = st.slider("Co-occurrence ağırlığı (delta)", 0.0, 1.0, 0.15)

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        # detect date column, joker and number columns
        date_col = None
        for c in raw.columns:
            if 'tarih' in c.lower() or 'date' in c.lower():
                date_col = c; break
        if date_col is None:
            date_col = raw.columns[0]  # fallback

        joker_col = None
        for c in raw.columns:
            if 'joker' in c.lower():
                joker_col = c; break

        # try to find five number columns by names or numeric dtype
        num_cols = [c for c in raw.columns if c!=date_col and c!=joker_col and raw[c].dtype != 'O']
        if len(num_cols) < 5:
            # fallback: assume columns order date, then 5 numbers, then joker
            cols = list(raw.columns)
            try:
                idx = cols.index(date_col)
                cand = cols[idx+1: idx+6]
                if len(cand) == 5:
                    num_cols = cand
                else:
                    num_cols = cols[1:6]
            except Exception:
                num_cols = cols[1:6]

        df = pd.DataFrame()
        df['Tarih'] = pd.to_datetime(raw[date_col], errors='coerce')
        df['Sayi_1'] = pd.to_numeric(raw[num_cols[0]], errors='coerce').astype('Int64')
        df['Sayi_2'] = pd.to_numeric(raw[num_cols[1]], errors='coerce').astype('Int64')
        df['Sayi_3'] = pd.to_numeric(raw[num_cols[2]], errors='coerce').astype('Int64')
        df['Sayi_4'] = pd.to_numeric(raw[num_cols[3]], errors='coerce').astype('Int64')
        df['Sayi_5'] = pd.to_numeric(raw[num_cols[4]], errors='coerce').astype('Int64')
        if joker_col is not None:
            df['Joker'] = pd.to_numeric(raw[joker_col], errors='coerce').astype('Int64')
        else:
            # if no joker col, try 6th column
            cols = list(raw.columns)
            if len(cols) >= 6:
                df['Joker'] = pd.to_numeric(raw[cols[5]], errors='coerce').astype('Int64')
            else:
                df['Joker'] = pd.Series([np.nan]*len(raw), dtype='Int64')

        df = df.dropna(subset=['Tarih']).reset_index(drop=True)
        df['Ana'] = df[['Sayi_1','Sayi_2','Sayi_3','Sayi_4','Sayi_5']].values.tolist()
        df['Ana'] = df['Ana'].apply(lambda r: [int(x) for x in r])

        st.success(f"Veri yüklendi — toplam çekiliş: {len(df)}")
        st.dataframe(df.head(10))

        # compute with user halflife for weights where applicable
        # override get_weights by local halflife setting via lambda closure
        global get_weights
        def get_weights_local(dates, halflife_days=halflife):
            dates = pd.to_datetime(dates)
            days_ago = (dates.max() - dates).dt.days
            return 0.5 ** (days_ago / halflife_days)
        get_weights = get_weights_local

        single_freq = compute_single_freq(df)
        pair_freq = compute_pair_freq(df)
        cond_prob = compute_conditional(pair_freq, single_freq)
        transitions = build_markov_order2(df)
        nb, gb, joker_freq, joker_nb = train_models(df)

        # show stats
        st.subheader("Temel İstatistikler")
        c1,c2 = st.columns(2)
        with c1:
            st.write("Ana sayı frekansları (zaman ağırlıklı)")
            st.bar_chart(single_freq)
            st.write(single_freq.sort_values(ascending=False).head(10))
        with c2:
            st.write("Joker frekansları")
            st.bar_chart(pd.Series(joker_freq.values, index=range(1,15)))
            st.write(pd.Series(joker_freq.values, index=range(1,15)).sort_values(ascending=False).head(5))

        st.subheader("Birlikte Çıkma - En sık çiftler")
        pairs = []
        for a in range(1,35):
            for b in range(a+1,35):
                pairs.append(((a,b), pair_freq.at[a,b]))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:30]
        top_pairs = pd.DataFrame([{'Çift': f'{p[0][0]} & {p[0][1]}', 'Sıklık': p[1]} for p in pairs_sorted])
        st.table(top_pairs)

        if st.button("Tahmin Üret"):
            with st.spinner("Tahminler üretiliyor... (birkaç saniye)"):
                preds = generate_predictions(df, single_freq, cond_prob, pair_freq, transitions, nb, gb, joker_freq, joker_nb,
                                             n_sets=n_sets, trials=int(trials), alpha=alpha, beta=beta, gamma=gamma, delta=delta)
            st.success("Tahminler hazır — sadece arayüzde gösteriliyor.")
            for i,p in enumerate(preds, start=1):
                st.markdown(f"**{i}. Tahmin Seti**")
                st.write("Ana sayılar:", ", ".join(f'{x:02d}' for x in p['Ana']))
                st.write("Joker:", f"{p['Joker']:02d}")
                st.caption(f"Model skoru: {p['Score']:.3e}")

    except Exception as e:
        st.error(f"Hata: {e}")

else:
    st.info("Lütfen sol menüden CSV dosyanızı yükleyin. (Tarih,Sayi_1..Sayi_5,Joker)")
