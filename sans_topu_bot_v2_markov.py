# sanstopu_v2.py
# Şans Topu Tahmin Botu v2
# Streamlit uygulaması — Markov + Co-occurrence + Time-weighted frequency
#
# CSV formatı (zorunlu): Date, Num1, Num2, Num3, Num4, Num5, Joker
# Kullanım:
#   pip install -r requirements.txt
#   streamlit run sanstopu_v2.py

import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
import math

# --------- Page config ----------
st.set_page_config(page_title="Şans Topu Tahmin Botu v2", page_icon="🎯", layout="wide")

# --------- Helpers & cached computations ----------
@st.cache_data
def load_csv_from_url(url: str):
    return pd.read_csv(url)

@st.cache_data
def parse_and_normalize(df_raw: pd.DataFrame):
    df = df_raw.copy()
    # try to normalize column names and find necessary columns
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    # Date column
    date_col = None
    for c in cols:
        if 'date' in c.lower() or 'tarih' in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = cols[0]
    # Find joker
    joker_col = None
    for c in cols:
        if 'joker' in c.lower():
            joker_col = c
            break
    # find five number columns (heuristic)
    num_candidates = [c for c in cols if c != date_col and c != joker_col]
    # if any column names contain 'num' or 'sayi' use them
    num_cols = [c for c in num_candidates if any(k in c.lower() for k in ['num','sayi','say'])]
    if len(num_cols) < 5:
        # fallback to ordering: take first 5 non-date,non-joker columns
        num_cols = [c for c in num_candidates][:5]
    # Ensure we have 5
    if len(num_cols) < 5:
        raise ValueError("CSV içinde 5 ana sayı sütunu bulunamadı. Beklenen format: Date, Num1..Num5, Joker")
    # assemble normalized dataframe
    norm = pd.DataFrame()
    norm['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    for i in range(5):
        norm[f'Num{i+1}'] = pd.to_numeric(df[num_cols[i]], errors='coerce').astype('Int64')
    if joker_col is not None:
        norm['Joker'] = pd.to_numeric(df[joker_col], errors='coerce').astype('Int64')
    else:
        # try 6th column if exists
        if len(num_candidates) >= 6:
            norm['Joker'] = pd.to_numeric(df[num_candidates[5]], errors='coerce').astype('Int64')
        else:
            norm['Joker'] = pd.Series([pd.NA]*len(norm), dtype='Int64')
    norm = norm.dropna(subset=['Date']).reset_index(drop=True)
    norm['Ana'] = norm[[f'Num{i+1}' for i in range(5)]].values.tolist()
    norm['Ana'] = norm['Ana'].apply(lambda row: [int(x) for x in row])
    return norm

@st.cache_data
def compute_time_weights(dates: pd.Series, halflife_days: int = 180):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    # exponential decay
    weights = 0.5 ** (days_ago / halflife_days)
    return weights

@st.cache_data
def compute_single_freq(df: pd.DataFrame, halflife_days: int = 180):
    weights = compute_time_weights(df['Date'], halflife_days)
    freq = pd.Series(0.0, index=range(1, 35))
    for i, row in df.iterrows():
        w = weights.iloc[i]
        for n in row['Ana']:
            freq.at[n] += w
    if freq.sum() == 0:
        return freq + 1.0
    return freq / freq.sum()

@st.cache_data
def compute_pair_freq(df: pd.DataFrame, halflife_days: int = 180):
    weights = compute_time_weights(df['Date'], halflife_days)
    mat = pd.DataFrame(0.0, index=range(1,35), columns=range(1,35))
    for i, row in df.iterrows():
        w = weights.iloc[i]
        nums = row['Ana']
        for a,b in combinations(nums, 2):
            mat.at[a,b] += w
            mat.at[b,a] += w
    return mat

@st.cache_data
def build_markov_order2(df: pd.DataFrame):
    transitions = {}
    # if not enough rows, return empty dict
    if len(df) < 3:
        return transitions
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

@st.cache_data
def aggregate_markov(transitions):
    # aggregated distribution if specific key not found
    if not transitions:
        return pd.Series(1/34, index=range(1,35))
    agg = np.zeros(34, dtype=float)
    for v in transitions.values():
        agg += v
    if agg.sum() == 0:
        return pd.Series(1/34, index=range(1,35))
    agg = agg / agg.sum()
    return pd.Series(agg, index=range(1,35))

@st.cache_data
def train_models(df: pd.DataFrame):
    # Flatten training set: feature = draw index, target = numbers (for NB/GB helpers)
    if len(df) == 0:
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
    # Joker frequency & NB
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

# --------- Scoring & prediction ----------
def markov_scores_for_seed(transitions, recent_two):
    key = tuple(sorted(recent_two[0] + recent_two[1]))
    if key in transitions:
        return pd.Series(transitions[key], index=range(1,35))
    else:
        return aggregate_markov(transitions)

def score_combo(combo, single_freq, pair_freq, markov_scores, alpha=0.35, beta=0.25, gamma=0.25, delta=0.15):
    # combo: iterable of 5 numbers
    # alpha: weight for single freq, beta: cond/pair, gamma: markov, delta: cooccurrence boost
    # single product
    sp = 1.0
    for n in combo:
        sp *= single_freq.at[n] if n in single_freq.index else 1e-9
    # conditional/pair product
    pp = 1.0
    for a,b in combinations(combo,2):
        val = pair_freq.at[a,b] if pair_freq.at[a,b] > 0 else 1e-9
        pp *= val
    # markov product (use markov_probs for each)
    mp = 1.0
    for n in combo:
        mp *= markov_scores.at[n] if n in markov_scores.index else 1e-9
    # co-occurrence boost factor (additive)
    max_pair = pair_freq.values.max() if pair_freq.values.size > 0 else 1.0
    co_boost = 1.0
    for a,b in combinations(combo,2):
        co_boost += (pair_freq.at[a,b] / (max_pair + 1e-12)) * delta
    # combine with weighting (use sum then multiply by boost)
    combined = (alpha * sp) + (beta * pp) + (gamma * mp)
    final = combined * co_boost
    return final

def generate_predictions(df: pd.DataFrame, single_freq, pair_freq, transitions, nb, gb, joker_freq, joker_nb,
                         n_sets=4, trials=8000, alpha=0.35, beta=0.25, gamma=0.25, delta=0.15):
    results = []
    nums = list(range(1,35))
    single_p = single_freq.reindex(nums).values
    single_p = np.clip(single_p, 1e-9, None)
    single_p = single_p / single_p.sum()
    # prepare markov seed
    if len(df) >= 2:
        recent_two = [df.iloc[-2]['Ana'], df.iloc[-1]['Ana']]
    elif len(df) == 1:
        recent_two = [df.iloc[-1]['Ana'], df.iloc[-1]['Ana']]
    else:
        recent_two = [[1,2,3,4,5],[1,2,3,4,5]]
    for s in range(n_sets):
        best_combo = None
        best_score = -math.inf
        markov_scores = markov_scores_for_seed(transitions, recent_two)
        for t in range(trials):
            chosen = np.random.choice(nums, size=5, replace=False, p=single_p)
            chosen = np.sort(chosen)
            sc = score_combo(chosen, single_freq, pair_freq, markov_scores, alpha, beta, gamma, delta)
            # ML tweak (small multiplicative factor)
            try:
                X_test = np.array([[len(df)+1]])
                gb_pred = gb.predict(X_test)[0] / 34.0 if gb is not None else 0.0
            except Exception:
                gb_pred = 0.0
            try:
                nb_score = 0.0
                if nb is not None and hasattr(nb, 'predict_proba'):
                    probs = nb.predict_proba(X_test)[0]
                    classes = nb.classes_
                    nb_score = np.mean([probs[np.where(classes==n)[0][0]] if n in classes else 0.0 for n in chosen])
            except Exception:
                nb_score = 0.0
            final = sc * (1.0 + gb_pred) * (1.0 + nb_score)
            # small diversity bonus if combo spans ranges
            ranges = [0,0,0,0]
            for n in chosen:
                if n <= 8: ranges[0]+=1
                elif n <=16: ranges[1]+=1
                elif n <=24: ranges[2]+=1
                else: ranges[3]+=1
            if max(ranges) <= 2:
                final *= 1.03
            if final > best_score:
                best_score = final
                best_combo = chosen.copy()
        # Joker selection: combine freq + joker_nb if available
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
        results.append({'Ana': sorted([int(x) for x in best_combo]), 'Joker': joker_pick, 'Score': float(best_score)})
    return results

# ---------- UI & flow ----------
def main():
    st.title("🎯 Şans Topu Tahmin Botu v2 — Gelişmiş")
    st.markdown("Markov (2nd order), zaman-ağırlıklı frekans ve birlikte çıkma analizleri ile 5+1 tahminleri üretir.")
    st.sidebar.header("Veri & Ayarlar")

    # session state for data & models to avoid resets
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'single_freq' not in st.session_state:
        st.session_state['single_freq'] = None
    if 'pair_freq' not in st.session_state:
        st.session_state['pair_freq'] = None
    if 'transitions' not in st.session_state:
        st.session_state['transitions'] = None
    if 'models' not in st.session_state:
        st.session_state['models'] = None
    if 'predictions' not in st.session_state:
        st.session_state['predictions'] = None

    data_source = st.sidebar.radio("Veri kaynağı:", ("GitHub (Raw URL)", "CSV yükle"))
    default_raw = "https://raw.githubusercontent.com/mhmtal04/SansTopu/main/sans_topu_ornek_veri.csv"
    url = st.sidebar.text_input("GitHub Raw URL", value=default_raw) if data_source.startswith("GitHub") else None
    uploaded = st.sidebar.file_uploader("Veya yerel CSV yükle", type=['csv']) if data_source == "CSV yükle" else None

    # analysis parameters
    halflife = st.sidebar.number_input("Halflife (gün) — zaman ağırlığı", min_value=30, max_value=1000, value=180)
    n_sets = st.sidebar.number_input("Kaç tahmin seti üretilsin?", min_value=1, max_value=10, value=4)
    trials = st.sidebar.number_input("Her set için deneme sayısı (trials)", min_value=200, max_value=50000, value=8000, step=200)
    alpha = st.sidebar.slider("Alpha — single freq ağırlığı", 0.0, 1.0, 0.35)
    beta = st.sidebar.slider("Beta — conditional/pair ağırlığı", 0.0, 1.0, 0.25)
    gamma = st.sidebar.slider("Gamma — markov ağırlığı", 0.0, 1.0, 0.25)
    delta = st.sidebar.slider("Delta — co-occurrence boost", 0.0, 1.0, 0.15)

    # Load data button (so page won't reload on other actions)
    if st.sidebar.button("Veriyi Yükle / Yenile"):
        try:
            if data_source == "GitHub (Raw URL)":
                raw = load_csv_from_url(url)
            else:
                raw = pd.read_csv(uploaded)
            st.session_state['df'] = parse_and_normalize(raw)
            st.success("Veri yüklendi ve normalleştirildi.")
            # compute and store resources
            st.session_state['single_freq'] = compute_single_freq(st.session_state['df'], halflife_days=halflife)
            st.session_state['pair_freq'] = compute_pair_freq(st.session_state['df'], halflife_days=halflife)
            st.session_state['transitions'] = build_markov_order2(st.session_state['df'])
            nb, gb, joker_freq, joker_nb = train_models(st.session_state['df'])
            st.session_state['models'] = (nb, gb, joker_freq, joker_nb)
            st.session_state['predictions'] = None
        except Exception as e:
            st.error(f"Veri yüklenirken hata: {e}")

    # show stats if loaded
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        st.subheader("Veri Önizleme & İstatistikler")
        c1, c2 = st.columns([2,1])
        with c1:
            st.write(f"Toplam çekiliş: {len(df)} — Başlangıç: {df['Date'].min().date()} — Son: {df['Date'].max().date()}")
            st.dataframe(df.tail(10))
        with c2:
            sf = st.session_state['single_freq']
            jf = st.session_state['models'][2] if st.session_state['models'] is not None else None
            st.write("En sık gelen ana sayılar (zaman ağırlıklı)")
            if sf is not None:
                st.bar_chart(sf)
                st.write(sf.sort_values(ascending=False).head(10))
            st.write("Joker frekans (örnek)")
            if jf is not None:
                st.bar_chart(pd.Series(jf.values, index=range(1,15)))

        st.markdown("---")
        st.subheader("Birlikte Çıkma (Co-occurrence) — En sık çiftler")
        pair_freq = st.session_state['pair_freq']
        if pair_freq is not None:
            pairs = []
            for a in range(1,35):
                for b in range(a+1,35):
                    pairs.append(((a,b), pair_freq.at[a,b]))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:30]
            top_pairs = pd.DataFrame([{'Çift': f"{p[0][0]} & {p[0][1]}", 'Sıklık': p[1]} for p in pairs_sorted])
            st.table(top_pairs)

        # Generate predictions button (won't reload data)
        if st.button("Tahmin Üret"):
            st.session_state['predictions'] = None
            try:
                nb, gb, joker_freq, joker_nb = st.session_state['models']
                preds = generate_predictions(
                    st.session_state['df'],
                    st.session_state['single_freq'],
                    st.session_state['pair_freq'],
                    st.session_state['transitions'],
                    nb, gb, joker_freq, joker_nb,
                    n_sets=int(n_sets),
                    trials=int(trials),
                    alpha=alpha, beta=beta, gamma=gamma, delta=delta
                )
                st.session_state['predictions'] = preds
            except Exception as e:
                st.error(f"Tahmin üretme sırasında hata: {e}")

        # show predictions if exist
        if st.session_state['predictions'] is not None:
            st.subheader("Üretilen Tahminler (5 ana + 1 joker)")
            for i, p in enumerate(st.session_state['predictions'], start=1):
                st.markdown(f"**{i}. Set** — Ana: {', '.join(f'{x:02d}' for x in p['Ana'])}  |  Joker: {p['Joker']:02d}")
                st.caption(f"Model skoru: {p['Score']:.3e}")

    else:
        st.info("Başlamak için sol menüden 'Veriyi Yükle / Yenile' butonuna basın (GitHub raw link veya CSV).")

if __name__ == "__main__":
    main() 
