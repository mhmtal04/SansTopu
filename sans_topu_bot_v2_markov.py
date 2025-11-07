# sanstopu_v4.py
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import io, requests, math
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor

# ---------------- Page config ----------------
st.set_page_config(page_title="Şans Topu Tahmin Botu v4 (Entegre)",
                   page_icon="🎯", layout="wide")

# ---------------- Helpers ----------------
def safe_to_int(val):
    try:
        return int(val)
    except Exception:
        return np.nan

@st.cache_data(ttl=3600)
def get_weights(dates, halflife_days=180):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    return 0.5 ** (days_ago / halflife_days)

@st.cache_data(ttl=3600)
def compute_single_freq(df, halflife_days=180):
    w = get_weights(df['Date'], halflife_days)
    freq = pd.Series(0.0, index=range(1,35))
    for i, row in df.iterrows():
        weight = w.iloc[i]
        for n in row['Numbers']:
            freq.at[n] += weight
    if freq.sum() > 0:
        return freq / freq.sum()
    return freq

@st.cache_data(ttl=3600)
def compute_pair_freq(df, halflife_days=180):
    w = get_weights(df['Date'], halflife_days)
    mat = pd.DataFrame(0.0, index=range(1,35), columns=range(1,35))
    for i, row in df.iterrows():
        weight = w.iloc[i]
        nums = row['Numbers']
        for a,b in combinations(nums, 2):
            mat.at[a,b] += weight
            mat.at[b,a] += weight
    return mat

@st.cache_data(ttl=3600)
def compute_conditional(single_prob, pair_freq):
    cond = pair_freq.copy().astype(float)
    for a in cond.index:
        p_a = single_prob.get(a, 0)
        if p_a > 0:
            cond.loc[a] = cond.loc[a] / p_a
        else:
            cond.loc[a] = 0.0
    return cond

@st.cache_data(ttl=3600)
def build_markov_order2(df):
    transitions = {}
    # key = sorted concat of two previous draws
    for i in range(2, len(df)):
        key = tuple(sorted(df.iloc[i-2]['Numbers'] + df.iloc[i-1]['Numbers']))
        curr = df.iloc[i]['Numbers']
        if key not in transitions:
            transitions[key] = np.zeros(34, dtype=float)
        for n in curr:
            transitions[key][n-1] += 1.0
    # normalize
    for k in list(transitions.keys()):
        s = transitions[k].sum()
        if s > 0:
            transitions[k] = transitions[k] / s
    return transitions

def markov_score_for_seed(transitions, recent_two):
    key = tuple(sorted(recent_two[0] + recent_two[1]))
    if key in transitions:
        return pd.Series(transitions[key], index=range(1,35))
    if len(transitions) == 0:
        return pd.Series(1.0/34, index=range(1,35))
    agg = np.zeros(34, dtype=float)
    for v in transitions.values():
        agg += v
    agg = agg / agg.sum()
    return pd.Series(agg, index=range(1,35))

def model_pattern_score_shans(combo):
    # simple balanced-range bonus
    ranges = [0,0,0,0]  # 1-8,9-16,17-24,25-34
    for n in combo:
        if n <= 8: ranges[0]+=1
        elif n <= 16: ranges[1]+=1
        elif n <= 24: ranges[2]+=1
        else: ranges[3]+=1
    if max(ranges) <= 2:
        return 1.2
    return 1.0

def structured_pattern_score_shans(combo, single_prob, pair_freq):
    eps = 1e-12
    model_score = model_pattern_score_shans(combo)
    single_product = np.prod([max(single_prob.get(n, eps), eps) for n in combo])
    pair_product = 1.0
    for a,b in combinations(combo,2):
        pair_product *= max(pair_freq.at[a,b], eps)
    return model_score * single_product * pair_product

@st.cache_data(ttl=3600)
def train_models(df):
    # NB and GB for main numbers; NB for Joker
    if len(df) < 1:
        return None, None, None, None
    # For main numbers: repeat index 5 times (one per number)
    X = np.repeat(df.index.values.reshape(-1,1), 5, axis=0)
    y = np.array([n for row in df['Numbers'] for n in row])
    try:
        nb = GaussianNB().fit(X, y)
    except Exception:
        nb = None
    try:
        gb = GradientBoostingRegressor().fit(X, y)
    except Exception:
        gb = None
    # Joker model + freq
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

# ---------------- Prediction ----------------
def generate_predictions_with_models(df, single_prob, pair_freq, cond_prob, transitions,
                                     nb, gb, joker_freq, joker_nb,
                                     n_sets=4, trials=5000, alpha=0.35, beta=0.25, gamma=0.25, delta=0.15):
    results = []
    numbers = list(range(1,35))
    p = single_prob.reindex(numbers).values
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()

    # recent seed for markov
    if len(df) >= 2:
        recent_two = [df.iloc[-2]['Numbers'], df.iloc[-1]['Numbers']]
    elif len(df) == 1:
        recent_two = [df.iloc[-1]['Numbers'], df.iloc[-1]['Numbers']]
    else:
        recent_two = [[1,2,3,4,5],[1,2,3,4,5]]

    total_combinations = math.comb(34,5)
    theoretical_odds = 1.0 / total_combinations

    for s in range(n_sets):
        best_combo = None
        best_score = -np.inf
        for t in range(trials):
            chosen = np.random.choice(numbers, size=5, replace=False, p=p)
            chosen.sort()
            # components
            freq_comp = np.prod([max(single_prob.at[n], 1e-12) for n in chosen])
            cond_comp = 1.0
            for a,b in combinations(chosen,2):
                cond_comp *= max(cond_prob.at[a,b], 1e-12)
            markov_s = markov_score_for_seed(transitions, recent_two)
            markov_comp = np.prod([max(markov_s.at[n], 1e-12) for n in chosen])
            coocc_boost = 1.0
            max_pair = pair_freq.values.max() if pair_freq.values.size>0 else 1.0
            for a,b in combinations(chosen,2):
                coocc_boost += (pair_freq.at[a,b] / (max_pair + 1e-12)) * delta

            # ML helpers
            X_test = np.array([[len(df)+1]])
            try:
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

            # structured pattern
            pattern_score = structured_pattern_score_shans(chosen, single_prob, pair_freq)

            # final combine (multiplicative + additive style)
            final = (alpha * freq_comp) + (beta * cond_comp) + (gamma * markov_comp) + (delta * (coocc_boost - 1.0))
            final = final * (1.0 + nb_score) * (1.0 + gb_pred) * (1.0 + 0.1*(pattern_score-1.0))

            if final > best_score:
                best_score = final
                best_combo = chosen.copy()

        # Joker pick: combine freq + NB if available
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
                nb_dist = np.ones(14) / 14
            combined = 0.6 * jk_probs + 0.4 * nb_dist
            combined = combined / combined.sum()
            joker_pick = int(np.random.choice(range(1,15), p=combined))
        except Exception:
            joker_pick = int(np.random.randint(1,15))

        advantage = best_score / theoretical_odds if theoretical_odds>0 else None
        results.append({'Ana': sorted([int(x) for x in best_combo]), 'Joker': joker_pick, 'Score': float(best_score), 'Adv': advantage})
    return results

# ---------------- Streamlit UI ----------------
def main():
    st.title("🎯 Şans Topu Tahmin Botu v4 — Süper Loto Modeliyle Entegre")

    st.sidebar.header("Veri & Ayarlar")
    source = st.sidebar.radio("Veri kaynağı:", ("GitHub Raw URL", "Dosya Yükle"))
    default_raw = "https://raw.githubusercontent.com/mhmtal04/SansTopu/main/sans_topu_ornek_veri.csv"
    raw_url = st.sidebar.text_input("GitHub Raw CSV URL", value=default_raw)
    uploaded = st.sidebar.file_uploader("Veya CSV dosya yükle", type=['csv'])
    n_sets = st.sidebar.number_input("Kaç tahmin seti üretilsin?", min_value=1, max_value=10, value=4)
    trials = st.sidebar.number_input("Her set için deneme sayısı", min_value=500, max_value=30000, value=5000, step=500)
    halflife = st.sidebar.number_input("Zaman ağırlığı (halflife gün)", min_value=30, max_value=1000, value=180)
    alpha = st.sidebar.slider("Frekans ağırlığı (alpha)", 0.0, 1.0, 0.35)
    beta = st.sidebar.slider("Koşullu ağırlık (beta)", 0.0, 1.0, 0.25)
    gamma = st.sidebar.slider("Markov ağırlığı (gamma)", 0.0, 1.0, 0.25)
    delta = st.sidebar.slider("Co-occurrence ağırlığı (delta)", 0.0, 1.0, 0.15)

    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'preds' not in st.session_state:
        st.session_state['preds'] = None
    if 'nb' not in st.session_state:
        st.session_state['nb'] = None
    if 'gb' not in st.session_state:
        st.session_state['gb'] = None
    if 'joker_nb' not in st.session_state:
        st.session_state['joker_nb'] = None
    if 'single_freq' not in st.session_state:
        st.session_state['single_freq'] = None

    # Load data
    df = None
    if source == "GitHub Raw URL":
        if st.sidebar.button("📥 GitHub'dan Veriyi Yükle"):
            try:
                txt = requests.get(raw_url).text
                raw = pd.read_csv(io.StringIO(txt))
                st.session_state['df'] = raw
                st.success("✅ Veri yüklendi (GitHub).")
            except Exception as e:
                st.error(f"Veri alınamadı: {e}")
    else:
        if uploaded is not None and st.sidebar.button("📥 Dosyadan Veriyi Yükle"):
            try:
                raw = pd.read_csv(uploaded)
                st.session_state['df'] = raw
                st.success("✅ Dosya yüklendi.")
            except Exception as e:
                st.error(f"Hata: {e}")

    if st.session_state['df'] is None:
        st.info("Soldan veri yükleyin (GitHub raw link veya dosya yükle).")
        return

    # Clean & normalize df
    raw = st.session_state['df']
    # Attempt to standardize columns
    cols_lc = [c.strip().lower() for c in raw.columns]
    if not any('date' in c for c in cols_lc):
        st.error("CSV içinde 'Date' sütunu bulunamadı. Lütfen format: Date, Num1, Num2, Num3, Num4, Num5, Joker")
        return
    # find Date column name
    date_col = next(c for c in raw.columns if 'date' in c.lower())
    # find joker col
    joker_col = None
    for c in raw.columns:
        if 'joker' in c.lower():
            joker_col = c
            break
    # find numeric columns for numbers (exclude date and joker)
    num_candidates = [c for c in raw.columns if c != date_col and c != joker_col and pd.api.types.is_numeric_dtype(raw[c])]
    if len(num_candidates) < 5:
        # fallback by position
        cols_list = list(raw.columns)
        try:
            idx = cols_list.index(date_col)
            num_candidates = cols_list[idx+1:idx+6]
        except Exception:
            num_candidates = cols_list[1:6]
    try:
        clean = pd.DataFrame()
        clean['Date'] = pd.to_datetime(raw[date_col], errors='coerce')
        clean['Num1'] = pd.to_numeric(raw[num_candidates[0]], errors='coerce').astype('Int64')
        clean['Num2'] = pd.to_numeric(raw[num_candidates[1]], errors='coerce').astype('Int64')
        clean['Num3'] = pd.to_numeric(raw[num_candidates[2]], errors='coerce').astype('Int64')
        clean['Num4'] = pd.to_numeric(raw[num_candidates[3]], errors='coerce').astype('Int64')
        clean['Num5'] = pd.to_numeric(raw[num_candidates[4]], errors='coerce').astype('Int64')
        if joker_col is not None:
            clean['Joker'] = pd.to_numeric(raw[joker_col], errors='coerce').astype('Int64')
        else:
            # try 6th column
            cols_list = list(raw.columns)
            if len(cols_list) >= 6:
                clean['Joker'] = pd.to_numeric(raw[cols_list[5]], errors='coerce').astype('Int64')
            else:
                clean['Joker'] = pd.NA
        clean = clean.dropna(subset=['Date']).reset_index(drop=True)
        clean['Numbers'] = clean[['Num1','Num2','Num3','Num4','Num5']].values.tolist()
        st.session_state['df'] = clean
    except Exception as e:
        st.error(f"Veri temizleme hatası: {e}")
        return

    df = st.session_state['df']
    st.subheader("Veri Önizleme (son 8 çekiliş)")
    st.dataframe(df.tail(8))

    # compute & cache heavy stuff
    single_freq = compute_single_freq(df, halflife_days=halflife)
    pair_freq = compute_pair_freq(df, halflife_days=halflife)
    cond_prob = compute_conditional(single_freq, pair_freq)
    transitions = build_markov_order2(df)

    # train ML models once and store in session_state
    if st.session_state['nb'] is None or st.session_state['gb'] is None or st.session_state['joker_nb'] is None:
        with st.spinner("ML modelleri eğitiliyor..."):
            nb, gb, joker_freq, joker_nb = train_models(df)
            st.session_state['nb'] = nb
            st.session_state['gb'] = gb
            st.session_state['joker_nb'] = joker_nb
            st.session_state['joker_freq'] = joker_freq
    else:
        nb = st.session_state['nb']
        gb = st.session_state['gb']
        joker_nb = st.session_state['joker_nb']
        joker_freq = st.session_state.get('joker_freq', pd.Series(1/14, index=range(1,15)))

    # show stats
    st.subheader("Temel İstatistikler")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.write("Ana sayı - zaman ağırlıklı frekans (üst 10)")
        st.bar_chart(single_freq)
        st.write(single_freq.sort_values(ascending=False).head(10))
    with c2:
        st.write("Joker dağılımı (frekans)")
        st.bar_chart(pd.Series(joker_freq.values, index=range(1,15)))
        st.write(pd.Series(joker_freq.values, index=range(1,15)).sort_values(ascending=False).head(5))
    with c3:
        st.metric("Toplam çekiliş", len(df))

    st.subheader("En sık birlikte çıkan çiftler")
    pairs = []
    for a in range(1,35):
        for b in range(a+1,35):
            pairs.append(((a,b), pair_freq.at[a,b]))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:20]
    top_pairs = pd.DataFrame([{'Çift': f'{p[0][0]} & {p[0][1]}', 'Sıklık': p[1]} for p in pairs_sorted])
    st.table(top_pairs)

    # Prediction (use session_state to avoid reset)
    if st.button("Tahmin Üret"):
        with st.spinner("Tahminler hesaplanıyor... (birkaç saniye)"):
            preds = generate_predictions_with_models(df, single_freq, pair_freq, cond_prob, transitions,
                                                     nb, gb, joker_freq, joker_nb,
                                                     n_sets=int(n_sets), trials=int(trials),
                                                     alpha=alpha, beta=beta, gamma=gamma, delta=delta)
            st.session_state['preds'] = preds

    if st.session_state.get('preds') is not None:
        st.subheader("Tahminler (arayüzde gösteriliyor)")
        for i,p in enumerate(st.session_state['preds'], start=1):
            st.markdown(f"**{i}. Tahmin**")
            st.write("Ana sayılar: " + ", ".join(f'{x:02d}' for x in p['Ana']))
            st.write("Joker: " + f"{p['Joker']:02d}")
            st.caption(f"Model skoru: {p['Score']:.3e} | Avantaj: {p['Adv']:.1f}x")

if __name__ == "__main__":
    main()
