# sanstopu_v3.py
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# --- Page config
st.set_page_config(page_title="Şans Topu Tahmin Botu v3", page_icon="🎯", layout="wide")

# -------------------------
# Cached helpers
# -------------------------
@st.cache_data(ttl=60*60)
def read_csv_from_url(url):
    return pd.read_csv(url)

@st.cache_data(ttl=60*60)
def read_csv_file(uploaded_file_bytes):
    # uploaded_file is a BytesIO-like object from st.file_uploader
    return pd.read_csv(uploaded_file_bytes)

@st.cache_data(ttl=60*60)
def time_weights(dates, halflife_days):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    return 0.5 ** (days_ago / halflife_days)

@st.cache_data(ttl=60*60)
def compute_single_freq(df, halflife_days=180):
    w = time_weights(df['Date'], halflife_days)
    freq = pd.Series(0.0, index=range(1,35))
    for i, row in df.iterrows():
        wi = w.iloc[i]
        for n in row['Numbers']:
            freq.at[n] += wi
    if freq.sum() > 0:
        return freq / freq.sum()
    return freq

@st.cache_data(ttl=60*60)
def compute_pair_freq(df, halflife_days=180):
    w = time_weights(df['Date'], halflife_days)
    mat = pd.DataFrame(0.0, index=range(1,35), columns=range(1,35))
    for i, row in df.iterrows():
        wi = w.iloc[i]
        nums = row['Numbers']
        for a,b in combinations(nums, 2):
            mat.at[a,b] += wi
            mat.at[b,a] += wi
    return mat

@st.cache_data(ttl=60*60)
def compute_markov_order2(df):
    transitions = {}
    for i in range(2, len(df)):
        prev_key = tuple(sorted(df.iloc[i-2]['Numbers'] + df.iloc[i-1]['Numbers']))
        curr = df.iloc[i]['Numbers']
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

@st.cache_data(ttl=60*60)
def train_models(df):
    if len(df) < 1:
        return None, None, None, None
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

# -------------------------
# Prediction engine
# -------------------------
def markov_score_for_seed(transitions, recent_two):
    key = tuple(sorted(recent_two[0] + recent_two[1]))
    if key in transitions:
        return pd.Series(transitions[key], index=range(1,35))
    if len(transitions) == 0:
        return pd.Series(1/34, index=range(1,35))
    agg = np.zeros(34, dtype=float)
    for v in transitions.values():
        agg += v
    agg = agg / agg.sum()
    return pd.Series(agg, index=range(1,35))

def generate_predictions(df, single_freq, pair_freq, transitions, nb, gb, joker_freq, joker_nb,
                         n_sets=4, trials=8000, alpha=0.35, beta=0.25, gamma=0.25, delta=0.15):
    results = []
    nums = list(range(1,35))
    p = single_freq.reindex(nums).values
    p = np.clip(p, 1e-9, None)
    p = p / p.sum()

    if len(df) >= 2:
        recent_two = [df.iloc[-2]['Numbers'], df.iloc[-1]['Numbers']]
    elif len(df) == 1:
        recent_two = [df.iloc[-1]['Numbers'], df.iloc[-1]['Numbers']]
    else:
        recent_two = [[1,2,3,4,5],[1,2,3,4,5]]

    for s in range(n_sets):
        best_combo = None
        best_score = -np.inf
        for t in range(trials):
            chosen = np.random.choice(nums, size=5, replace=False, p=p)
            chosen = np.sort(chosen)
            # components
            freq_comp = np.prod([max(single_freq.at[n], 1e-9) for n in chosen])
            cond_comp = 1.0
            for a,b in combinations(chosen,2):
                cond_comp *= (pair_freq.at[a,b] if pair_freq.at[a,b] > 0 else 1e-9)
            markov_s = markov_score_for_seed(transitions, recent_two)
            markov_comp = np.prod([max(markov_s.at[n], 1e-9) for n in chosen])
            coocc_boost = 1.0
            max_pair = pair_freq.values.max() if pair_freq.values.size>0 else 1.0
            for a,b in combinations(chosen,2):
                coocc_boost += (pair_freq.at[a,b] / (max_pair + 1e-12)) * delta

            # ML helpers
            X_test = np.array([[len(df) + 1]])
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

            final = (alpha * freq_comp) + (beta * cond_comp) + (gamma * markov_comp) + (delta * (coocc_boost - 1.0))
            final = final * (1.0 + nb_score) * (1.0 + gb_pred)

            # tiny pattern bonus for balanced ranges
            ranges = [0,0,0,0]
            for n in chosen:
                if n <= 8: ranges[0] += 1
                elif n <= 16: ranges[1] += 1
                elif n <= 24: ranges[2] += 1
                else: ranges[3] += 1
            if max(ranges) <= 2:
                final *= 1.05

            if final > best_score:
                best_score = final
                best_combo = chosen.copy()

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
                nb_dist = np.ones(14) / 14
            combined = 0.6 * jk_probs + 0.4 * nb_dist
            combined = combined / combined.sum()
            joker_pick = int(np.random.choice(range(1,15), p=combined))
        except Exception:
            joker_pick = int(np.random.randint(1,15))

        results.append({'Ana': sorted([int(x) for x in best_combo]), 'Joker': joker_pick, 'Score': float(best_score)})
    return results

# -------------------------
# UI: data load & controls
# -------------------------
st.title("🎯 Şans Topu Tahmin Botu v3 — Gelişmiş")

st.markdown("""
Bu uygulama:
- GitHub raw linkinden **CSV** veya yerel **CSV yükleme** ile veri alır.  
- Zaman-ağırlıklı frekans, birlikte çıkma, Markov (2-order) ve basit ML modellerini kullanarak **5+1** tahminler üretir.  
""")

with st.sidebar:
    st.header("Veri & Ayarlar")
    source = st.radio("Veri kaynağı:", ("GitHub Raw URL", "Dosya Yükle"))
    default_raw = "https://raw.githubusercontent.com/mhmtal04/SansTopu/main/sans_topu_ornek_veri.csv"
    raw_url = st.text_input("GitHub Raw CSV URL", value=default_raw)
    uploaded = st.file_uploader("Veya CSV dosyanızı yükleyin", type=["csv"])
    st.markdown("---")
    st.subheader("Tahmin Ayarları")
    n_sets = st.number_input("Kaç tahmin seti üretilsin?", min_value=1, max_value=10, value=4)
    trials = st.number_input("Her set için deneme sayısı", min_value=500, max_value=30000, value=8000, step=500)
    halflife = st.number_input("Zaman ağırlığı (halflife gün)", min_value=30, max_value=1000, value=180)
    alpha = st.slider("Frekans ağırlığı (alpha)", 0.0, 1.0, 0.35)
    beta = st.slider("Koşullu ağırlık (beta)", 0.0, 1.0, 0.25)
    gamma = st.slider("Markov ağırlığı (gamma)", 0.0, 1.0, 0.25)
    delta = st.slider("Co-occurrence ağırlığı (delta)", 0.0, 1.0, 0.15)
    st.markdown("---")
    if st.button("Veriyi Yükle ve Temizle"):
        try:
            if source == "GitHub Raw URL":
                raw = read_csv_from_url(raw_url)
            else:
                if uploaded is None:
                    st.error("Lütfen önce bir dosya seçin.")
                    raw = None
                else:
                    raw = read_csv_file(uploaded)
            if raw is not None:
                cols = [c.strip().lower() for c in raw.columns]
                if not any('date' in c for c in cols):
                    st.error("CSV içinde 'Date' sütunu bulunamadı. Format: Date, Num1, Num2, Num3, Num4, Num5, Joker")
                else:
                    # try to standardize / find number columns
                    mapping = {}
                    for c in raw.columns:
                        lc = c.strip().lower()
                        if 'date' in lc:
                            mapping[c] = 'Date'
                        elif 'joker' in lc:
                            mapping[c] = 'Joker'
                        # leave other names; we'll detect numerics
                    raw = raw.rename(columns=mapping)
                    num_candidates = [c for c in raw.columns if c not in ['Date','Joker'] and pd.api.types.is_numeric_dtype(raw[c])]
                    if len(num_candidates) < 5:
                        cols_list = list(raw.columns)
                        try:
                            date_idx = cols_list.index('Date')
                            num_candidates = cols_list[date_idx+1:date_idx+6]
                        except Exception:
                            num_candidates = cols_list[:5]
                    # ensure Joker present
                    if 'Joker' not in raw.columns:
                        cols_list = list(raw.columns)
                        if len(cols_list) >= 6:
                            raw = raw.rename(columns={cols_list[5]: 'Joker'})
                        else:
                            raw['Joker'] = pd.NA
                    clean = pd.DataFrame()
                    clean['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
                    # pick first five numeric candidates
                    clean['Num1'] = pd.to_numeric(raw[num_candidates[0]], errors='coerce').astype('Int64')
                    clean['Num2'] = pd.to_numeric(raw[num_candidates[1]], errors='coerce').astype('Int64')
                    clean['Num3'] = pd.to_numeric(raw[num_candidates[2]], errors='coerce').astype('Int64')
                    clean['Num4'] = pd.to_numeric(raw[num_candidates[3]], errors='coerce').astype('Int64')
                    clean['Num5'] = pd.to_numeric(raw[num_candidates[4]], errors='coerce').astype('Int64')
                    clean['Joker'] = pd.to_numeric(raw['Joker'], errors='coerce').astype('Int64')
                    clean = clean.dropna(subset=['Date']).reset_index(drop=True)
                    clean['Numbers'] = clean[['Num1','Num2','Num3','Num4','Num5']].values.tolist()
                    st.session_state['df'] = clean
                    # clear cached computations dependent on earlier df by resetting cache implicitly (cache uses params)
                    st.success("✅ Veri yüklendi ve temizlendi. Önizleme aşağıda.")
        except Exception as e:
            st.error(f"Hata: {e}")

# -------------------------
# If data loaded: compute & UI
# -------------------------
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("Veri Önizleme (son 10)")
    st.dataframe(df.tail(10))

    # compute analyses (cached)
    single_freq = compute_single_freq(df, halflife_days=halflife)
    pair_freq = compute_pair_freq(df, halflife_days=halflife)
    transitions = compute_markov_order2(df)
    nb, gb, joker_freq, joker_nb = train_models(df)

    # show stats
    st.subheader("Temel İstatistikler")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.write("Ana sayı - zaman ağırlıklı frekans")
        st.bar_chart(single_freq)
        st.write(single_freq.sort_values(ascending=False).head(10))
    with c2:
        st.write("Joker dağılımı")
        st.bar_chart(pd.Series(joker_freq.values, index=range(1,15)))
        st.write(pd.Series(joker_freq.values, index=range(1,15)).sort_values(ascending=False).head(5))
    with c3:
        st.metric("Toplam çekiliş", len(df))

    st.subheader("Birlikte Çıkma - En Sık Çiftler")
    pairs = []
    for a in range(1,35):
        for b in range(a+1,35):
            pairs.append(((a,b), pair_freq.at[a,b]))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)[:20]
    top_pairs = pd.DataFrame([{'Çift': f'{p[0][0]} & {p[0][1]}', 'Sıklık': p[1]} for p in pairs_sorted])
    st.table(top_pairs)

    # keep preds in session_state to avoid reset
    if 'preds' not in st.session_state:
        st.session_state['preds'] = None

    if st.button("Tahmin Üret"):
        with st.spinner("Tahminler hesaplanıyor..."):
            preds = generate_predictions(df, single_freq, pair_freq, transitions, nb, gb, joker_freq, joker_nb,
                                         n_sets=n_sets, trials=int(trials), alpha=alpha, beta=beta, gamma=gamma, delta=delta)
            st.session_state['preds'] = preds

    if st.session_state['preds'] is not None:
        st.subheader("Tahminler (arayüzde gösteriliyor)")
        for i, p in enumerate(st.session_state['preds'], start=1):
            st.markdown(f"**{i}. Tahmin**")
            st.write("Ana sayılar: " + ", ".join(f"{x:02d}" for x in p['Ana']))
            st.write("Joker: " + f"{p['Joker']:02d}")
            st.caption(f"Model skoru: {p['Score']:.3e}")

    if st.checkbox("Co-occurrence heatmap (önizleme)"):
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(pair_freq.values, aspect='auto')
        ax.set_title("Co-occurrence matris (1..34)")
        ax.set_xlabel("Sayı")
        ax.set_ylabel("Sayı")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

else:
    st.info("Soldan 'Veriyi Yükle ve Temizle' butonunu kullanarak GitHub raw URL veya dosya yükle yapın.")

# --- end of file ---
