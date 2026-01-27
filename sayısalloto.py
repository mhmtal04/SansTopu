import streamlit as st
import pandas as pd
from collections import Counter
import itertools
import random

st.set_page_config(page_title="Sayısal Loto Diziliş Botu", layout="wide")

# ======================================================
# YARDIMCI FONKSİYONLAR
# ======================================================

def decade(n):
    return n // 10

def generate_pattern(numbers):
    decades = [decade(n) for n in numbers]
    pattern = []
    count = 1
    for i in range(1, len(decades)):
        if decades[i] == decades[i-1]:
            count += 1
        else:
            pattern.append(count)
            count = 1
    pattern.append(count)
    return "-".join(map(str, pattern))

# ======================================================
# VERİ OKUMA
# ======================================================

@st.cache_data
def load_data(csv_file):
    return pd.read_csv(csv_file)

# ======================================================
# DİZİLİŞ & ÖRÜNTÜ ANALİZİ
# ======================================================

def analyze_patterns(df):
    patterns = df.apply(lambda r: generate_pattern(r.values), axis=1)
    return patterns, Counter(patterns)

def pattern_scores(pattern_series, window=20):
    total = len(pattern_series)
    freq = pattern_series.value_counts() / total
    recent = pattern_series.tail(window)
    scores = {}

    for p in freq.index:
        absence = 1 if p not in recent.values else 0
        streak_penalty = -0.3 if len(recent) >= 2 and recent.iloc[-1] == recent.iloc[-2] == p else 0
        score = freq[p] * 0.5 + absence * 0.3 + streak_penalty
        scores[p] = round(score, 3)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

# ======================================================
# T1–T6 POZİSYON ANALİZİ
# ======================================================

def analyze_positions(df):
    stats = {}
    for i in range(6):
        stats[f"T{i+1}"] = Counter(df.iloc[:, i])
    return stats

# ======================================================
# SICAK / NÖTR / SOĞUK SAYILAR
# ======================================================

def hot_neutral_cold(df, recent_window=10):
    all_numbers = df.values.flatten()
    total_freq = Counter(all_numbers)

    recent = df.tail(recent_window).values.flatten()
    recent_freq = Counter(recent)

    hot = [n for n, c in recent_freq.items() if c >= 2]
    cold = [n for n, c in total_freq.items() if n not in recent_freq]
    neutral = [n for n in total_freq if n not in hot and n not in cold]

    return hot, neutral, cold

# ======================================================
# BİRLİKTE ÇIKAN SAYILAR
# ======================================================

def analyze_pairs(df):
    pairs = Counter()
    for _, row in df.iterrows():
        for a, b in itertools.combinations(sorted(row.values), 2):
            pairs[(a, b)] += 1
    return pairs

# ======================================================
# SAYI PUANLAMA MOTORU
# ======================================================

def score_number(n, position_counter, hot, neutral, cold, recent_numbers):
    score = 0

    if n in hot:
        score += 2
    elif n in neutral:
        score += 1
    elif n in cold:
        score += 1.5

    if n in recent_numbers:
        score -= 1

    score += position_counter.get(n, 0) / 50
    return score

# ======================================================
# KOLON ÜRETİCİ (ANA MOTOR)
# ======================================================

def generate_column(df, target_pattern):
    pos_stats = analyze_positions(df)
    hot, neutral, cold = hot_neutral_cold(df)
    pairs = analyze_pairs(df)
    recent_numbers = set(df.tail(2).values.flatten())

    pattern_blocks = list(map(int, target_pattern.split("-")))
    decades_list = []

    for block in pattern_blocks:
        decades_list.append(block)

    selected_numbers = []
    used_decades = []

    for i, block in enumerate(pattern_blocks):
        candidates = []
        pos = f"T{i+1}"

        for n in range(1, 91):
            if block > 1 and used_decades.count(decade(n)) >= block:
                continue

            s = score_number(
                n,
                pos_stats[pos],
                hot,
                neutral,
                cold,
                recent_numbers
            )

            candidates.append((n, s))

        candidates.sort(key=lambda x: x[1], reverse=True)

        chosen = []
        for n, _ in candidates:
            if len(chosen) < block:
                chosen.append(n)
                used_decades.append(decade(n))

        selected_numbers.extend(sorted(chosen))

    return sorted(set(selected_numbers))[:6]

# ======================================================
# STREAMLIT ARAYÜZ
# ======================================================

st.title("🎯 Sayısal Loto Diziliş & Örüntü Botu")

uploaded_file = st.file_uploader("CSV dosyanı yükle (6 sütun – S1…S6)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"{len(df)} çekiliş yüklendi")

    patterns, pattern_counts = analyze_patterns(df)
    scores = pattern_scores(patterns)

    st.subheader("⭐ Örüntüye Göre En Olası Dizilişler")
    st.dataframe(pd.DataFrame(scores.items(), columns=["Diziliş", "Skor"]).head(5))

    top_patterns = list(scores.keys())[:3]

    st.subheader("🎯 Önerilen Kolonlar")
    cols = st.columns(3)

    for i, pat in enumerate(top_patterns):
        with cols[i]:
            col = generate_column(df, pat)
            st.markdown(f"**Diziliş:** `{pat}`")
            st.markdown(f"🎲 **Kolon:** `{', '.join(map(str, col))}`")

    st.subheader("📍 T1–T6 En Çok Çıkan Sayılar")
    pos_stats = analyze_positions(df)
    pos_cols = st.columns(6)
    for i, (pos, counter) in enumerate(pos_stats.items()):
        with pos_cols[i]:
            st.markdown(f"**{pos}**")
            st.write(counter.most_common(5))

    hot, neutral, cold = hot_neutral_cold(df)

    st.subheader("🔥 Sıcak / ⚖️ Nötr / 🧊 Soğuk Sayılar")
    c1, c2, c3 = st.columns(3)
    c1.write(sorted(hot))
    c2.write(sorted(neutral)[:15])
    c3.write(sorted(cold)[:15])
