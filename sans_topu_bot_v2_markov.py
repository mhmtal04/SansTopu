import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor

# --- Yardımcı Fonksiyonlar ---
def get_weights(dates):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    max_days = days_ago.max() + 1
    return (max_days - days_ago) / max_days

def weighted_single_probabilities(df):
    weights = get_weights(df['Date'])
    total_weight = weights.sum()
    freq = pd.Series(0, index=range(1, 35), dtype=float)
    for idx, row in df.iterrows():
        for n in row[['Num1','Num2','Num3','Num4','Num5']]:
            freq[n] += weights[idx]
    return freq / total_weight

def pair_frequencies(df):
    weights = get_weights(df['Date'])
    pair_freq = pd.DataFrame(0, index=range(1,35), columns=range(1,35), dtype=float)
    for idx, row in df.iterrows():
        nums = row[['Num1','Num2','Num3','Num4','Num5']]
        for a,b in combinations(nums,2):
            pair_freq.at[a,b]+=weights[idx]
            pair_freq.at[b,a]+=weights[idx]
    return pair_freq

def conditional_probabilities(single_prob, pair_freq):
    cond_prob = pd.DataFrame(0, index=range(1,35), columns=range(1,35), dtype=float)
    for a in range(1,35):
        if single_prob[a]>0:
            cond_prob.loc[a] = pair_freq.loc[a]/single_prob[a]
    return cond_prob

def markov_chain(df):
    transitions = np.zeros((35,35))
    for i in range(1,len(df)):
        prev = df.iloc[i-1][['Num1','Num2','Num3','Num4','Num5']]
        curr = df.iloc[i][['Num1','Num2','Num3','Num4','Num5']]
        for a in prev:
            for b in curr:
                transitions[a-1][b-1]+=1
    row_sums = transitions.sum(axis=1, keepdims=True)
    return np.divide(transitions,row_sums,out=np.zeros_like(transitions),where=row_sums!=0)

def train_naive_bayes(df):
    X = np.repeat(df.index.values.reshape(-1,1),5,axis=0)
    y = np.array([n for row in df[['Num1','Num2','Num3','Num4','Num5']].values for n in row])
    model = GaussianNB()
    model.fit(X,y)
    return model

def train_gradient_boost(df):
    X = np.repeat(df.index.values.reshape(-1,1),5,axis=0)
    y = np.array([n for row in df[['Num1','Num2','Num3','Num4','Num5']].values for n in row])
    model = GradientBoostingRegressor()
    model.fit(X,y)
    return model

def generate_predictions(df, single_prob, cond_prob, nb_model, gb_model, markov_probs, pair_freq, n_preds=3, trials=5000):
    predictions=[]
    numbers=list(range(1,35))
    probs=single_prob.values
    for _ in range(n_preds):
        best=None; best_score=-1
        for __ in range(trials):
            chosen=np.random.choice(numbers,size=5,replace=False,p=probs/probs.sum())
            chosen=np.sort(chosen)
            combo_score=np.prod([single_prob[n] for n in chosen])
            for a,b in combinations(chosen,2):
                combo_score*=cond_prob.at[a,b]
            X_test=np.array([[len(df)+1]])
            nb_pred=nb_model.predict(X_test)[0]
            gb_pred=gb_model.predict(X_test)[0]
            markov_score=np.mean([markov_probs[a-1].mean() for a in chosen])
            score=combo_score*(1+nb_pred/35)*(1+gb_pred/35)*(1+markov_score)
            if score>best_score:
                best_score=score
                best=chosen
        predictions.append(best)
    return predictions

# --- Streamlit Arayüz ---
def main():
    st.title("🎯 Şans Topu | Gelişmiş Tahmin Botu (Markov + Bayes + Boosting)")

    st.markdown("Veri kaynağını seçin:")
    option = st.radio("Kaynak Seç:", ("📂 Dosya Yükle", "🌐 GitHub Raw Link"))

    df = None

    if option == "📂 Dosya Yükle":
        uploaded = st.file_uploader("CSV dosyanızı yükleyin (Date, Num1~Num5, Joker)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
    else:
        url = st.text_input("GitHub Raw CSV URL’sini girin:")
        if url:
            df = pd.read_csv(url)

    if df is not None:
        try:
            df['Date']=pd.to_datetime(df['Date'])
            st.success(f"✅ Veriler yüklendi ({len(df)} çekiliş)")
            st.dataframe(df.tail(10))

            with st.spinner("🧠 Modeller eğitiliyor..."):
                single_prob=weighted_single_probabilities(df)
                pair_freq=pair_frequencies(df)
                cond_prob=conditional_probabilities(single_prob,pair_freq)
                markov_probs=markov_chain(df)
                nb_model=train_naive_bayes(df)
                gb_model=train_gradient_boost(df)

            n_preds=st.number_input("🎲 Kaç tahmin üretmek istersiniz?",1,10,3)

            if st.button("🚀 Tahmin Üret"):
                with st.spinner("Tahminler hesaplanıyor..."):
                    preds=generate_predictions(df,single_prob,cond_prob,nb_model,gb_model,markov_probs,pair_freq,n_preds)
                st.success("🎉 Tahminler hazır!")
                for i,p in enumerate(preds):
                    st.write(f"**{i+1}. Tahmin:** {', '.join(map(str,p))}")

        except Exception as e:
            st.error(f"❌ Hata: {e}")

if __name__ == "__main__":
    main()
