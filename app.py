import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="百家樂 AI 預測器", layout="centered")
st.title("🎲 百家樂 AI 預測器 v1.1")
st.markdown("請點選下方按鈕輸入歷史結果，然後按下「預測下一局」")

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🟥 莊 (B)"):
        st.session_state.history.append("B")
with col2:
    if st.button("🟦 閒 (P)"):
        st.session_state.history.append("P")
with col3:
    if st.button("🔄 清除"):
        st.session_state.history = []

history = st.session_state.history
if history:
    st.markdown(f"### 📋 當前紀錄：{' → '.join(history)}")
else:
    st.info("尚未輸入任何結果，請點選『莊』或『閒』輸入歷史資料。")

LOOKBACK = 8
if st.button("🔍 預測下一局"):
    if len(history) < LOOKBACK + 1:
        st.warning(f"請至少輸入 {LOOKBACK + 1} 局資料再進行預測。")
    else:
        X, y = [], []
        for i in range(LOOKBACK, len(history)):
            feature = [1 if x == 'B' else 0 for x in history[i - LOOKBACK:i]]
            label = 1 if history[i] == 'B' else 0
            X.append(feature)
            y.append(label)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)

        latest_feature = [1 if x == 'B' else 0 for x in history[-LOOKBACK:]]
        prediction = model.predict([latest_feature])[0]
        prob = model.predict_proba([latest_feature])[0]

        result = "🟥 莊 (B)" if prediction == 1 else "🟦 閒 (P)"
        st.success(f"✅ 預測下一局為：**{result}**")
        st.markdown(f"📊 機率：莊 `{prob[1]:.2f}`，閒 `{prob[0]:.2f}`")
