import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.title("📊 Smart Investment Advisor using P/E Ratio")

st.markdown("### 💡 This tool helps investors estimate returns based on valuation (P/E ratio)")

# Upload dataset
uploaded_file = st.file_uploader("Upload historical dataset", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'PE' not in df.columns or 'Return (%)' not in df.columns:
        st.error("Dataset must contain 'PE' and 'Return (%)'")
    else:
        # Train model
        X = sm.add_constant(df['PE'])
        y = df['Return (%)']
        model = sm.OLS(y, X).fit()

        st.success("Model trained successfully ✅")

        # --- USER INPUT ---
        st.subheader("📥 Enter Investment Details")

        user_pe = st.number_input("Enter P/E Ratio of stock", min_value=1.0, value=20.0)
        investment = st.number_input("Enter Investment Amount (₹)", min_value=1000.0, value=10000.0)

        # Prediction
        pred_return = model.predict([1, user_pe])[0]

        expected_value = investment * (1 + pred_return/100)
        profit = expected_value - investment

        # --- OUTPUT ---
        st.subheader("📊 Prediction Results")

        st.write(f"📈 Expected Return: **{pred_return:.2f}%**")
        st.write(f"💰 Expected Value: ₹ **{expected_value:.2f}**")
        st.write(f"📊 Profit / Loss: ₹ **{profit:.2f}**")

        # --- ADVICE ---
        st.subheader("📌 Investment Advice")

        if pred_return > 10:
            st.success("✅ Good investment opportunity (Undervalued stock)")
        elif pred_return > 0:
            st.warning("⚠ Moderate returns expected")
        else:
            st.error("❌ Overvalued stock - Risky investment")

        # --- GRAPH ---
        st.subheader("📉 Market Trend Visualization")

        m, b = np.polyfit(df['PE'], df['Return (%)'], 1)

        plt.figure()
        plt.scatter(df['PE'], df['Return (%)'])
        plt.plot(df['PE'], m*df['PE'] + b)
        plt.xlabel("P/E Ratio")
        plt.ylabel("Return (%)")
        plt.title("Market Relationship")
        st.pyplot(plt)

else:
    st.info("👆 Upload your dataset to start analysis")