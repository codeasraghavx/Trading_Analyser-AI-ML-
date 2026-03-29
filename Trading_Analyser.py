import streamlit as st
import yfinance as yf
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="AI Multi-Stock Analyzer", layout="wide")

st.title("📊 AI Stock Analysis Dashboard")
st.markdown("Developed for **CSA2001: AI/ML Project**")

st.sidebar.header("1. Select Asset")

ticker_choice=st.sidebar.selectbox(
    "Choose a Stock",
    options=["AAPL (Apple)", "TSLA(Tesla)", "NVDA(Nvidia)", "MSFT ( Microsoft)", "CUSTOM"]
)

if ticker_choice=="CUSTOM":
    ticker_symbol=st.sidebar.text_input("Enter Custom Ticker", value="GOOGL").upper()
else:
    ticker_symbol=ticker_choice.split(" ")[0]

st.sidebar.markdown("---")
st.sidebar.header("2.Model Controls")

forecast_out= st.sidebar.slider("Forecast Days(Lookahead)",1,30,1)

n_trees=st.sidebar.slider("Random Forest Trees",10,500,100)

@st.cache_data
def get_data(ticker,period="5y"):
    df=yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    return df

data=get_data(ticker_symbol)

if data.empty:
    st.error("Data not found. Please check the ticker symbol")
else:
    ma_window=20
    data['MA_User']=data['Close'].rolling(window=ma_window).mean()
    data['Volatility']= data['Close'].pct_change().rolling(window=10).std()

    delta=data['Close'].diff()
    gain=(delta.where(delta>0,0)).rolling(window=14).mean()
    loss=(-delta.where(delta<0,0)).rolling(window=14).mean()
    data['RSI']=100-(100/(1+(gain/loss)))

    data['Target']=data['Close'].shift(-forecast_out)
    df=data.dropna().copy()

    features=['Close','MA_User','Volatility','RSI','Volume']
    X=df[features]
    y=df['Target']

    split=int(len(df)*0.8)
    X_train, X_test=X[:split],X[split:]
    y_train, y_test=y[:split],y[split:]

    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled= scaler.transform(X_test)

    model=RandomForestRegressor(n_estimators=n_trees, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds=model.predict(X_test_scaled)
    last_row_scaled=scaler.transform(X.tail(1))
    future_price=model.predict(last_row_scaled)[0]

    col1,col2,col3=st.columns(3)
    current=df['Close'].iloc[-1]
    col1.metric(f"Current{ticker_symbol}",f"${current:.2f}")
    col2.metric(f"AI Prediction({forecast_out}Day)",f"${future_price:.2f}",f"{future_price-current:.2f}")
    col3.metric("Model R² Score",f"{r2_score(y_test,preds):.3f}")

    st.subheader(f"Price Trend & AI Forecast: {ticker_symbol}")
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(df.index[split:], y_test, label="actual Price", color="#2ecc71")
    ax.plot(df.index[split:], preds, label="AI Prediciton", color="#e74c3c", linestyle="--")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Decision Factor Analysis (XAI)")
    importance_df=pd.DataFrame({'Feature':features,'Importance':model.feature_importances_})
    st.bar_chart(importance_df.set_index('Feature'))

    st.success(f"Dashboard updated for {ticker_symbol} using {n_trees} trees.")