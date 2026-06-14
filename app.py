import streamlit as st
import pandas as pd
import numpy as np
import ta
import joblib
import feedparser
from transformers import pipeline

# Load files
model = joblib.load("stock_model.pkl")
features = joblib.load("feature_columns.pkl")
encoder = joblib.load("stock_encoder.pkl")

@st.cache_resource
def load_finbert():
    
    classifier = pipeline(
        "text-classification",
        model="ProsusAI/finbert"
    )

    return classifier

finbert = load_finbert()


st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide"
)

st.title("📈 Stock Price Predictor")

uploaded_file = st.file_uploader(
    "Upload Stock CSV",
    type=["csv"]
)


def fetch_news(stock_name):

    query = stock_name.replace(
        ".NS",
        ""
    )

    url = (
        f"https://news.google.com/rss/search?"
        f"q={query}+stock"
    )

    feed = feedparser.parse(url)

    headlines = []

    for entry in feed.entries[:10]:

        headlines.append(
            entry.title
        )

    return headlines

def analyze_news_finbert(headlines):

    results = []

    for headline in headlines:

        prediction = finbert(
            headline
        )[0]

        results.append({

            "headline": headline,

            "label":
            prediction["label"],

            "score":
            prediction["score"]
        })

    return results

if uploaded_file:
    
    
    # choose stock
    st.subheader("Select Stock")

    stock_name = st.selectbox(
        "Choose Stock",
        list(encoder.classes_)
    )
    
    analyze = st.button(
        "Analyze Stock"
    )
    
    if analyze:
        
        with st.spinner(
            "Analyzing stock and news..."
        ):

            df = pd.read_csv(uploaded_file)
        
        numeric_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for col in numeric_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "")
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )
        
        df.dropna(inplace=True)


        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        required_cols = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing:

            st.error(
                f"Missing columns: {missing}"
            )

        else:

            close = df["Close"]

            for i in range(1,11):
                df[f"Lag_{i}"] = close.shift(i)

            df["SMA_10"] = close.rolling(10).mean()
            df["SMA_20"] = close.rolling(20).mean()

            df["EMA_10"] = close.ewm(
                span=10
            ).mean()

            df["EMA_20"] = close.ewm(
                span=20
            ).mean()

            df["RSI"] = ta.momentum.RSIIndicator(
                close
            ).rsi()

            macd = ta.trend.MACD(close)

            df["MACD"] = macd.macd()

            df["MACD_SIGNAL"] = (
                macd.macd_signal()
            )

            df["RETURN"] = (
                close.pct_change()
            )

            df["VOL_CHANGE"] = (
                df["Volume"].pct_change()
            )
            
            
            st.subheader(
                "Financial News Sentiment"
            )

            news = fetch_news(
                stock_name
            )

            results = analyze_news_finbert(
                news
            )
            
            if len(results) == 0:

                st.warning(
                    "No news articles found."
                )

            else:

                results = analyze_news_finbert(
                    news
                )

            positive = 0
            negative = 0
            neutral = 0

            for item in results:

                if item["label"] == "positive":
                    positive += 1

                elif item["label"] == "negative":
                    negative += 1

                else:
                    neutral += 1
                    
            col1,col2,col3 = st.columns(3)

            col1.metric(
                "Positive News",
                positive
            )

            col2.metric(
                "Neutral News",
                neutral
            )

            col3.metric(
                "Negative News",
                negative
            )
                        
            if positive > negative:

                overall = "Bullish"

            elif negative > positive:

                overall = "Bearish"

            else:

                overall = "Neutral"



            
            with st.expander(
                "View Financial Headlines"
            ):

                for item in results:

                    st.write(
                        item["headline"]
                    )

                    st.caption(
                        f"{item['label'].upper()} "
                        f"({item['score']:.2%})"
                    )
            
            
            try:

                stock_id = encoder.transform(
                    [stock_name]
                )[0]

            except:

                stock_id = 0

            df["Stock_ID"] = stock_id

            df.replace(
                [np.inf, -np.inf],
                np.nan,
                inplace=True
            )

            df.dropna(inplace=True)

            if len(df) > 0:

                latest = df.iloc[-1]

                X = pd.DataFrame(
                    [latest[features]]
                )

                prediction = (
                    model.predict(X)[0]
                )

                current_price = (
                    latest["Close"]
                )

                change = (
                    (prediction-current_price)
                    / current_price
                ) * 100
                
                if change > 1:
                    signal = "BUY"
                elif change < -1:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                    
 

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Current Price",
                    f"₹{current_price:.2f}"
                )

                col2.metric(
                    "Predicted Price",
                    f"₹{prediction:.2f}"
                )

                col3.metric(
                    "Expected Change %",
                    f"{change:.2f}%"
                )
                
                
                
                if signal == "BUY" and overall == "Bullish":
                    recommendation = "STRONG BUY"

                elif signal == "SELL" and overall == "Bearish":
                    recommendation = "STRONG SELL"

                elif signal == "BUY":
                    recommendation = "BUY"

                elif signal == "SELL":
                    recommendation = "SELL"

                else:
                    recommendation = "HOLD" 
                
                st.metric(
                    "Stock Price Recommendation",
                    signal
                )
                
                st.metric(
                    "News Sentiment",
                    overall
                )
                    
                st.metric(
                    "Final Recommendation",
                    recommendation
                )
                
                
                price_diff = prediction - current_price

                confidence = 100 - 2.22  # using model MAPE
                
                st.subheader("Prediction Insights")

                st.write(f"Price Difference: ₹{price_diff:.2f}")
                st.write("R² Score : 0.949")
                st.write("MAPE : 2.22%")
                st.write("Stocks Used : NIFTY 50")
                st.write("Training Period : 2014-2026")


            else:

                st.error(
                    "Not enough rows after feature engineering."
                )
                
                
                
                
            import plotly.express as px

            fig = px.line(
                df,
                x="Date",
                y="Close",
                title="Historical Close Price"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
            
