import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Price Trend Prediction')

selected_stock = st.text_input('Enter stock symbol (e.g., GOOG, AAPL, MSFT, GME):')

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if data.empty:
        st.error(f"No data found for the ticker symbol: {ticker}")
        return None
    data.reset_index(inplace=True)
    return data

if selected_stock:
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    if data is not None:
        data_load_state.text('Loading data... done!')

        st.subheader('Raw data')
        st.write(data.tail())

        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader('Predicted Data')
        st.write(forecast.tail())

        st.subheader(f'Prediction of {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.subheader("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)
    else:
        st.error('Data could not be loaded. Please try again.')
else:
    st.warning('Please enter a stock symbol.')
