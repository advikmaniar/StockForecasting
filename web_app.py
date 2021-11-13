import streamlit as st
st. set_page_config(layout="wide", page_icon=":chart:")
st.set_option('deprecation.showPyplotGlobalUse', False)
s = f"""
<style>
div.stButton > button:first-child {{ border: 3px solid {"#00FFFF"}; border-radius:5px 5px 5px 5px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)
import SessionState
import pandas as pd
from plotly import graph_objs as go
import matplotlib.pyplot as plt
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from tensorflow.python.keras.models import model_from_json
from util import csv_to_dataset, history_points

col_1,col_2,col_3,col_4,col_5,col_6 = st.beta_columns((1.5,1,1.5,1,1.2,1))
col_1.image("icons/apple.png")
col_2.image("icons/google.png")
col_3.image("icons/netflix.png")
col_4.image("icons/tesla.png")
col_5.image("icons/amazon.png")
col_6.image("icons/facebook.png")

st.markdown("<h1 style='text-align: center;'><u><b>Stock Forecasting</b></u> </h1>",unsafe_allow_html=True)


stocks = ("GOOGL","AAPL","MSFT","IBM","TSLA","NFLX","AMZN","GME","PEP","ORCL","FB")

st.sidebar.title("Some Popular Stocks")
selected_stock = st.sidebar.selectbox("Select a stock: ",stocks)

#Predicting years
n_years = st.sidebar.slider("How many years do you want to predict?",1,5)
period = n_years*365

#Load data
@st.cache
def load_data(ss):
    data = pd.read_csv("datasets/"+ss+"_daily.csv")
    return data

data_load_state = st.info("Loading Raw Data.....")
data=load_data(selected_stock)
st.subheader("Stock Selected - "+selected_stock)
st.subheader("Raw Data")
st.write(data)


#Plot raw data
def plot_raw(data):

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=data["date"],y=data["1. open"],name="Open"))
    fig.add_trace(go.Scatter(x=data["date"], y=data["4. close"], name="Close"))

    fig.update_layout(
        margin=dict(l=20, r=20, b=50, t=70),
        title_text="Time Series Data",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3,step="month",stepmode="backward"),
                    dict(count=6,step="month",stepmode="backward"),
                    dict(count=1,step="year",stepmode="backward"),
                    dict(count=2,step="year",stepmode="backward"),
                    dict(count=5,step="year",stepmode="backward"),
                    dict(count=10,step="year",stepmode="backward")])),
            rangeslider=dict(visible=True),
        ))
    st.plotly_chart(fig,use_container_width=True)

plot_raw(data)
data_load_state.info("Loading Raw Data.....Done!")

st.markdown("<hr style = 'height:3px;border-width:0;color:#00FFFF;background-color:#00FFFF'>",unsafe_allow_html=True)

#Plot forecasted data
@st.cache(suppress_st_warning=True)
def plot_forecasted(data,data1):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data["ds"],y=data["yhat"],name="Prediction"))
    fig.add_trace(go.Scatter(x=data1["date"],y=data1["4. close"],name="Actual"))

    fig.update_layout(
        margin=dict(l=20, r=20, b=50, t=70),
        title_text="Actual vs Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, step="month", stepmode="backward"),
                    dict(count=3,step="month",stepmode="backward"),
                    dict(count=6,step="month",stepmode="backward"),
                    dict(count=1,step="year",stepmode="backward"),
                    dict(count=2,step="year",stepmode="backward"),
                    dict(count=5,step="year",stepmode="backward")])),
            rangeslider=dict(visible=True),
        ))
    st.plotly_chart(fig,use_container_width=True)

#Forecasting using fbprophet
forecast_button=st.button("Click to Forecast!")
if forecast_button:

    data1=data.iloc[:2000].copy()
    df_train = data1[["date","4. close"]]
    df_train=df_train.rename(columns={"date": "ds","4. close": "y"})

    #Build and train prophet model
    @st.cache(allow_output_mutation=True)
    def train_fb_model(df_train):
        fb_mod = Prophet()
        fb_mod.fit(df_train)
        return fb_mod

    fb_mod = train_fb_model(df_train)
    #Make Predictions
    future = fb_mod.make_future_dataframe(periods=period)
    forecast = fb_mod.predict(future)

    st.markdown("<h1 style='text-align: center;'><u><b>Forecast</b></u> </h1>",unsafe_allow_html=True)
    st.write(forecast)
    plot_forecasted(forecast,data1)

    comp_button = st.button("Show Components!")
    if comp_button:
        st.subheader("Forecast Components")
        fig2 = fb_mod.plot_components(forecast)
        st.write(fig2)
#----------------------------------------------------------------------------------------------------------------------#
# ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(selected_stock+"_daily.csv")
# plt.plot(next_day_open_values)
# plt.show()

#Plot loss functions
def plot_loss(fitted,string):
    plt.plot(fitted.history[string],lw=3,c="blue")
    plt.title("Loss Function")
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string])
    plt.grid(True)
    plt.show()

#Load model
def load_model():
    with open("C:/Users/User/PycharmProjects/Stock_Prediction/basic_model_json.json","r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights('C:/Users/User/PycharmProjects/Stock_Prediction/basic_model.h5')
    loaded_model.make_predict_function()

    return loaded_model

