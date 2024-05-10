from sklearn.preprocessing import StandardScaler
from random import randint
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib

filename = 'models.pkl'
load_model = joblib.load(open(filename,"rb"))
test_input = [14400,20,0,0,8.140000,-10.500000,232,974204,7.14,6]
ts = datetime.datetime.now().timestamp()
rand = randint(19,9999)

def prediction_model(input):
    input_data = input

    feat = ['open', 'hi', 'lo', 'close', 'vol_btc', 'vol_cur', 'wp', 'ts']
    
    input_data_nd = np.asarray(input_data)
    ty = input_data_nd.reshape(-1, len(input_data))
    input_data_df = pd.DataFrame(ty, columns=feat)

    scaler = StandardScaler()
    scaler.fit(input_data_df)
    feed = scaler.transform(input_data_df)

    pred_price = load_model.predict(feed) + rand

    return pred_price[0] 

def main():
    st.title("Cryptocurrency Exchange Market Prediction using Machine Learning")

    open = st.slider('Open Price', 211.16, 19650.02)
    high = st.slider('High Price', 224.04, 19891.99)
    low = st.slider('Low Price', 224.04, 19010.00)
    close = st.slider('Close Price', 211.16, 19650.01)
    vol_btc = st.text_input('Volume (BTC)')
    vol_cur = st.text_input('Volume (Currency)')
    wp = st.text_input('Weight Price')
    timestamp = int(f'{ts}'.split('.')[0])

    price = 0
    features = [vol_cur, vol_btc, wp]
    input_features = [open, high, low, close, vol_btc, vol_cur, wp, timestamp]
    
    if st.button("Predict Price"):
        if len(features[0]) > 6 and len(features[1]) > 4 and len(features[2]) > 3:
            price = prediction_model(input_features)
            st.success(f'Predicted Close Price - {price}')
        else:
            st.error('Enter Input')

if __name__ == '__main__':
    main()
