from fileinput import close
from ipaddress import summarize_address_range
import tensorflow as tf
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import streamlit as st
import yfinance as yf
import altair as alt

start='2017-05-18'
end='2022-05-14'

st.title('Ztockify' )

ticker_list = pd.read_csv("T5STOCKS.txt")
user_input = st.selectbox('Stock ticker', ticker_list)
df=pdr.DataReader(user_input,'yahoo',start,end)

#Ticker Information
tickerdata=yf.Ticker(user_input)
logo='<img src=%s>' % tickerdata.info['logo_url']
name=tickerdata.info['longName']
summary=tickerdata.info['longBusinessSummary']

#Describing Data
st.markdown(logo,unsafe_allow_html=True)
st.header('**%s**'%name)
st.write(summary,color='y')
st.subheader('Data from June 2017 - June 2022')
st.write(df.describe())

#Visulaizations
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel("Year")
plt.ylabel("Price")
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,color='green')
plt.plot(ma200,color='yellow')
plt.plot(df.Close)
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend(['MA100','MA200'],loc="upper left")
st.pyplot(fig)

#Scalling of Data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
dfc=scaler.fit_transform(np.array(df['Close']).reshape(-1,1))

#Splitting Data into Training and Testing
training_size=int(len(dfc)*0.65)
test_size=len(dfc)-training_size
train_data,test_data=dfc[0:training_size,:],dfc[training_size:len(dfc),:1] 

#convert an array of values into a dataset matrix
def datasetmatrix(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]  #i=0 , 0,1,2....199 200
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX),np.array(dataY)   

#reshape into X=t,t+1,t+2..t+n , Y=t+n+1
time_step=200
X_train,y_train=datasetmatrix(train_data,time_step)
X_test,y_test=datasetmatrix(test_data,time_step)    

#Load my model
user_input=user_input+'.h5'
model=load_model(user_input)

#Testing Part
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

#Inverse Scaling
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

# Plotting
# shift train predictions for plotting
back=200
trainpredictplt=np.empty_like(dfc)
trainpredictplt[:,:]=np.nan
trainpredictplt[back:len(train_predict)+back,:]=train_predict
# shift test predictions for plotting
testpredictplt=np.empty_like(dfc)
testpredictplt[:,:]=np.nan
testpredictplt[len(train_predict)+(back*2)+1:len(dfc)-1,:]=test_predict
# plot baseline and predictions
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(dfc))
plt.plot(trainpredictplt)
plt.plot(testpredictplt)
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(['Original'],loc="upper left")
st.pyplot(fig2)

#Previous 200 Days data
x_input=test_data[240:].reshape(1,-1)
temp_list=list(x_input)
temp_list=temp_list[0].tolist()

# Next 50 days prediction
from numpy import array
lst_output=[]
n_steps=200
i=0
while(i<50): 
    if(len(temp_list)>200):
        x_input=np.array(temp_list[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_list.extend(yhat[0].tolist())
        temp_list=temp_list[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((-1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_list.extend(yhat[0].tolist())
        print(len(temp_list))
        lst_output.extend(yhat.tolist())
        i=i+1 
  
#Plotting next 50 days
st.subheader('Next 50 days predictions')
#fig3=plt.figure(figsize=(12,6))
days_200=np.arange(1,201)
days_pred=np.arange(201,251)
df2=dfc.tolist()
df2.extend(lst_output)
df2=scaler.inverse_transform(df2).tolist()
lst=[item for item in range(1, 1308)]
df2=pd.DataFrame(df2,columns=["Price"])
df2["Time"]=lst
chart=alt.Chart(df2).mark_line().encode(x="Time",y="Price",tooltip=["Time","Price"])
st.altair_chart(chart,use_container_width=True)
#plt.plot(df2)  
#plt.xlabel("Time")
#plt.ylabel("Price")
#plt.legend(['Predicted(Smooth)'],loc="upper left")
#st.pyplot(fig3)