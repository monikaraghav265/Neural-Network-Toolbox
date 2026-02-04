import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import train_backprop, predict

st.write("App Started")
st.title("Neural Network Tool Box")

file = st.file_uploader("Upload CSV", type="csv")

epochs = st.slider("Epochs", 10, 200, 50)
lr = st.slider("Learning Rate", 0.01, 0.5, 0.1)

if file is not None:
    data = pd.read_csv(file)
    st.dataframe(data)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    if st.button("Train Model"):
        W1, W2, loss = train_backprop(X, y, epochs, lr)
        data["Prediction"] = predict(X, W1, W2)
        st.dataframe(data)

        plt.plot(loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        st.pyplot(plt)

