import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load iris data and train a simple model
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)

st.title("🌱 Simple Iris Species Predictor")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)
predict = st.button("Predict Species")
# Predict button
if predict:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(input_data)[0]
    species = iris.target_names[pred]
    st.success(f"Predicted Species: **{species}**")

st.markdown("---")
st.write("Adjust the input values and click 'Predict' to see the predicted iris species based on the Random Forest model.")