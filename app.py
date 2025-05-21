import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load iris dataset
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names
df = pd.DataFrame(iris.data, columns=feature_names)
df['species'] = [target_names[i] for i in iris.target]

# Load trained model
model = joblib.load("naive_bayes_model.pkl")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Data Description", "Prediction", "About"])

# Page 1: Data Description
if page == "Data Description":
    st.title("🌸 Iris Dataset Description")
    st.write("Dataset containing measurements of iris flowers and their species.")

    st.subheader("Data Sample")
    st.dataframe(df.head())

    st.subheader("Species Distribution")
    st.bar_chart(df['species'].value_counts())

    st.subheader("Feature Correlation")
    fig, ax = plt.subplots()
    sns.heatmap(df[feature_names].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Page 2: Prediction
elif page == "Prediction":
    st.title("🔮 Predict Iris Species")

    st.write("Input the flower measurements below:")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"The predicted species is: **{target_names[prediction]}** 🌺")

# Page 3: About
elif page == "About":
    st.title("📘 About this App")
    st.write("""
        This Streamlit web app is built to demonstrate classification of iris flowers using a Naive Bayes model.
        
        **Features:**
        - Explore the Iris dataset
        - Make real-time predictions
        - Learn more about the project

        **Model**: Gaussian Naive Bayes  
        **Library**: scikit-learn  
        **UI**: Streamlit
    """)
    st.markdown("---")
    st.write("Made with ❤️ using Python and Streamlit.")
