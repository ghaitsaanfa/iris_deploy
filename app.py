import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load('model.pkl')
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Data Description", "Prediction", "About"])

# Page: Data Description
if page == "Data Description":
    st.title("🌸 Iris Dataset Description")
    st.write("This dataset contains measurements of iris flowers.")
    st.dataframe(df.head())

    st.subheader("Class Distribution")
    st.bar_chart(df['target_name'].value_counts())

    st.subheader("Feature Correlation")
    corr = df.iloc[:, :4].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Page: Prediction
elif page == "Prediction":
    st.title("🔍 Iris Species Prediction")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        predicted_class = iris.target_names[prediction]
        st.success(f"The predicted species is: **{predicted_class}** 🌺")

# Page: About
elif page == "About":
    st.title("📘 About this App")
    st.write("""
        - **Model**: Naive Bayes (GaussianNB)
        - **Data**: Iris Flower Dataset from scikit-learn
        - **App**: Built with Streamlit
        - **Created by**: [Your Name]
    """)
    st.markdown("---")
    st.write("Feel free to customize this app or extend it with more features!")

