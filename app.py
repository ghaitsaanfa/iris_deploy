import streamlit as st
import pandas as pd
import numpy as np
import joblib # Untuk memuat model .pkl
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB # Sebagai fallback jika model.pkl tidak ditemukan
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Configuration ---
st.set_page_config(
    page_title="Iris Classifier Pro",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading (Cached) ---
@st.cache_data # Cache data loading
def load_data_iris():
    """Loads the Iris dataset and returns data object, DataFrame, feature names, and target names."""
    iris = load_iris()
    feature_names = iris.feature_names
    target_names = iris.target_names
    df = pd.DataFrame(iris.data, columns=feature_names)
    df['species_id'] = iris.target
    df['species_name'] = df['species_id'].apply(lambda x: target_names[x])
    return iris, df, feature_names, target_names

# --- Model Loading (Cached) ---
@st.cache_resource # Cache model object
def load_custom_model(model_path="naive_bayes_model.pkl"):
    """Loads a pre-trained model from a .pkl file."""
    try:
        model = joblib.load(model_path)
        st.sidebar.success(f"✔️ Model '{model_path}' berhasil dimuat.")
        return model
    except FileNotFoundError:
        st.sidebar.error(f"⚠️ File model '{model_path}' tidak ditemukan!")
        st.sidebar.warning("Menggunakan model Naive Bayes default yang dilatih saat ini.")
        # Fallback: latih model baru jika file .pkl tidak ditemukan
        iris_temp, _, _, _ = load_data_iris()
        X_temp = iris_temp.data
        y_temp = iris_temp.target
        fallback_model = GaussianNB()
        fallback_model.fit(X_temp, y_temp)
        return fallback_model
    except Exception as e:
        st.sidebar.error(f"❌ Error saat memuat model: {e}")
        st.sidebar.warning("Menggunakan model Naive Bayes default yang dilatih saat ini.")
        iris_temp, _, _, _ = load_data_iris()
        X_temp = iris_temp.data
        y_temp = iris_temp.target
        fallback_model = GaussianNB()
        fallback_model.fit(X_temp, y_temp)
        return fallback_model

# --- Load Data and Model ---
iris_data_obj, iris_df, feature_names, target_names = load_data_iris()
# Ganti "model.pkl" jika nama file model Anda berbeda
model = load_custom_model("model.pkl")


# --- Page 1: Data Description ---
def page_data_description():
    st.header("📊 Deskripsi Dataset Iris")
    st.markdown("""
    Dataset bunga Iris adalah dataset multivariat yang diperkenalkan oleh ahli statistik dan biologi Inggris, Ronald Fisher.
    Ini adalah dataset klasik dalam machine learning dan statistik, sering digunakan untuk menguji algoritma klasifikasi.
    Dataset ini berisi 150 sampel dari tiga spesies bunga Iris:
    - Iris setosa
    - Iris versicolor
    - Iris virginica

    Untuk setiap sampel, empat fitur diukur:
    1.  Panjang Sepal (cm)
    2.  Lebar Sepal (cm)
    3.  Panjang Petal (cm)
    4.  Lebar Petal (cm)
    """)

    st.subheader("Contoh Data (10 Baris Pertama)")
    st.dataframe(iris_df.head(10))

    st.subheader("Statistik Deskriptif")
    st.write(iris_df[feature_names].describe())

    st.subheader("Distribusi Spesies")
    species_counts = iris_df['species_name'].value_counts()
    st.bar_chart(species_counts)

    st.subheader("Heatmap Korelasi Fitur")
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(iris_df[feature_names].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)
    plt.clf()

    st.subheader("Pair Plot Fitur (Diwarnai berdasarkan Spesies)")
    fig_pair = sns.pairplot(iris_df, hue='species_name', vars=feature_names, palette={'setosa':'red', 'versicolor':'green', 'virginica':'blue'})
    st.pyplot(fig_pair)
    plt.clf()

    st.subheader("Distribusi Fitur berdasarkan Spesies")
    for feature in feature_names:
        fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
        sns.kdeplot(data=iris_df, x=feature, hue='species_name', fill=True, ax=ax_dist, palette={'setosa':'red', 'versicolor':'green', 'virginica':'blue'})
        ax_dist.set_title(f"Distribusi {feature} berdasarkan Spesies")
        st.pyplot(fig_dist)
        plt.clf()


# --- Page 2: Prediction ---
def page_prediction():
    st.header("🔮 Prediksi Spesies Iris")
    st.markdown("Gunakan slider untuk memasukkan ukuran bunga dan memprediksi spesiesnya.")

    input_data = {}
    col1, col2 = st.columns(2)

    with col1:
        input_data[feature_names[0]] = st.slider(
            f"{feature_names[0].replace(' (cm)', '')} (cm)",
            float(iris_df[feature_names[0]].min()),
            float(iris_df[feature_names[0]].max()),
            float(iris_df[feature_names[0]].mean())
        )
        input_data[feature_names[1]] = st.slider(
            f"{feature_names[1].replace(' (cm)', '')} (cm)",
            float(iris_df[feature_names[1]].min()),
            float(iris_df[feature_names[1]].max()),
            float(iris_df[feature_names[1]].mean())
        )
    with col2:
        input_data[feature_names[2]] = st.slider(
            f"{feature_names[2].replace(' (cm)', '')} (cm)",
            float(iris_df[feature_names[2]].min()),
            float(iris_df[feature_names[2]].max()),
            float(iris_df[feature_names[2]].mean())
        )
        input_data[feature_names[3]] = st.slider(
            f"{feature_names[3].replace(' (cm)', '')} (cm)",
            float(iris_df[feature_names[3]].min()),
            float(iris_df[feature_names[3]].max()),
            float(iris_df[feature_names[3]].mean())
        )

    input_features = np.array([list(input_data.values())])

    if st.button("Prediksi Spesies", type="primary", use_container_width=True):
        if model: # Pastikan model berhasil dimuat
            prediction_id = model.predict(input_features)[0]
            prediction_name = target_names[prediction_id]
            probabilities = model.predict_proba(input_features)[0]

            st.subheader("Hasil Prediksi:")
            st.success(f"Spesies yang diprediksi adalah: **{prediction_name.capitalize()}** 🌺")

            if prediction_name == 'setosa':
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/320px-Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris Setosa", width=200)
            elif prediction_name == 'versicolor':
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg", caption="Iris Versicolor", width=200)
            elif prediction_name == 'virginica':
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/320px-Iris_virginica.jpg", caption="Iris Virginica", width=200)

            st.subheader("Probabilitas Prediksi:")
            prob_df = pd.DataFrame({
                'Spesies': [name.capitalize() for name in target_names],
                'Probabilitas': probabilities
            })
            prob_df = prob_df.sort_values(by='Probabilitas', ascending=False).reset_index(drop=True)
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
        else:
            st.error("Model tidak tersedia untuk prediksi. Silakan periksa pesan error di sidebar.")


# --- Page 3: About ---
def page_about():
    st.header("💡 Tentang Aplikasi & Model Ini")
    st.markdown("""
        Aplikasi web Streamlit ini mendemonstrasikan klasifikasi spesies bunga Iris menggunakan model **Naive Bayes**.
        
        ### Fitur Utama:
        -   **Eksplorasi Data**: Lihat detail, statistik, dan visualisasi dari dataset Iris.
        -   **Prediksi Real-time**: Masukkan ukuran bunga untuk mendapatkan prediksi spesies instan beserta probabilitasnya.
        -   **Informasi Model**: Detail dasar tentang model yang digunakan.
    """)

    if model:
        st.subheader("Informasi Model yang Digunakan")
        model_name = model.__class__.__name__
        if "model.pkl" in st.session_state.get("model_load_message", "") and "berhasil dimuat" in st.session_state.get("model_load_message", ""): # Crude check
             st.info(f"Model yang digunakan adalah model pra-terlatih **{model_name}** yang dimuat dari `model.pkl`.")
        else:
             st.info(f"Model yang digunakan adalah **{model_name}** yang dilatih secara default oleh aplikasi ini.")

        st.markdown("Jika Anda ingin menguji performa model yang dimuat pada dataset Iris saat ini:")
        if st.button("Uji Model pada Data Iris Saat Ini"):
            X_all = iris_data_obj.data
            y_all_true = iris_data_obj.target
            try:
                y_all_pred = model.predict(X_all)
                acc = accuracy_score(y_all_true, y_all_pred)
                st.success(f"Akurasi model yang dimuat pada dataset Iris saat ini: **{acc*100:.2f}%**")
            except Exception as e:
                st.error(f"Tidak dapat mengevaluasi model: {e}")
    else:
        st.warning("Model tidak dapat dimuat. Tidak ada informasi model untuk ditampilkan.")


    st.markdown("---")
    st.markdown("Dibuat dengan ❤️ menggunakan Python, Streamlit, Scikit-learn, Pandas, Matplotlib, dan Seaborn.")
    st.markdown("Dikembangkan sebagai demonstrasi untuk klasifikasi Iris.")


# --- Main App Logic with Sidebar Navigation ---
def main():
    st.sidebar.title("🧭 Navigasi")
    page_options = {
        "📊 Deskripsi Data": page_data_description,
        "🔮 Prediksi": page_prediction,
        "💡 Tentang": page_about
    }
    selection = st.sidebar.radio("Pindah ke halaman:", list(page_options.keys()))

    # Simpan pesan status pemuatan model untuk referensi di halaman 'Tentang'
    # Ini adalah cara sederhana; state management yang lebih canggih bisa digunakan
    if 'model_load_message' not in st.session_state:
        st.session_state.model_load_message = ""
    # Cek pesan dari sidebar (jika ada)
    # Ini agak rumit karena pesan sidebar tidak langsung masuk ke session state
    # Cara yang lebih baik adalah jika load_custom_model mengembalikan status juga

    page_function = page_options[selection]
    page_function()

    st.sidebar.markdown("---")
    st.sidebar.info("Aplikasi ini menampilkan klasifikasi Naive Bayes untuk dataset Iris.")

if __name__ == "__main__":
    main()
