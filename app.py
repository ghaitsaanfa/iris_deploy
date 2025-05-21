import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os # Untuk path absolut

# --- Global Configuration ---
st.set_page_config(
    page_title="Iris Classifier Pro",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading (Cached) ---
@st.cache_data
def load_data_iris():
    iris = load_iris()
    feature_names = iris.feature_names
    target_names = iris.target_names
    df = pd.DataFrame(iris.data, columns=feature_names)
    df['species_id'] = iris.target
    df['species_name'] = df['species_id'].apply(lambda x: target_names[x])
    return iris, df, feature_names, target_names

# --- Model Loading with st.status (Cached) ---
@st.cache_resource
def load_model_with_status(model_path="naive_bayes_model.pkl"):
    """Loads a pre-trained model or falls back to a default one, using st.status for UI."""
    model = None
    iris_data_for_fallback, _, _, _ = load_data_iris() # Muat data jika butuh fallback

    # Inisialisasi session state jika belum ada
    if 'model_load_message' not in st.session_state:
        st.session_state.model_load_message = "Status pemuatan model belum diketahui."
    if 'model_type_loaded' not in st.session_state:
        st.session_state.model_type_loaded = "unknown" # 'custom', 'default_not_found', 'default_error'

    with st.status(f"Mencoba memuat model '{model_path}'...", expanded=True) as status_ui:
        try:
            abs_model_path = os.path.abspath(model_path)
            st.write(f"Mencari model di: {abs_model_path}")
            if not os.path.exists(abs_model_path):
                raise FileNotFoundError(f"File model tidak ditemukan di {abs_model_path}")

            model = joblib.load(abs_model_path)
            st.session_state.model_load_message = f"✔️ Model kustom '{model_path}' berhasil dimuat."
            st.session_state.model_type_loaded = "custom"
            status_ui.update(label=st.session_state.model_load_message, state="complete", expanded=False)

        except FileNotFoundError as fnf_err:
            st.write(f"Error: {fnf_err}. Melatih model Naive Bayes default sebagai fallback.")
            fallback_model = GaussianNB()
            fallback_model.fit(iris_data_for_fallback.data, iris_data_for_fallback.target)
            model = fallback_model
            st.session_state.model_load_message = f"⚠️ {fnf_err}. Model Naive Bayes default telah dilatih dan digunakan."
            st.session_state.model_type_loaded = "default_not_found"
            status_ui.update(label=st.session_state.model_load_message, state="warning", expanded=True)

        except Exception as e:
            st.write(f"Error saat memuat model: {e}. Melatih model Naive Bayes default sebagai fallback.")
            fallback_model = GaussianNB()
            fallback_model.fit(iris_data_for_fallback.data, iris_data_for_fallback.target)
            model = fallback_model
            st.session_state.model_load_message = f"❌ Terjadi error: {e}. Model Naive Bayes default telah dilatih dan digunakan."
            st.session_state.model_type_loaded = "default_error"
            status_ui.update(label=st.session_state.model_load_message, state="error", expanded=True)
    return model

# --- Load Data and Model ---
iris_data_obj, iris_df, feature_names, target_names = load_data_iris()

# Panggil fungsi pemuatan model. st.status akan muncul di sini saat pertama kali dijalankan.
# Status juga disimpan di st.session_state
model = load_model_with_status("model.pkl")


# --- Page 1: Data Description ---
def page_data_description():
    st.header("📊 Deskripsi Dataset Iris")
    st.markdown("...") # Konten deskripsi data sama seperti sebelumnya
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
    # ... Konten prediksi sama seperti sebelumnya ...

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
        if model:
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
            st.error("Model tidak tersedia untuk prediksi. Silakan periksa status pemuatan model.")


# --- Page 3: About ---
def page_about():
    st.header("💡 Tentang Aplikasi & Model Ini")
    st.markdown("""
        Aplikasi web Streamlit ini mendemonstrasikan klasifikasi spesies bunga Iris.
    """)

    st.subheader("Status Pemuatan Model")
    # Tampilkan pesan status dari session_state
    if st.session_state.model_type_loaded == "custom":
        st.success(st.session_state.model_load_message)
    elif st.session_state.model_type_loaded == "default_not_found":
        st.warning(st.session_state.model_load_message)
    elif st.session_state.model_type_loaded == "default_error":
        st.error(st.session_state.model_load_message)
    else:
        st.info(st.session_state.model_load_message) # fallback jika status tidak dikenali

    if model:
        st.subheader("Informasi Model yang Digunakan")
        model_class_name = model.__class__.__name__
        if st.session_state.model_type_loaded == "custom":
             st.write(f"Model yang digunakan adalah model kustom **{model_class_name}** yang dimuat dari `model.pkl`.")
        else:
             st.write(f"Model yang digunakan adalah model default **{model_class_name}** yang dilatih oleh aplikasi ini.")

        st.markdown("Anda dapat menguji performa model yang saat ini dimuat pada dataset Iris bawaan:")
        if st.button("Uji Akurasi Model Saat Ini"):
            X_all = iris_data_obj.data
            y_all_true = iris_data_obj.target
            try:
                y_all_pred = model.predict(X_all)
                acc = accuracy_score(y_all_true, y_all_pred)
                st.metric(label="Akurasi pada Dataset Iris (Full)", value=f"{acc*100:.2f}%")
            except Exception as e:
                st.error(f"Tidak dapat mengevaluasi model: {e}")
    else:
        st.warning("Model tidak dapat dimuat. Tidak ada informasi model untuk ditampilkan.")


    st.markdown("---")
    st.markdown("Dibuat dengan ❤️ menggunakan Python, Streamlit, Scikit-learn, Pandas, Matplotlib, dan Seaborn.")


# --- Main App Logic with Sidebar Navigation ---
def main():
    st.sidebar.title("🧭 Navigasi")
    page_options = {
        "📊 Deskripsi Data": page_data_description,
        "🔮 Prediksi": page_prediction,
        "💡 Tentang": page_about
    }
    selection = st.sidebar.radio("Pindah ke halaman:", list(page_options.keys()))

    page_function = page_options[selection]
    page_function()

    st.sidebar.markdown("---")
    st.sidebar.info("Aplikasi ini menampilkan klasifikasi Naive Bayes untuk dataset Iris.")

if __name__ == "__main__":
    main()
