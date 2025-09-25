import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==== Paths ====
MODEL_PATH = r"D:\CattleBreedApp\ML_Model\cattle_breed_model.tflite"
LABELS_PATH = r"D:\CattleBreedApp\ML_Model\labels.txt"
BREED_INFO_PATH = r"D:\CattleBreedApp\ML_Model\breed_info.csv"

IMG_SIZE = (224, 224)
TOP_K = 5

# ==== Load TFLite Model ====
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# ==== Load Labels ====
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

# ==== Load Breed Info CSV ====
breed_info = pd.read_csv(BREED_INFO_PATH)

def normalize(name):
    return name.strip().replace(" ", "_").replace("-", "_").lower()

breed_info["Breed_norm"] = breed_info["Breed"].apply(normalize)
label_norm = [normalize(l) for l in labels]

# ==== Streamlit Page Config ====
st.set_page_config(page_title="Cattle Breed Identifier", page_icon="🐄", layout="centered")

# ==== Custom CSS for Theme ====
st.markdown(
    """
    <style>
    body {
        background-color: #e6f2ff;  /* Light blue background */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 24px;
        margin: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .prediction-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px #aaa;
        margin: 10px 0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==== Initialize History ====
if "history" not in st.session_state:
    st.session_state.history = []

# ==== Tabs ====
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📚 Browse Breeds", "🕘 History"])

# ================== HOME TAB ==================
with tab1:
    st.markdown("<h2 style='text-align:center; color:#004080;'>🐄 Cattle Breed Identifier</h2>", unsafe_allow_html=True)
    st.write("<p style='text-align:center; color:gray;'>Identify cattle breeds instantly with AI-powered recognition</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload or Capture Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Analyzing Image... Our AI is identifying the breed"):
            time.sleep(2)  # Fake loading

            # Preprocess
            img_proc = img.convert("RGB").resize(IMG_SIZE)
            arr = np.array(img_proc, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            # Run model
            interpreter.set_tensor(input_details["index"], arr.astype(input_details["dtype"]))
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details["index"])[0]

            # Softmax normalize
            exps = np.exp(preds - np.max(preds))
            probs = exps / np.sum(exps)

            top_idx = np.argsort(probs)[::-1][:TOP_K]
            best_breed = labels[top_idx[0]]

        # Save to history
        st.session_state.history.append(
            {"breed": best_breed, "confidence": f"{probs[top_idx[0]]*100:.2f}%", "time": time.strftime("%H:%M:%S")}
        )

        st.markdown(
            f"<div class='prediction-card'><b>✅ Predicted Breed:</b> {best_breed} ({probs[top_idx[0]]*100:.2f}%)</div>",
            unsafe_allow_html=True,
        )

        # ==== Charts for Top-5 ====
        st.subheader("📊 Top 5 Predictions")

        # Bar Chart
        fig, ax = plt.subplots()
        ax.barh([labels[i] for i in top_idx], [probs[i]*100 for i in top_idx], color="skyblue")
        ax.set_xlabel("Probability (%)")
        ax.set_ylabel("Breed")
        ax.set_title("Top 5 Predictions - Bar Chart")
        st.pyplot(fig)

        # Pie Chart
        fig2, ax2 = plt.subplots()
        ax2.pie([probs[i]*100 for i in top_idx], labels=[labels[i] for i in top_idx], autopct='%1.1f%%')
        ax2.set_title("Top 5 Predictions - Pie Chart")
        st.pyplot(fig2)

        # Alternate Predictions
        st.markdown("### 🔄 Alternate Predictions")
        for i in top_idx[1:]:
            st.write(f"- {labels[i]} ({probs[i]*100:.2f}%)")

        # Breed Info
        st.markdown("### 📘 Breed Information")
        best_norm = normalize(best_breed)
        if best_norm in list(breed_info["Breed_norm"]):
            info = breed_info[breed_info["Breed_norm"] == best_norm].iloc[0]

            food = info.get("Food", "N/A")
            climate = info.get("Climate", "N/A")
            milk = info.get("Milk_Production (L/day)", info.get("Milk_Production", "N/A"))
            diseases = info.get("Diseases", "N/A")
            lifespan = info.get("Life_Span (years)", info.get("Life_Span", "N/A"))
            region = info.get("Region", "N/A")

            st.markdown(
                f"""
                <div class="prediction-card">
                🌾 <b>Food:</b> {food} <br>
                🌦 <b>Climate:</b> {climate} <br>
                🥛 <b>Milk Production:</b> {milk} <br>
                🦠 <b>Diseases:</b> {diseases} <br>
                ⏳ <b>Life Span:</b> {lifespan} <br>
                📍 <b>Region:</b> {region}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠ No information found for this breed in CSV.")

# ================== BROWSE BREEDS TAB ==================
with tab2:
    st.markdown("### 📚 Browse Breeds")
    st.dataframe(breed_info[["Breed", "Food", "Climate", "Milk_Production (L/day)", "Region"]])

# ================== HISTORY TAB ==================
with tab3:
    st.markdown("### 🕘 History of Predictions")
    if st.session_state.history:
        for record in reversed(st.session_state.history[-10:]):  # Show last 10
            st.markdown(
                f"<div class='prediction-card'>🕘 {record['time']} — <b>{record['breed']}</b> ({record['confidence']})</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("📌 No predictions made yet.")
