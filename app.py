import streamlit as st
import torch
import numpy as np
from model import InstaFakeDetector
import joblib
import os

st.set_page_config(page_title="Fake Instagram Profile Detector", layout="centered")
st.title("🕵️ Fake Instagram Profile Detector")

st.markdown("### Enter Instagram profile details below:")

# Helper to convert "Yes/No" to 1/0
def binary_input(label):
    return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

# User Inputs
profile_pic = binary_input("🖼️ Profile Picture Present?")
username_digit_ratio = st.number_input("🔢 Digits per length of username", min_value=0.0, max_value=1.0, step=0.01)
fullname_words = st.number_input("🧑 Number of words in full name", min_value=0)
fullname_digit_ratio = st.number_input("🔢 Digits per length of full name", min_value=0.0, max_value=1.0, step=0.01)
same_name_username = binary_input("🔁 Is Name same as Username?")
description_length = st.number_input("📝 Bio description length", min_value=0)
external_url = binary_input("🔗 External URL in bio?")
private = binary_input("🔒 Is the profile private?")
num_posts = st.number_input("📸 Number of posts", min_value=0)
followers = st.number_input("👥 Number of followers", min_value=0)
follows = st.number_input("🔄 Number of follows", min_value=0)

# Create feature vector
features = np.array([[
    profile_pic,
    username_digit_ratio,
    fullname_words,
    fullname_digit_ratio,
    same_name_username,
    description_length,
    external_url,
    private,
    num_posts,
    followers,
    follows
]])

if st.button("🚀 Predict"):
    if os.path.exists("trained_model.pth") and os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
        features_scaled = scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        model = InstaFakeDetector(input_dim=features.shape[1])
        model.load_state_dict(torch.load("trained_model.pth"))
        model.eval()

        with torch.no_grad():
            prediction = model(features_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()

        if predicted_class == 1:
            st.error("⚠️ This profile is likely **FAKE**.")
        else:
            st.success("✅ This profile is likely **GENUINE**.")
    else:
        st.error("❌ Trained model or scaler file not found. Please run `train.py` first.")
