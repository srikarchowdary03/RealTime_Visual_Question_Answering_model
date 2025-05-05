import streamlit as st
from PIL import Image
from predict_vqa import predict_answer

st.set_page_config(page_title="VQA App", layout="centered")
st.title("📷 Visual Question Answering with ViLT")
st.write("Ask a question about what you see — your webcam will help answer it!")

# Step 1: Input Question
question = st.text_input("❓ Ask a question about the image:")

# Step 2: Capture image using webcam
image_file = st.camera_input("📸 Take a picture")

# Step 3: Predict
if question and image_file:
    image = Image.open(image_file).convert("RGB")
    with st.spinner("Thinking..."):
        answer = predict_answer(image, question)
        st.success(f"💬 Answer: **{answer}**")
