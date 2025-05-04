import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
from fpdf import FPDF
import os

# Page settings
st.set_page_config(page_title="🧠 Fetal Brain Abnormality Detection", page_icon="assets/favicon.png", layout="wide")

# Add animated background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        data = base64.b64encode(file.read()).decode()
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/gif;base64,{data}");
             background-size: cover;
             background-position: center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_local('assets/background.gif')  # Use .gif or adjust if you use .mp4 differently

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("model/vit_model_latest.pth", map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Class labels
class_names = ['Abnormality', 'Hydrocephalus', 'Porencephaly', 'Arachnoid-cyst', 'Normal']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Generate report
def generate_report(predictions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Fetal Brain Abnormality Detection Report", ln=True, align="C")
    pdf.ln(10)
    for i, (filename, prediction, confidence) in enumerate(predictions):
        pdf.cell(200, 10, txt=f"{i+1}. {filename}", ln=True)
        pdf.cell(200, 10, txt=f"   Prediction: {prediction} ({confidence:.2f}%)", ln=True)
        pdf.ln(5)
    report_path = "reports/Prediction_Report.pdf"
    os.makedirs("reports", exist_ok=True)
    pdf.output(report_path)
    return report_path

# ========== SIDEBAR ========== #
with st.sidebar:
    # College logo
    st.image("assets/college_logo.png", use_container_width=True)
    st.markdown("---")

    # Project Info
    st.markdown("## Project Info 🎓")
    st.markdown("**Project Title:** Fetal Brain Abnormality Detection")
    st.markdown("**Name:** Karan Pahwa")
    st.markdown("**Roll No:** 07320802821")
    st.markdown("**Guide:** Mrs. Prachi Kaushik")
    st.markdown("**College:** Bhagwan Parshuram Institute Of Technology [BPIT]")
    st.markdown("---")
    # st.markdown("Made with ❤️ using Streamlit")

# ========== MAIN PAGE ========== #
st.markdown("<h1 style='text-align: center; color: white;'>🧠 Fetal Brain Abnormality Detection 🚼</h1>", unsafe_allow_html=True)
st.markdown("## Upload fetal Ultrasound image(s) to detect abnormalities")

# Upload multiple images
uploaded_files = st.file_uploader("Upload Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

predictions = []

if uploaded_files:
    with st.spinner('Predicting... Please wait'):
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption=uploaded_file.name, width=350)  # <- 📷 SMALLER IMAGE

                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    predicted_class = class_names[predicted.item()]
                    predicted_confidence = confidence.item() * 100
                    
                    st.success(f"🧠 **Prediction:** {predicted_class} ({predicted_confidence:.2f}%)")

                    # Display a progress bar for confidence
                    st.progress(int(predicted_confidence))
                    
                    predictions.append((uploaded_file.name, predicted_class, predicted_confidence))
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

    if predictions:
        st.markdown("---")
        if st.button("📄 Download Report"):
            report_path = generate_report(predictions)
            with open(report_path, "rb") as f:
                st.download_button("Download Report PDF", f, file_name="FetalBrainDetection_Report.pdf")

# Footer
st.markdown("---")
# st.markdown("<center>Made with ❤️ for Final Year B.Tech Major Project</center>", unsafe_allow_html=True)
