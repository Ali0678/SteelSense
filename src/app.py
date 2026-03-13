import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from model import SteelCNN

st.set_page_config(page_title="SteelSense AI", page_icon="🏭")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'models', 'best_model.pth')
    model = SteelCNN(num_classes=6).to(DEVICE)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    else:
        st.error(f"❌ Model not found at {model_path}. Please train it first!")
        return None

model = load_model()
st.title("🏭 SteelSense: Defect Detection")
st.write("Upload an image of a steel surface to detect defects like rust, scratches, or cracks.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Steel Image', use_container_width=True)
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    if st.button("Analyze Surface"):
        if model:
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                top_prob, top_idx = torch.max(probabilities, 1)
                prediction = CLASSES[top_idx.item()]
                confidence = top_prob.item() * 100

            st.success(f"**Prediction: {prediction.upper()}**")
            st.progress(int(confidence))
            st.caption(f"Confidence: {confidence:.2f}%")
            with st.expander("See full probability breakdown"):
                for i, class_name in enumerate(CLASSES):
                    prob = probabilities[0][i].item() * 100
                    st.write(f"- {class_name}: {prob:.2f}%")