import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import io

MODEL_DIR = "."
MODEL_FILES = {
    "VGG16": os.path.join(MODEL_DIR, "best_vgg16.pth"),
    "ResNet18": os.path.join(MODEL_DIR, "best_resnet18.pth"),
    "DenseNet121": os.path.join(MODEL_DIR, "best_densenet121.pth"),
}
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["Fake Face", "Real Face"] 
MODEL_OUTPUT_UNITS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_pytorch_model(model_name, model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please ensure the .pth model files are in the correct directory.")
        return None
    try:
        if model_name == "VGG16":
            model = models.vgg16(weights=None)
            model.classifier[3] = nn.Linear(in_features=4096, out_features=1024)
            model.classifier[6] = nn.Linear(in_features=1024, out_features=MODEL_OUTPUT_UNITS)
        elif model_name == "ResNet18":
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, MODEL_OUTPUT_UNITS)
            )
        elif model_name == "DenseNet121":
            model = models.densenet121(weights=None)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, MODEL_OUTPUT_UNITS)
            )
        else:
            st.error(f"Unknown model name: {model_name}")
            return None

        state_dict = torch.load(model_path, map_location=DEVICE)
        if next(iter(state_dict)).startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

preprocess_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

def preprocess_pil_image(image_pil):
    try:
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        img_tensor = preprocess_transform(image_pil)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(DEVICE)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict(model, processed_image_tensor):
    try:
        with torch.no_grad():
            output_logit = model(processed_image_tensor)
            probability_real = torch.sigmoid(output_logit.squeeze()).item()

            predicted_class_name = ""
            confidence_score = 0.0

            if probability_real > 0.5:
                predicted_class_name = CLASS_NAMES[1] # Real Face
                confidence_score = probability_real
            else:
                predicted_class_name = CLASS_NAMES[0] # Fake Face
                confidence_score = 1.0 - probability_real

            return predicted_class_name, confidence_score

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def main():
    st.set_page_config(page_title="Real vs. Fake Face Detection", layout="wide")

    st.title("🎭 Real vs. Fake Face Detection")
    st.markdown("""
        Upload a facial image to predict whether it's a real human face or an AI-generated (fake) one.
        This application uses pre-trained PyTorch models.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Configuration")

        selected_model_name = st.selectbox(
            "Choose a Model:",
            list(MODEL_FILES.keys())
        )

        model_path = MODEL_FILES[selected_model_name]

        uploaded_file = st.file_uploader("Upload Face Image", type=["png", "jpg", "jpeg"])

        if uploaded_file:
            st.markdown("---")
            st.subheader("🖼️ Uploaded Image")
            try:
                image_bytes = uploaded_file.getvalue()
                image_pil = Image.open(io.BytesIO(image_bytes))
                st.image(image_pil, caption="Uploaded Face", use_column_width=True)
            except Exception as e:
                st.error(f"Error opening image: {e}")
                uploaded_file = None

    with col2:
        st.subheader("💡 Prediction Result")
        if uploaded_file:
            model = load_pytorch_model(selected_model_name, model_path)

            if model:
                st.info(f"Processing with **{selected_model_name}** model on **{DEVICE}**...")

                processed_image = preprocess_pil_image(image_pil)

                if processed_image is not None:
                    predicted_class, confidence = predict(model, processed_image)

                    if predicted_class and confidence is not None:
                        if predicted_class == CLASS_NAMES[0]:
                            st.error(f"**Prediction:** {predicted_class}")
                        else: # Real Face
                            st.success(f"**Prediction:** {predicted_class}")
                        st.info(f"**Confidence in prediction:** {confidence*100:.2f}%")

                        st.markdown("---")
                        st.markdown(f"**Model Interpretation Note:**")
                        st.markdown(f"- Class index 0: {CLASS_NAMES[0]}")
                        st.markdown(f"- Class index 1: {CLASS_NAMES[1]}")
                        st.markdown(f"- Model output (after sigmoid) is $P(\\text{{Real}})$. If $P(\\text{{Real}}) > 0.5$, classified as '{CLASS_NAMES[1]}', else '{CLASS_NAMES[0]}'.")
                    else:
                        st.warning("Could not get a prediction.")
                else:
                    st.warning("Image could not be preprocessed.")
            else:
                st.warning(f"Could not load the {selected_model_name} model. Please check file path, integrity, and model definition.")
        else:
            st.info("Upload an image to see the prediction.")

    st.markdown("---")
    st.markdown("Distinguishing Real and AI-Generated Human Faces")
    st.markdown(f"Dataset source: [140k Real and Fake Faces on Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)")

if __name__ == "__main__":
    main()