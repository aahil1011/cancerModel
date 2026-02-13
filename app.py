# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# from dotenv import load_dotenv
# from google import genai

# # --------------------------------------------------
# # Load Environment Variables
# # --------------------------------------------------
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     st.error("Gemini API key not found. Add it in .env file.")
#     st.stop()

# client = genai.Client(api_key=GEMINI_API_KEY)

# # --------------------------------------------------
# # Page Config
# # --------------------------------------------------
# st.set_page_config(
#     page_title="MediPal - Breast Cancer AI",
#     page_icon="🧬",
#     layout="centered"
# )

# st.title("🧬 Breast Cancer Classification with MediPal")
# st.markdown("Upload a pathology slide for AI-powered analysis")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --------------------------------------------------
# # Load Model
# # --------------------------------------------------
# @st.cache_resource
# def load_model():
#     model = models.resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, 3)
#     model.load_state_dict(torch.load("breast_model_resnet18.pth", map_location=device))
#     model.to(device)
#     model.eval()
#     return model

# model = load_model()
# classes = ['Benign', 'Malignant', 'Normal']

# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],
#                          [0.229,0.224,0.225])
# ])

# # --------------------------------------------------
# # Grad-CAM
# # --------------------------------------------------
# def generate_gradcam(model, image_tensor, target_class):

#     gradients = []
#     activations = []

#     def backward_hook(module, grad_input, grad_output):
#         gradients.append(grad_output[0])

#     def forward_hook(module, input, output):
#         activations.append(output)

#     target_layer = model.layer4

#     fh = target_layer.register_forward_hook(forward_hook)
#     bh = target_layer.register_full_backward_hook(backward_hook)

#     output = model(image_tensor)
#     model.zero_grad()
#     loss = output[0, target_class]
#     loss.backward()

#     grads = gradients[0]
#     acts = activations[0]

#     pooled_grads = torch.mean(grads, dim=[0,2,3])

#     for i in range(acts.shape[1]):
#         acts[:, i, :, :] *= pooled_grads[i]

#     heatmap = torch.mean(acts, dim=1).squeeze()
#     heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)

#     if np.max(heatmap) != 0:
#         heatmap /= np.max(heatmap)

#     fh.remove()
#     bh.remove()

#     return heatmap

# # --------------------------------------------------
# # Gemini Analysis
# # --------------------------------------------------
# def generate_analysis(predicted_class, confidence):

#     prompt = f"""
# You are a medical AI assistant.

# Model Prediction: {predicted_class}
# Confidence: {confidence:.2f}%

# Provide a 100-word educational pathology-style explanation.
# Do NOT provide clinical diagnosis.
# Explain possible histological patterns.
# """

#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=prompt
#     )

#     return response.text


# # --------------------------------------------------
# # Upload Section
# # --------------------------------------------------
# uploaded_file = st.file_uploader("Upload Slide Image", type=["jpg","png","tif"])

# if uploaded_file:

#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Original Image", width="stretch")

#     img_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(img_tensor)
#         probs = torch.softmax(output, dim=1)[0]

#     predicted_index = torch.argmax(probs).item()
#     predicted_class = classes[predicted_index]
#     confidence = probs[predicted_index].item() * 100

#     # ---------------- Prediction ----------------
#     st.subheader("🔎 Prediction")
#     st.success(f"{predicted_class} ({confidence:.2f}%)")

#     # ---------------- Bar Chart ----------------
#     st.subheader("📊 Prediction Probability Distribution")

#     prob_values = probs.cpu().numpy() * 100
#     class_names = ['Benign', 'Malignant', 'Normal']

#     fig, ax = plt.subplots()
#     bars = ax.bar(class_names, prob_values)

#     ax.set_ylabel("Probability (%)")
#     ax.set_ylim(0, 100)
#     ax.set_title("Model Confidence per Class")

#     for bar in bars:
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 1,
#                 f'{height:.1f}%', ha='center')

#     st.pyplot(fig)

#     # ---------------- Grad-CAM ----------------
#     heatmap = generate_gradcam(model, img_tensor, predicted_index)
#     heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
#     heatmap_uint8 = np.uint8(255 * heatmap)
#     colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

#     overlay = cv2.addWeighted(np.array(image), 0.6, colored_heatmap, 0.4, 0)

#     st.subheader("🔥 Grad-CAM Heatmap")
#     st.image(overlay, width="stretch")

#     # ---------------- Gemini Analysis ----------------
#     st.subheader("🩺 Gemini AI Analysis")

#     with st.spinner("Generating analysis..."):
#         try:
#             analysis_text = generate_analysis(predicted_class, confidence)
#             st.write(analysis_text)
#         except Exception as e:
#             st.error("Gemini API Error")
#             st.write(e)

# # --------------------------------------------------
# # MediPal Chatbot
# # --------------------------------------------------
# st.divider()
# st.subheader("🤖 MediPal - Ask Follow-up Questions")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_input = st.text_input("Ask MediPal about pathology or cancer concepts")

# if st.button("Send"):

#     if user_input.strip() == "":
#         st.warning("Please enter a question before sending.")
#     else:
#         try:
#             chat_prompt = f"""
# You are MediPal, an educational medical AI assistant.
# Answer clearly and safely.
# Do not provide diagnosis.

# User Question: {user_input}
# """

#             response = client.models.generate_content(
#                 model="gemini-2.5-flash",
#                 contents=chat_prompt
#             )

#             st.session_state.chat_history.append(("You", user_input))
#             st.session_state.chat_history.append(("MediPal", response.text))

#         except Exception as e:
#             st.error("Gemini API Error")
#             st.write(e)

# for role, message in st.session_state.chat_history:
#     st.write(f"**{role}:** {message}")

# st.warning("⚠️ MediPal is for educational purposes only. Not for clinical use.")


#######################new one##########################
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from dotenv import load_dotenv
from google import genai

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MediPal", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key missing.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

st.title("🧬 MediPal")
st.caption("Breast Cancer Histopathology AI System")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("breast_model_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
classes = ['Benign', 'Malignant', 'Normal']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- GRAD CAM ----------------
def generate_gradcam(model, image_tensor, target_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    grads = gradients[0]
    acts = activations[0]
    pooled_grads = torch.mean(grads, dim=[0,2,3])

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    fh.remove()
    bh.remove()
    return heatmap

# ---------------- GEMINI ----------------
def generate_analysis(predicted_class, confidence):
    prompt = f"""
You are a medical AI assistant.

Prediction: {predicted_class}
Confidence: {confidence:.2f}%

Provide a concise educational explanation (100 words).
Do not provide diagnosis.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ---------------- UPLOAD SECTION ----------------
uploaded_file = st.file_uploader("Upload pathology slide", type=["jpg","png","tif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    predicted_index = torch.argmax(probs).item()
    predicted_class = classes[predicted_index]
    confidence = probs[predicted_index].item() * 100

    # Grad-CAM
    heatmap = generate_gradcam(model, img_tensor, predicted_index)
    heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, colored_heatmap, 0.4, 0)

    # Highlight regions
    mask = np.uint8(heatmap_resized > 0.6) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circled_image = img_np.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(circled_image, (int(x), int(y)), int(radius), (255,0,0), 3)

    st.subheader("Visual Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_np, width=280, caption="Original")
    with col2:
        st.image(overlay, width=280, caption="Grad-CAM")
    with col3:
        st.image(circled_image, width=280, caption="High Activation")

    st.divider()

    colA, colB = st.columns([1,1])

    with colA:
        st.metric("Diagnosis", predicted_class)
        st.metric("Confidence", f"{confidence:.2f}%")

    with colB:
        fig, ax = plt.subplots(figsize=(4,3))
        prob_values = probs.cpu().numpy() * 100
        bars = ax.bar(classes, prob_values, color=["green","red","blue"])
        ax.set_ylim(0,100)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, height+1,
                    f'{height:.1f}%', ha='center')
        st.pyplot(fig)

    st.subheader("🩺 AI Explanation")
    with st.spinner("Generating..."):
        explanation = generate_analysis(predicted_class, confidence)
        st.write(explanation)

# ---------------- EVALUATION DASHBOARD ----------------
@st.cache_data
def evaluate_model():
    test_dataset = datasets.ImageFolder("dataset/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)

st.divider()
st.subheader("📈 Model Evaluation Dashboard")

if st.button("Show Evaluation Metrics"):

    probs, labels = evaluate_model()
    preds = np.argmax(probs, axis=1)

    cm = confusion_matrix(labels, preds)
    labels_bin = label_binarize(labels, classes=[0,1,2])

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        fig_cm, ax_cm = plt.subplots(figsize=(4,4))
        im = ax_cm.imshow(cm, cmap="Reds")
        ax_cm.set_xticks(range(3))
        ax_cm.set_yticks(range(3))
        ax_cm.set_xticklabels(classes)
        ax_cm.set_yticklabels(classes)
        for i in range(3):
            for j in range(3):
                ax_cm.text(j, i, cm[i,j], ha="center", va="center")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

    # ROC Curve
    with col2:
        fig_roc, ax_roc = plt.subplots(figsize=(4,4))
        for i, color in zip(range(3), ["green","red","blue"]):
            fpr, tpr, _ = roc_curve(labels_bin[:,i], probs[:,i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr,
                        label=f"{classes[i]} AUC={roc_auc:.2f}",
                        color=color)
        ax_roc.plot([0,1],[0,1],'k--')
        ax_roc.legend()
        ax_roc.set_title("ROC Curve")
        st.pyplot(fig_roc)

# ---------------- CHATBOT ----------------
st.divider()
st.subheader("🤖 MediPal Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask MediPal a question")

if st.button("Send"):
    if user_input.strip():
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_input
        )
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("MediPal", response.text))

for role, msg in st.session_state.chat_history:
    st.write(f"**{role}:** {msg}")

st.caption("Educational use only. Not medical advice.")
