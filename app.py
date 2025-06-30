import streamlit as st
import torch
from torchvision import models


@st.cache_resource
def load_model():
    return models.resnet18(pretrained=True)


st.title("🚀 PyTorch + Streamlit Demo")
model = load_model()

# Input interaktif
input_num = st.slider("Pilih angka:", 1, 10, 5)
if st.button("Generate Random Tensor"):
    random_tensor = torch.rand(input_num, input_num)
    st.write("Hasil Tensor:", random_tensor)

# Tampilkan arsitektur model
st.subheader("Arsitektur ResNet18")
st.code(str(model))
