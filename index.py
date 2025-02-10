import streamlit as st
from streamlit_drawable_canvas import st_canvas, _resize_img
import torch
from torch import load

from torchvision.transforms import v2
from nn import NeuralNetwork
from PIL import Image
import torchshow as tw

torch.classes.__path__ = []

CLASSES = [str(n) for n in range(10)]

DEVICE = "cpu"

def wide():
    st.set_page_config(layout="wide")

wide()

st.title("MNIST Number Draw")

canvas, predictions = st.columns(2)

model = NeuralNetwork().to(DEVICE)
model.load_state_dict(load("model.pth", weights_only=True, map_location=torch.device('cpu')))
model.eval()
pred = None

with canvas:
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )


    stroke_width = st.sidebar.slider("Stroke width: ", 40, 160, 60, 20)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 40, 160, 80, 20)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")

    realtime_update = st.sidebar.checkbox("Real-time Inferences", True)

    canvas_result = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=560,
        width=560,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

with predictions:
    st.header("Predictions", divider="gray")
    if canvas_result.image_data is not None:
        pil_img = _resize_img(Image.fromarray(canvas_result.image_data), 28, 28)
        img_tensor = (v2.Compose([v2.Grayscale(3), v2.PILToTensor(), v2.ToDtype(torch.float32, scale=False)])((pil_img))).to(DEVICE)
        # tw.save(img_tensor, ".")

        pred = model(img_tensor)
        st.subheader(f'Probably a {CLASSES[pred[0].argmax(0)]}')
        for classification, percentage in enumerate(pred[0]):
            st.slider(label=str(classification), value=float(percentage), disabled=True)