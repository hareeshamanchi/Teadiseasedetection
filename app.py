import io
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from utils.yolox_loader import load_yolox_model
from utils.yolox_predict import predict_yolox, draw_yolox_predictions
from utils.yolox_gradcam import generate_yolox_gradcam


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tea Leaf Disease Detection",
    page_icon="🍃",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

YOLO11_PATH = "models/YOLO11_best.pt"


# ============================================================
# CLASS NAMES
# ============================================================

YOLO11_CLASSES = {
    0: "Anthracnose",
    1: "Brown Blight",
    2: "White Spot",
    3: "Algal Leaf Spot",
}

YOLOX_CLASSES = {
    0: "Algal Leaf Spot",
    1: "Anthracnose",
    2: "Brown Spot Disease",
    3: "White Spot Disease",
}


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Anthracnose": {
        "description":
            "A fungal leaf disease associated with dark, "
            "necrotic lesions on tea leaves.",
        "symptoms":
            "Dark brown to black lesions, necrotic patches, "
            "and progressive tissue damage.",
        "impact":
            "Severe infection can reduce healthy leaf area "
            "and affect leaf quality.",
    },

    "Brown Blight": {
        "description":
            "A tea-leaf disease category represented by "
            "brown blight-type lesions in the dataset.",
        "symptoms":
            "Brown or dark necrotic patches appearing on "
            "the leaf surface.",
        "impact":
            "Extensive blighting can reduce functional "
            "leaf area.",
    },

    "Brown Spot Disease": {
        "description":
            "A brown spot disease category represented "
            "in the YOLOX training classes.",
        "symptoms":
            "Localized brown or dark spots and lesions "
            "on the leaf.",
        "impact":
            "Multiple spots may reduce healthy leaf "
            "surface area.",
    },

    "White Spot": {
        "description":
            "A tea-leaf disease category represented by "
            "light or white spot-like lesions.",
        "symptoms":
            "Small pale or white areas that contrast "
            "with the surrounding green tissue.",
        "impact":
            "Multiple lesions can affect visible leaf "
            "quality and healthy surface area.",
    },

    "White Spot Disease": {
        "description":
            "The YOLOX class corresponding to the white "
            "spot disease category.",
        "symptoms":
            "Pale or white spot-like lesions on the leaf.",
        "impact":
            "Extensive spotting may affect healthy leaf "
            "surface area.",
    },

    "Algal Leaf Spot": {
        "description":
            "A tea-leaf disease category associated with "
            "localized algal leaf spotting.",
        "symptoms":
            "Localized circular or irregular spots that "
            "can become brownish or reddish.",
        "impact":
            "Extensive spotting can reduce healthy "
            "photosynthetic leaf area.",
    },
}


# ============================================================
# LOAD YOLO11m
# ============================================================

@st.cache_resource
def load_yolo11():

    model = YOLO(
        YOLO11_PATH
    )

    return model


# ============================================================
# LOAD YOLOX-M
# ============================================================

@st.cache_resource
def load_yolox():

    model, exp, device = load_yolox_model()

    return model, exp, device


# ============================================================
# HEADER
# ============================================================

st.title(
    "🍃 Tea Leaf Disease Detection"
)

st.markdown(
    """
### YOLO11m + YOLOX-M + Explainable AI

Upload a tea-leaf image to detect diseases using **two trained
object-detection models** and visualize the model decision with
Grad-CAM.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Detection Settings"
)

confidence_threshold = st.sidebar.slider(
    "YOLO11m Confidence",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

yolox_confidence = st.sidebar.slider(
    "YOLOX Confidence",
    min_value=0.01,
    max_value=0.95,
    value=0.25,
    step=0.01
)


st.sidebar.markdown("---")

st.sidebar.info(
    """
**Models**

YOLO11m:
40.5 MB trained checkpoint

YOLOX-M:
202 MB trained checkpoint

Both models are used for independent
prediction and explanation.
"""
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    yolo11_model = load_yolo11()

except Exception as e:

    st.error(
        f"YOLO11m model loading failed:\n\n{e}"
    )

    st.stop()


try:

    yolox_model, yolox_exp, yolox_device = load_yolox()

except Exception as e:

    st.error(
        f"YOLOX-M model loading failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

with st.sidebar:

    st.success(
        "YOLO11m loaded"
    )

    st.success(
        f"YOLOX-M loaded ({yolox_device})"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a tea-leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload a tea-leaf image to begin detection."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

image_bytes = uploaded_file.getvalue()

pil_image = Image.open(
    io.BytesIO(image_bytes)
).convert("RGB")

image_rgb = np.array(
    pil_image
)

image_bgr = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

st.subheader(
    "📷 Uploaded Image"
)

st.image(
    pil_image,
    caption=uploaded_file.name,
    width="stretch"
)


# ============================================================
# RUN YOLO11m
# ============================================================

with st.spinner(
    "Running YOLO11m detection..."
):

    start_time = time.perf_counter()

    yolo11_results = yolo11_model.predict(
        source=image_rgb,
        imgsz=640,
        conf=confidence_threshold,
        iou=0.45,
        verbose=False
    )

    yolo11_time = (
        time.perf_counter()
        -
        start_time
    )


yolo11_result = yolo11_results[0]

yolo11_boxes = yolo11_result.boxes


# ============================================================
# RUN YOLOX-M
# ============================================================

with st.spinner(
    "Running YOLOX-M detection..."
):

    start_time = time.perf_counter()

    yolox_result = predict_yolox(
        model=yolox_model,
        exp=yolox_exp,
        device=yolox_device,
        image=image_bgr,
        confidence_threshold=yolox_confidence
    )

    yolox_time = (
        time.perf_counter()
        -
        start_time
    )


yolox_detections = yolox_result[
    "detections"
]


# ============================================================
# YOLO11m RESULTS
# ============================================================

yolo11_rows = []

if yolo11_boxes is not None:

    for i in range(
        len(yolo11_boxes)
    ):

        class_id = int(
            yolo11_boxes.cls[i].item()
        )

        confidence = float(
            yolo11_boxes.conf[i].item()
        )

        bbox = [
            float(v)
            for v in yolo11_boxes.xyxy[i].tolist()
        ]

        disease = YOLO11_CLASSES.get(
            class_id,
            f"Class {class_id}"
        )

        yolo11_rows.append({

            "Detection": i + 1,

            "Disease": disease,

            "Confidence": confidence,

            "x1": bbox[0],

            "y1": bbox[1],

            "x2": bbox[2],

            "y2": bbox[3],

        })


yolo11_df = pd.DataFrame(
    yolo11_rows
)


# ============================================================
# YOLOX RESULTS
# ============================================================

yolox_rows = []

for i, detection in enumerate(
    yolox_detections
):

    bbox = detection[
        "bbox"
    ]

    yolox_rows.append({

        "Detection": i + 1,

        "Disease": detection[
            "class_name"
        ],

        "Confidence": detection[
            "confidence"
        ],

        "Objectness": detection[
            "objectness"
        ],

        "Class Confidence":
            detection[
                "class_confidence"
            ],

        "x1": bbox[0],

        "y1": bbox[1],

        "x2": bbox[2],

        "y2": bbox[3],

    })


yolox_df = pd.DataFrame(
    yolox_rows
)


# ============================================================
# TOP PREDICTION
# ============================================================

yolo11_top = None

if len(yolo11_df) > 0:

    yolo11_top = yolo11_df.iloc[
        yolo11_df["Confidence"].idxmax()
    ]


yolox_top = None

if len(yolox_df) > 0:

    yolox_top = yolox_df.iloc[
        yolox_df["Confidence"].idxmax()
    ]


# ============================================================
# TWO MODEL SUMMARY
# ============================================================

st.markdown("---")

st.header(
    "🔎 Detection Results"
)


col1, col2 = st.columns(2)


# ============================================================
# YOLO11m CARD
# ============================================================

with col1:

    st.subheader(
        "🟦 YOLO11m"
    )

    if yolo11_top is not None:

        st.success(
            f"Prediction: {yolo11_top['Disease']}"
        )

        st.metric(
            "Confidence",
            f"{yolo11_top['Confidence'] * 100:.2f}%"
        )

        st.metric(
            "Total Detections",
            len(yolo11_df)
        )

        st.metric(
            "Inference Time",
            f"{yolo11_time * 1000:.1f} ms"
        )

    else:

        st.warning(
            "YOLO11m found no detections."
        )


# ============================================================
# YOLOX CARD
# ============================================================

with col2:

    st.subheader(
        "🟥 YOLOX-M"
    )

    if yolox_top is not None:

        st.success(
            f"Prediction: {yolox_top['Disease']}"
        )

        st.metric(
            "Confidence",
            f"{yolox_top['Confidence'] * 100:.2f}%"
        )

        st.metric(
            "Total Detections",
            len(yolox_df)
        )

        st.metric(
            "Inference Time",
            f"{yolox_time * 1000:.1f} ms"
        )

    else:

        st.warning(
            "YOLOX-M found no detections."
        )


# ============================================================
# DETECTION IMAGES
# ============================================================

st.markdown("---")

st.header(
    "📦 Bounding Box Predictions"
)


col1, col2 = st.columns(2)


# YOLO11 image

with col1:

    st.subheader(
        "YOLO11m Prediction"
    )

    if yolo11_boxes is not None and len(
        yolo11_boxes
    ) > 0:

        yolo11_annotated = (
            yolo11_result.plot(
                conf=True,
                labels=True
            )
        )

        yolo11_annotated = cv2.cvtColor(
            yolo11_annotated,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            yolo11_annotated,
            width="stretch"
        )

    else:

        st.info(
            "No YOLO11m prediction image."
        )


# YOLOX image

with col2:

    st.subheader(
        "YOLOX-M Prediction"
    )

    if len(yolox_detections) > 0:

        yolox_annotated = (
            draw_yolox_predictions(
                image_bgr,
                yolox_detections
            )
        )

        yolox_annotated = cv2.cvtColor(
            yolox_annotated,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            yolox_annotated,
            width="stretch"
        )

    else:

        st.info(
            "No YOLOX-M prediction image."
        )


# ============================================================
# ALL DETECTIONS
# ============================================================

st.markdown("---")

st.header(
    "📋 All Predictions"
)


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "YOLO11m"
    )

    if len(yolo11_df) > 0:

        display_df = yolo11_df.copy()

        display_df[
            "Confidence"
        ] = (
            display_df[
                "Confidence"
            ] * 100
        ).round(2)

        display_df.rename(
            columns={
                "Confidence":
                    "Confidence (%)"
            },
            inplace=True
        )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

    else:

        st.write(
            "No detections."
        )


with col2:

    st.subheader(
        "YOLOX-M"
    )

    if len(yolox_df) > 0:

        display_df = yolox_df.copy()

        display_df[
            "Confidence"
        ] = (
            display_df[
                "Confidence"
            ] * 100
        ).round(2)

        display_df[
            "Objectness"
        ] = (
            display_df[
                "Objectness"
            ] * 100
        ).round(2)

        display_df[
            "Class Confidence"
        ] = (
            display_df[
                "Class Confidence"
            ] * 100
        ).round(2)

        display_df.rename(
            columns={
                "Confidence":
                    "Confidence (%)",

                "Objectness":
                    "Objectness (%)",

                "Class Confidence":
                    "Class Confidence (%)",
            },
            inplace=True
        )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

    else:

        st.write(
            "No detections."
        )


# ============================================================
# DISEASE DETAILS
# ============================================================

st.markdown("---")

st.header(
    "🦠 Disease Details"
)


disease_for_info = None

if yolo11_top is not None:

    disease_for_info = (
        yolo11_top["Disease"]
    )

elif yolox_top is not None:

    disease_for_info = (
        yolox_top["Disease"]
    )


if disease_for_info is not None:

    info = DISEASE_INFO.get(
        disease_for_info,
        {}
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "### Description"
        )

        st.write(
            info.get(
                "description",
                "Disease information is not available."
            )
        )

    with col2:

        st.markdown(
            "### Visual Symptoms"
        )

        st.write(
            info.get(
                "symptoms",
                "See the highlighted detection region."
            )
        )

    with col3:

        st.markdown(
            "### Potential Impact"
        )

        st.write(
            info.get(
                "impact",
                "Interpret this as a model prediction."
            )
        )


# ============================================================
# YOLOX GRAD-CAM
# ============================================================

st.markdown("---")

st.header(
    "🔥 YOLOX-M Grad-CAM"
)

st.caption(
    "The heatmap highlights image regions contributing "
    "to the selected YOLOX-M detection."
)


if yolox_top is not None:

    # Find corresponding detection
    top_index = int(
        yolox_df[
            "Confidence"
        ].idxmax()
    )

    selected_yolox_detection = (
        yolox_detections[
            top_index
        ]
    )

    with st.spinner(
        "Generating YOLOX-M Grad-CAM..."
    ):

        try:

            cam_result = (
                generate_yolox_gradcam(
                    model=yolox_model,
                    exp=yolox_exp,
                    device=yolox_device,
                    image=image_bgr,
                    detection=selected_yolox_detection
                )
            )

            heatmap_bgr = cam_result[
                "heatmap"
            ]

            overlay_bgr = cam_result[
                "overlay"
            ]

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "YOLOX Heatmap"
                )

                st.image(
                    cv2.cvtColor(
                        heatmap_bgr,
                        cv2.COLOR_BGR2RGB
                    ),
                    width="stretch"
                )

            with col2:

                st.subheader(
                    "YOLOX Heatmap + Prediction"
                )

                st.image(
                    cv2.cvtColor(
                        overlay_bgr,
                        cv2.COLOR_BGR2RGB
                    ),
                    width="stretch"
                )


            # ------------------------------------------------
            # Download heatmap
            # ------------------------------------------------

            ok, encoded_heatmap = cv2.imencode(
                ".jpg",
                heatmap_bgr,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

            if ok:

                st.download_button(
                    "⬇️ Download YOLOX Grad-CAM",
                    encoded_heatmap.tobytes(),
                    "YOLOX_GradCAM.jpg",
                    "image/jpeg"
                )


            # ------------------------------------------------
            # Download overlay
            # ------------------------------------------------

            ok, encoded_overlay = cv2.imencode(
                ".jpg",
                overlay_bgr,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

            if ok:

                st.download_button(
                    "⬇️ Download YOLOX Heatmap + Box",
                    encoded_overlay.tobytes(),
                    "YOLOX_GradCAM_Prediction.jpg",
                    "image/jpeg"
                )


            st.write(
                f"**CAM target:** "
                f"{cam_result['class_name']}"
            )

            st.write(
                f"**Confidence:** "
                f"{cam_result['confidence'] * 100:.2f}%"
            )

            st.write(
                f"**Matched raw prediction locations:** "
                f"{cam_result['matched_anchor_count']}"
            )


        except Exception as e:

            st.error(
                f"YOLOX Grad-CAM failed: {e}"
            )

else:

    st.info(
        "YOLOX-M did not produce a detection, "
        "so a detection-specific heatmap cannot be generated."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("---")

st.header(
    "⚖️ YOLO11m vs YOLOX-M"
)


comparison_rows = []


if yolo11_top is not None:

    comparison_rows.append({

        "Model":
            "YOLO11m",

        "Predicted Disease":
            yolo11_top["Disease"],

        "Confidence":
            yolo11_top["Confidence"],

        "Detections":
            len(yolo11_df),

        "Inference (ms)":
            yolo11_time * 1000,

    })


if yolox_top is not None:

    comparison_rows.append({

        "Model":
            "YOLOX-M",

        "Predicted Disease":
            yolox_top["Disease"],

        "Confidence":
            yolox_top["Confidence"],

        "Detections":
            len(yolox_df),

        "Inference (ms)":
            yolox_time * 1000,

    })


if comparison_rows:

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df[
        "Confidence"
    ] = (
        comparison_df[
            "Confidence"
        ] * 100
    ).round(2)

    comparison_df[
        "Inference (ms)"
    ] = comparison_df[
        "Inference (ms)"
    ].round(2)

    comparison_df.rename(
        columns={
            "Confidence":
                "Confidence (%)"
        },
        inplace=True
    )

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# DOWNLOAD PREDICTION CSV
# ============================================================

st.markdown("---")

st.header(
    "📥 Download Results"
)


all_download_rows = []


for row in yolo11_rows:

    row_copy = row.copy()

    row_copy["Model"] = "YOLO11m"

    all_download_rows.append(
        row_copy
    )


for row in yolox_rows:

    row_copy = row.copy()

    row_copy["Model"] = "YOLOX-M"

    all_download_rows.append(
        row_copy
    )


if all_download_rows:

    download_df = pd.DataFrame(
        all_download_rows
    )

    csv_data = download_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ Download All Predictions CSV",
        csv_data,
        "tea_leaf_predictions.csv",
        "text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Tea Leaf Disease Detection | "
    "YOLO11m + YOLOX-M | "
    "Explainable AI"
)