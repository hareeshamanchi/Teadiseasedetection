import cv2
import numpy as np
import torch

from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess


# ============================================================
# YOLOX CLASS NAMES
# ============================================================

YOLOX_CLASS_NAMES = [
    "Algal Leaf Spot",
    "Anthracnose",
    "Brown Spot Disease",
    "White Spot Disease",
]


# ============================================================
# PREPROCESSOR
# ============================================================

PREPROCESS = ValTransform(
    legacy=False
)


# ============================================================
# YOLOX PREDICTION
# ============================================================

def predict_yolox(
    model,
    exp,
    device,
    image,
    confidence_threshold=None,
):
    """
    Run YOLOX-M inference on one image.

    Parameters
    ----------
    model:
        Loaded YOLOX model.

    exp:
        YOLOX experiment configuration.

    device:
        "cuda" or "cpu".

    image:
        OpenCV BGR image.

    confidence_threshold:
        Optional confidence threshold.
        Defaults to exp.test_conf.

    Returns
    -------
    detections:
        List of dictionaries containing:
        class_id
        class_name
        confidence
        objectness
        class_confidence
        bbox
    """

    if confidence_threshold is None:
        confidence_threshold = exp.test_conf


    # ========================================================
    # ORIGINAL IMAGE SIZE
    # ========================================================

    original_height, original_width = image.shape[:2]


    # ========================================================
    # YOLOX PREPROCESSING
    # ========================================================

    processed_image, _ = PREPROCESS(
        image,
        None,
        exp.test_size
    )


    # IMPORTANT:
    # Do NOT use the second return value from ValTransform
    # as the resize ratio.
    #
    # The previous Kaggle environment returned an array there.
    # We therefore calculate the ratio explicitly.

    ratio = min(
        float(exp.test_size[0]) / float(original_height),
        float(exp.test_size[1]) / float(original_width)
    )


    # ========================================================
    # CONVERT TO TENSOR
    # ========================================================

    tensor = torch.from_numpy(
        processed_image
    ).unsqueeze(0).float().to(device)


    # ========================================================
    # MODEL FORWARD
    # ========================================================

    with torch.no_grad():

        outputs = model(
            tensor
        )

        outputs = postprocess(
            outputs,
            exp.num_classes,
            confidence_threshold,
            exp.nmsthre,
            class_agnostic=True
        )


    # ========================================================
    # NO DETECTIONS
    # ========================================================

    if outputs is None or outputs[0] is None:

        return {
            "detections": [],
            "ratio": ratio,
            "image_width": original_width,
            "image_height": original_height,
        }


    output = outputs[0]


    # ========================================================
    # CONVERT DETECTIONS TO ORIGINAL IMAGE COORDINATES
    # ========================================================

    output = output.cpu()

    # YOLOX postprocess format:
    #
    # x1, y1, x2, y2,
    # objectness,
    # class_confidence,
    # class_id

    output[:, :4] /= ratio


    # Keep boxes inside original image
    output[:, 0] = output[:, 0].clamp(
        0,
        original_width
    )

    output[:, 1] = output[:, 1].clamp(
        0,
        original_height
    )

    output[:, 2] = output[:, 2].clamp(
        0,
        original_width
    )

    output[:, 3] = output[:, 3].clamp(
        0,
        original_height
    )


    # ========================================================
    # BUILD DETECTION LIST
    # ========================================================

    detections = []

    for row in output:

        x1 = float(row[0])
        y1 = float(row[1])
        x2 = float(row[2])
        y2 = float(row[3])

        objectness = float(row[4])

        class_confidence = float(row[5])

        class_id = int(row[6])

        # Final YOLOX confidence
        confidence = (
            objectness *
            class_confidence
        )


        if class_id >= len(
            YOLOX_CLASS_NAMES
        ):
            class_name = f"Class {class_id}"
        else:
            class_name = YOLOX_CLASS_NAMES[
                class_id
            ]


        detections.append({

            "class_id": class_id,

            "class_name": class_name,

            "confidence": confidence,

            "objectness": objectness,

            "class_confidence": class_confidence,

            "bbox": [
                x1,
                y1,
                x2,
                y2
            ],

        })


    # ========================================================
    # SORT BY CONFIDENCE
    # ========================================================

    detections.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )


    return {
        "detections": detections,

        "ratio": ratio,

        "image_width": original_width,

        "image_height": original_height,
    }


# ============================================================
# DRAW YOLOX PREDICTIONS
# ============================================================

def draw_yolox_predictions(
    image,
    detections,
):
    """
    Draw YOLOX bounding boxes and labels.
    """

    output = image.copy()


    for detection in detections:

        x1, y1, x2, y2 = [
            int(v)
            for v in detection["bbox"]
        ]

        class_name = detection[
            "class_name"
        ]

        confidence = detection[
            "confidence"
        ]


        # YOLOX prediction box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )


        label = (
            f"{class_name} "
            f"{confidence * 100:.2f}%"
        )


        # Text background
        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )
        )


        text_y = max(
            text_height + 5,
            y1
        )


        cv2.rectangle(
            output,
            (
                x1,
                text_y - text_height - baseline - 5
            ),
            (
                x1 + text_width + 5,
                text_y
            ),
            (0, 0, 255),
            -1
        )


        cv2.putText(
            output,
            label,
            (
                x1 + 2,
                text_y - 4
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    return output