import cv2
import numpy as np
import torch


# ============================================================
# YOLOX-M GRAD-CAM CONFIGURATION
# ============================================================

YOLOX_TARGET_LAYER = "dark3"

ANCHOR_IOU_THRESHOLD = 0.30
TOP_K_FALLBACK = 15


# ============================================================
# FIND YOLOX DARK3 LAYER
# ============================================================

def get_dark3_layer(model):
    """
    Get the YOLOX-M dark3 feature layer.

    This is the same YOLOX backbone layer used in the
    previously verified Kaggle Grad-CAM implementation.
    """

    return model.backbone.backbone.dark3


# ============================================================
# IOU CALCULATION
# ============================================================

def calculate_iou(box1, boxes2):
    """
    Calculate IoU between one box and multiple boxes.

    box format:
        [x1, y1, x2, y2]
    """

    box1 = np.asarray(
        box1,
        dtype=np.float32
    )

    boxes2 = np.asarray(
        boxes2,
        dtype=np.float32
    )

    x1 = np.maximum(
        box1[0],
        boxes2[:, 0]
    )

    y1 = np.maximum(
        box1[1],
        boxes2[:, 1]
    )

    x2 = np.minimum(
        box1[2],
        boxes2[:, 2]
    )

    y2 = np.minimum(
        box1[3],
        boxes2[:, 3]
    )

    intersection = np.maximum(
        0,
        x2 - x1
    ) * np.maximum(
        0,
        y2 - y1
    )

    area1 = (
        max(0, box1[2] - box1[0])
        *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        np.maximum(
            0,
            boxes2[:, 2] - boxes2[:, 0]
        )
        *
        np.maximum(
            0,
            boxes2[:, 3] - boxes2[:, 1]
        )
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    return intersection / (
        union + 1e-8
    )


# ============================================================
# YOLOX RAW OUTPUT BOX DECODING
# ============================================================

def get_raw_boxes(raw_output):
    """
    Extract raw YOLOX box coordinates.

    YOLOX raw output format:

        cx, cy, w, h, obj, cls0, cls1, cls2, cls3

    """

    if isinstance(
        raw_output,
        (tuple, list)
    ):
        raw_output = raw_output[0]

    # Expected shape:
    #
    # [batch, anchors, 5 + num_classes]

    if raw_output.ndim != 3:
        raise ValueError(
            f"Unexpected YOLOX output shape: "
            f"{raw_output.shape}"
        )

    # Convert to:
    #
    # [anchors, channels]

    predictions = raw_output[0]

    boxes = predictions[:, :4]

    return predictions, boxes


# ============================================================
# YOLOX GRAD-CAM
# ============================================================

def generate_yolox_gradcam(
    model,
    exp,
    device,
    image,
    detection,
):
    """
    Generate a true gradient-based object-specific
    Grad-CAM heatmap for a YOLOX detection.

    Parameters
    ----------
    model:
        Loaded YOLOX-M model.

    exp:
        YOLOX experiment.

    device:
        cpu or cuda.

    image:
        Original BGR image.

    detection:
        Detection dictionary returned by predict_yolox().
    """

    # ========================================================
    # IMAGE SIZE
    # ========================================================

    original_height, original_width = image.shape[:2]


    # ========================================================
    # PREPROCESS
    # ========================================================

    from yolox.data.data_augment import ValTransform

    preprocess = ValTransform(
        legacy=False
    )

    processed_image, _ = preprocess(
        image,
        None,
        exp.test_size
    )


    # ========================================================
    # EXPLICIT RESIZE RATIO
    # ========================================================

    ratio = min(
        float(exp.test_size[0]) / float(original_height),
        float(exp.test_size[1]) / float(original_width)
    )


    # ========================================================
    # TENSOR
    # ========================================================

    tensor = torch.from_numpy(
        processed_image
    ).unsqueeze(0).float().to(device)

    tensor.requires_grad_(True)


    # ========================================================
    # TARGET DETECTION
    # ========================================================

    target_box_original = np.asarray(
        detection["bbox"],
        dtype=np.float32
    )

    target_class = int(
        detection["class_id"]
    )


    # Convert original-image coordinates
    # to YOLOX 800x800 coordinates.

    target_box_resized = (
        target_box_original
        *
        ratio
    )


    # ========================================================
    # CAPTURE ACTIVATION + GRADIENT
    # ========================================================

    target_layer = get_dark3_layer(
        model
    )

    activation = []
    gradient = []


    def forward_hook(
        module,
        input_data,
        output
    ):
        activation.append(
            output
        )


    def backward_hook(
        module,
        grad_input,
        grad_output
    ):
        gradient.append(
            grad_output[0]
        )


    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )


    try:

        # ====================================================
        # FORWARD
        # ====================================================

        model.zero_grad(
            set_to_none=True
        )

        raw_output = model(
            tensor
        )

        if isinstance(
            raw_output,
            (tuple, list)
        ):
            raw_output = raw_output[0]


        # ====================================================
        # RAW OUTPUT
        # ====================================================

        predictions, raw_boxes = get_raw_boxes(
            raw_output
        )


        # ====================================================
        # RAW BOXES
        # ====================================================

        raw_boxes_np = (
            raw_boxes.detach()
            .cpu()
            .numpy()
        )


        # YOLOX raw boxes are:
        #
        # cx, cy, w, h
        #
        # Convert to x1,y1,x2,y2

        raw_xyxy = np.zeros_like(
            raw_boxes_np
        )

        raw_xyxy[:, 0] = (
            raw_boxes_np[:, 0]
            -
            raw_boxes_np[:, 2] / 2
        )

        raw_xyxy[:, 1] = (
            raw_boxes_np[:, 1]
            -
            raw_boxes_np[:, 3] / 2
        )

        raw_xyxy[:, 2] = (
            raw_boxes_np[:, 0]
            +
            raw_boxes_np[:, 2] / 2
        )

        raw_xyxy[:, 3] = (
            raw_boxes_np[:, 1]
            +
            raw_boxes_np[:, 3] / 2
        )


        # ====================================================
        # MATCH TARGET DETECTION TO RAW PREDICTIONS
        # ====================================================

        ious = calculate_iou(
            target_box_resized,
            raw_xyxy
        )


        matched_indices = np.where(
            ious >= ANCHOR_IOU_THRESHOLD
        )[0]


        # ====================================================
        # FALLBACK
        # ====================================================

        if len(
            matched_indices
        ) == 0:

            k = min(
                TOP_K_FALLBACK,
                len(ious)
            )

            matched_indices = (
                np.argsort(
                    ious
                )[-k:]
            )


        # ====================================================
        # MATCH WEIGHTS
        # ====================================================

        weights = ious[
            matched_indices
        ]

        weights = np.maximum(
            weights,
            0
        )


        # ====================================================
        # TARGET SCORE
        # ========================================================

        # YOLOX raw channel layout:
        #
        # 0 cx
        # 1 cy
        # 2 w
        # 3 h
        # 4 objectness
        # 5... class scores

        objectness = predictions[
            matched_indices,
            4
        ]

        class_scores = predictions[
            matched_indices,
            5 + target_class
        ]


        target_scores = (
            objectness
            *
            class_scores
        )


        weight_tensor = torch.tensor(
            weights,
            dtype=target_scores.dtype,
            device=device
        )


        # ====================================================
        # BACKWARD TARGET
        # ====================================================

        target_score = (
            target_scores
            *
            weight_tensor
        ).sum()


        # ====================================================
        # BACKPROPAGATION
        # ====================================================

        target_score.backward(
            retain_graph=False
        )


        # ====================================================
        # ACTIVATION + GRADIENT
        # ====================================================

        if not activation:
            raise RuntimeError(
                "YOLOX activation was not captured."
            )

        if not gradient:
            raise RuntimeError(
                "YOLOX gradient was not captured."
            )


        activations = activation[
            0
        ]

        gradients = gradient[
            0
        ]


        # ====================================================
        # LAYERCAM
        # ====================================================

        positive_gradients = torch.relu(
            gradients
        )

        cam = (
            positive_gradients
            *
            activations
        ).sum(
            dim=1
        )


        cam = torch.relu(
            cam
        )


        cam = cam[
            0
        ]


        # ====================================================
        # NORMALIZE
        # ====================================================

        cam = cam.detach().cpu().numpy()

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:
            cam /= max_value


        # ====================================================
        # RESIZE TO ORIGINAL IMAGE
        # ====================================================

        cam = cv2.resize(
            cam,
            (
                original_width,
                original_height
            ),
            interpolation=cv2.INTER_LINEAR
        )


        # ====================================================
        # HEATMAP
        # ====================================================

        heatmap = np.uint8(
            255 * cam
        )

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )


        # ====================================================
        # OVERLAY
        # ====================================================

        overlay = cv2.addWeighted(
            image,
            0.55,
            heatmap_color,
            0.45,
            0
        )


        # ====================================================
        # DRAW DETECTION
        # ====================================================

        x1, y1, x2, y2 = [
            int(v)
            for v in target_box_original
        ]


        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )


        label = (
            f"{detection['class_name']} "
            f"{detection['confidence'] * 100:.2f}%"
        )


        cv2.putText(
            overlay,
            label,
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # RETURN
        # ====================================================

        return {
            "heatmap": heatmap_color,
            "overlay": overlay,
            "grayscale_cam": cam,
            "class_id": target_class,
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],
            "bbox": target_box_original.tolist(),
            "matched_anchor_count": int(
                len(matched_indices)
            ),
        }


    finally:

        # ====================================================
        # REMOVE HOOKS
        # ====================================================

        forward_handle.remove()
        backward_handle.remove()