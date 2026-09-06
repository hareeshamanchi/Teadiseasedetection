import torch

from yolox.exp import get_exp


# ============================================================
# YOLOX MODEL CONFIGURATION
# ============================================================

CONFIG_PATH = "models/YOLOX/yolox_config.py"
CHECKPOINT_PATH = "models/YOLOX/best_ckpt (2).pth"


def load_yolox_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("LOADING YOLOX-M MODEL")
    print("=" * 70)

    print("Config     :", CONFIG_PATH)
    print("Checkpoint :", CHECKPOINT_PATH)
    print("Device     :", device)

    # Load exact YOLOX experiment
    exp = get_exp(CONFIG_PATH, None)

    # Build YOLOX-M architecture
    model = exp.get_model()

    # Load checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    model.to(device)
    model.eval()

    # Enable gradients for Grad-CAM
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    print("\nYOLOX-M loaded successfully.")
    print("Device      :", device)
    print("Classes     :", exp.num_classes)
    print("Input size  :", exp.input_size)
    print("Test size   :", exp.test_size)

    return model, exp, device