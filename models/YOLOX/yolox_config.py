from yolox.exp import Exp as MyExp


class Exp(MyExp):

    def __init__(self):
        super().__init__()

        # =========================================================
        # DATASET
        # =========================================================

        self.num_classes = 4

        # These paths are NOT used during Streamlit inference.
        # They are kept here to reproduce the training experiment.
        self.data_dir = ""

        self.train_ann = ""
        self.val_ann = ""

        self.train_name = ""
        self.val_name = ""


        # =========================================================
        # YOLOX-M ARCHITECTURE
        # =========================================================

        self.depth = 0.67
        self.width = 0.75


        # =========================================================
        # IMAGE SIZE
        # =========================================================

        self.input_size = (800, 800)
        self.test_size = (800, 800)


        # =========================================================
        # TEST / INFERENCE
        # =========================================================

        self.test_conf = 0.01
        self.nmsthre = 0.65


# =============================================================
# YOLOX CLASS NAMES
# =============================================================

YOLOX_CLASS_NAMES = [
    "Algal Leaf Spot",
    "Anthracnose",
    "Brown Spot Disease",
    "White Spot Disease",
]