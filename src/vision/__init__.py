"""Oriented vehicle detection harness for MTC Peru intersections."""

__version__ = "0.1.0"

CLASSES = {
    1: "auto",
    2: "combi",
    3: "microbus",
    4: "minibus",
    5: "omnibus",
    6: "articulado",
    7: "camion",
    8: "mototaxi",
    9: "motocicleta",
}
CLASS_IDS = tuple(CLASSES.keys())
NUM_CLASSES = len(CLASSES)

# Official rIoU thresholds for local metric reproduction.
RIOU_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
