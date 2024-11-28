from src.utils.load import get_class_in_module
from .base import Task


SRC_DIR = "src/tasks"

CV_ACCENT = {
    "cv-aus": (f"{SRC_DIR}/commonvoice.py", "AUSTask"),
    "cv-eng": (f"{SRC_DIR}/commonvoice.py", "ENGTask"),
    "cv-ind": (f"{SRC_DIR}/commonvoice.py", "INDTask"),
    "cv-ire": (f"{SRC_DIR}/commonvoice.py", "IRETask"),
    "cv-sco": (f"{SRC_DIR}/commonvoice.py", "SCOTask"),
    "cv-us": (f"{SRC_DIR}/commonvoice.py", "USTask"),
    "cv-all": (f"{SRC_DIR}/commonvoice.py", "AllTask"),
}

EXP = {
    
}

TASK_MAPPING = {
    **CV_ACCENT,
    **EXP,
}


def get_task(name) -> Task:
    module_path, class_name = TASK_MAPPING[name]
    return get_class_in_module(class_name, module_path)()
