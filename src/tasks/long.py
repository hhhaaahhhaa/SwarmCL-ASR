from .utils import TaskSequence


class Long1Sequence(TaskSequence):
    def __init__(self):
        tnames = ["LS_AA_5", "LS_AC_5", "LS_BA_5", "LS_CM_5", "LS_MU_5",
              "LS_NB_5", "LS_SD_5", "LS_TP_5", "LS_VC_5"]
        super().__init__(tnames)


class Long2Sequence(TaskSequence):
    def __init__(self):
        tnames = [
            "cv-eng", "cv-aus", "cv-ind", "cv-sco", "cv-ire",
            "LS_AA_5", "LS_AC_5", "LS_BA_5", "LS_CM_5", "LS_MU_5",
            "LS_NB_5", "LS_SD_5", "LS_TP_5", "LS_VC_5"
        ]
        super().__init__(tnames)
