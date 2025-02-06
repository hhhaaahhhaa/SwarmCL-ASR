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


class Long2ASequence(TaskSequence):
    def __init__(self):
        tnames = [
            "LS_AA_5", "cv-eng", "LS_AC_5", "cv-aus", "LS_BA_5", "LS_CM_5", "LS_MU_5",
            "cv-ind", "LS_NB_5", "LS_SD_5", "LS_TP_5", "cv-sco", "cv-ire", "LS_VC_5"
        ]
        super().__init__(tnames)


class Long2BSequence(TaskSequence):
    def __init__(self):
        tnames = [
            "LS_TP_5", "LS_VC_5", "cv-ire", "LS_SD_5", "cv-aus", "LS_MU_5", "LS_BA_5",
            "cv-ind", "LS_AA_5", "LS_NB_5", "LS_CM_5", "cv-eng", "LS_AC_5", "cv-sco"
        ]
        super().__init__(tnames)


class Long3ASequence(TaskSequence):
    def __init__(self):
        tnames = [
            'LS_NB_5', 'cv-sco_GS_5', 'cv-aus_GS_5', 'LS_GS_5', 'cv-eng',
            'cv-ind_GS_5', 'LS_SD_5', 'cv-ind', 'cv-sco', 'LS_MU_5',
            'LS_AA_5', 'LS_BA_5', 'cv-ire', 'cv-ire_GS_5', 'LS_AC_5',
            'cv-eng_GS_5', 'LS_CM_5', 'cv-aus', 'LS_VC_5', 'LS_TP_5'
        ]
        super().__init__(tnames)


class Long3BSequence(TaskSequence):
    def __init__(self):
        tnames = [
            'LS_SD_5', 'LS_AA_5', 'cv-ind_GS_5', 'cv-ire_GS_5', 'LS_MU_5',
            'cv-eng', 'cv-aus_GS_5', 'cv-aus', 'cv-eng_GS_5', 'LS_AC_5', 
            'LS_NB_5', 'LS_VC_5', 'LS_GS_5', 'LS_TP_5', 'LS_BA_5',
            'cv-sco', 'LS_CM_5', 'cv-ind', 'cv-sco_GS_5', 'cv-ire'
        ]
        super().__init__(tnames)


class Long3CSequence(TaskSequence):
    def __init__(self):
        tnames = [
            'LS_BA_5', 'LS_AC_5', 'cv-ind_GS_5', 'LS_CM_5', 'cv-ire_GS_5',
            'cv-eng_GS_5', 'LS_VC_5', 'cv-eng', 'cv-sco', 'LS_MU_5',
            'LS_GS_5', 'cv-ire', 'LS_NB_5', 'LS_TP_5', 'cv-aus_GS_5',
            'LS_AA_5', 'LS_SD_5', 'cv-sco_GS_5', 'cv-ind', 'cv-aus'
        ]
        super().__init__(tnames)
