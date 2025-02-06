import os
import yaml
import copy

from one import load_system
from src.systems.load import get_system_cls
from .particle import ModelParticle, system2particle


def load_exp0_results():
    res = {}
    particles = []
    ckpt_paths = []
    for accent in ["eng", "aus", "ind", "sco", "ire"]:
        ckpt_path = f"results/exp0/{accent}/ckpt/best.ckpt"
        config = yaml.load(open(f"results/exp0/{accent}/config.yaml", "r"), Loader=yaml.FullLoader)
        system_cls = get_system_cls(config["system_name"])
        system = system_cls.load_from_checkpoint(ckpt_path)
        particles.append(system2particle(system))
        ckpt_paths.append(ckpt_path)
        # print(len(particles[-1].get_data().keys()))
    res["particles"] = particles
    res["raw_paths"] = ckpt_paths

    # load pretrained system used in exp0
    res["ref_system"] = load_system(
        system_name="wav2vec2",
        system_config=yaml.load(open("config/system/base.yaml", "r"), Loader=yaml.FullLoader)
    )
    return res


def load_exp0_results_long(tnames: list[str]):
    res = {}
    ckpt_paths = []
    # tnames = [
    #     "LS_AA_5", "cv-eng", "LS_AC_5", "cv-aus", "LS_BA_5", "LS_CM_5", "LS_MU_5",
    #           "cv-ind", "LS_NB_5", "LS_SD_5", "LS_TP_5", "cv-ire", "cv-sco", "LS_VC_5"]
    # tnames = [
    #     "LS_TP_5", "LS_VC_5", "cv-ire", "LS_SD_5", "cv-aus", "LS_MU_5", "LS_BA_5",
    #     "cv-ind", "LS_AA_5", "LS_NB_5", "LS_CM_5", "cv-eng", "LS_AC_5", "cv-sco"
    # ]
    for tname in tnames:
        ckpt_path = f"results/exp0/{tname}/ckpt/best.ckpt"
        ckpt_paths.append(ckpt_path)
    res["raw_paths"] = ckpt_paths
    
    # define getter
    def particle_getter(tid: int):
        system = load_system(
            system_name="wav2vec2",
            checkpoint=res["raw_paths"][tid],
            loader="lightning"
        )
        return system2particle(system)

    return res, particle_getter


def load_cl_results(exp_dir: str) -> list[ModelParticle]:
    res = {}
    particles = []
    tid = 0
    while 1:
        tid += 1
        ckpt_path = f"{exp_dir}/ckpt/tid={tid}.ckpt"
        if not os.path.exists(ckpt_path):
            break
        config = yaml.load(open(f"{exp_dir}/config.yaml", "r"), Loader=yaml.FullLoader)
        system = load_system(
            system_name=config["strategy_config"]["system_name"],
            system_config=copy.deepcopy(config["system_config"]),
            checkpoint=ckpt_path,
            loader="torch"
        )
        particles.append(system2particle(system))
    res["particles"] = particles
    
    return res
