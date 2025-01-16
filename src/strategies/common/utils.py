import os
import yaml
import copy

from one import load_system
from src.systems.load import get_system_cls
from .particle import ModelParticle, system2particle


def load_exp0_results():
    res = {}
    particles = []
    for accent in ["aus", "eng", "ind", "ire", "sco"]:
        ckpt_path = f"results/exp0/{accent}/ckpt/best.ckpt"
        config = yaml.load(open(f"results/exp0/{accent}/config.yaml", "r"), Loader=yaml.FullLoader)
        system_cls = get_system_cls(config["system_name"])
        system = system_cls.load_from_checkpoint(ckpt_path)
        particles.append(system2particle(system))
        # print(len(particles[-1].get_data().keys()))
    res["particles"] = particles

    # load pretrained system used in exp0
    res["ref_system"] = load_system(
        system_name="wav2vec2",
        system_config=yaml.load(open("config/system/base.yaml", "r"), Loader=yaml.FullLoader)
    )
    return res


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
