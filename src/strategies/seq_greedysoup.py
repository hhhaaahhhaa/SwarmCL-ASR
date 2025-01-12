import os
import copy
import yaml
from functools import partial

from one import train_one_task, load_system
from src.strategies.base import IStrategy
from src.utils.tool import wer
from .common.greedysoup import GreedySoupExecutor
from .swarm_cl.particle import ModelParticle, linear_combination, system2particle, particle2system


class SeqGreedySoupStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

    def _get_exp_root(self, tid: int):
        return f"{self.config['output_dir']['log_dir']}/tid={tid+1}"
    
    def _get_checkpoint_path(self, tid: int):
        return f"{self.config['output_dir']['ckpt_dir']}/tid={tid+1}.ckpt"
    
    def _get_initial_system(self):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            system_config=copy.deepcopy(self.config["system_config"])
        )

    def _get_ft_system(self, tid: int):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            checkpoint=f"{self._get_exp_root(tid)}/ckpt/best.ckpt",
            loader="lightning"
        )

    def _get_merged_system(self, tid: int):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            system_config=copy.deepcopy(self.config["system_config"]),
            checkpoint=self._get_checkpoint_path(tid),
            loader="torch"
        )

    def _finetune(self, tid: int, task_name: str):
        # create full configuration
        task_config = {
            "system_name": self.config["strategy_config"]["system_name"],
            "task_name": task_name,
            "checkpoint": None if tid == 0 else self._get_checkpoint_path(tid-1),
            "config": copy.deepcopy(self.config["system_config"]),
        }
        exp_root = self._get_exp_root(tid)
        os.makedirs(exp_root, exist_ok=True)
        task_config["config"]["output_dir"] = {
            "exp_root": exp_root,
            "log_dir": f"{exp_root}/log",
            "result_dir": f"{exp_root}/result",
            "ckpt_dir": f"{exp_root}/ckpt"
        }
        with open(f"{exp_root}/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(task_config, f, sort_keys=False)
        
        train_one_task(task_config, loader="torch")

    def _eval_particle(self, particle: ModelParticle, ds) -> float:
        if getattr(self, "ref_system_for_eval", None) is None:
            self.ref_system_for_eval = self._get_initial_system()
        system = particle2system(particle, ref_system=self.ref_system_for_eval)
        system.eval()
        system.cuda()
        gt, predictions = [], []
        for sample in ds:
            predictions.extend(system.inference([sample["wav"]]))
            gt.append(sample["text"])
        word_error_rate = wer(gt, predictions)
        return -word_error_rate
    
    def _load_particles(self, tid: int) -> list[ModelParticle]:
        # determine checkpoints that will participate
        include_last_k = self.config["strategy_config"]["include_last_k"]
        if include_last_k == -1:  # include all
            include_last_k = 2e9
        
        res = []
        tmp = tid
        while 1:
            if tmp >= 0:
                if tmp < tid:  # first iteration has no merged checkpoint since that is what we want to generate
                    res.append(system2particle(self._get_merged_system(tmp)))
                    if len(res) == include_last_k:
                        break
                res.append(system2particle(self._get_ft_system(tmp)))
                if len(res) == include_last_k:
                    break
                tmp -= 1
            else:  # the last iteration, add pretrained checkpoint
                res.append(system2particle(self._get_initial_system()))
                break  # ensure breaking the while loop
        return res

    def _load_start(self, tid, data_obj):
        pass

    def _integrate(self, tid, data_obj):
        soup_config = {"cache_dir": self._get_exp_root(tid)}
        executor = GreedySoupExecutor(
            soup_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=partial(self._eval_particle, ds=data_obj.get_buffer(tid))
        )

        particles = self._load_particles(tid)
        merged_particle = executor.run(particles)
        merged_system = particle2system(merged_particle, ref_system=self._get_initial_system())
        merged_system.save(self._get_checkpoint_path(tid))

    def run(self, data_obj):
        task_name, data_obj = data_obj
        assert task_name in ["cv-seq", "cv-seq-500"]
        for tid, task_name in enumerate(data_obj.task_names):
            # start
            self._load_start(tid, data_obj)
            self._finetune(tid, task_name)
            self._integrate(tid, data_obj)


class CGreedySoupStrategy(SeqGreedySoupStrategy):
    def _load_particles(self, tid: int) -> list[ModelParticle]:
        # determine checkpoints that will participate
        include_last_k = 2e9  # include all
        
        res = []
        tmp = tid
        while 1:
            if tmp >= 0:
                res.append(system2particle(self._get_ft_system(tmp)))
                if len(res) == include_last_k:
                    break
                tmp -= 1
            else:  # the last iteration, add pretrained checkpoint
                res.append(system2particle(self._get_initial_system()))
                break  # ensure breaking the while loop
        return res


class CDoubleSoupStrategy(SeqGreedySoupStrategy):
    def _get_merged_path(self, tid: int):
        return f"{self.config['output_dir']['ckpt_dir']}/tid={tid+1}-merged.ckpt"
    
    def _get_merged_system(self, tid: int):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            system_config=copy.deepcopy(self.config["system_config"]),
            checkpoint=self._get_merged_path(tid),
            loader="torch"
        )
    
    def _load_particles(self, tid: int, mode="ft") -> list[ModelParticle]:
        # determine checkpoints that will participate
        include_last_k = 2e9  # include all
        
        res = []
        tmp = tid
        while 1:
            if tmp >= 0:
                if mode == "merged":  # please be CAREFUL to call this only after merged checkpoint is generated
                    res.append(system2particle(self._get_merged_system(tmp)))
                    if len(res) == include_last_k:
                        break
                if mode == "ft":
                    res.append(system2particle(self._get_ft_system(tmp)))
                    if len(res) == include_last_k:
                        break
                tmp -= 1
            else:  # the last iteration, add pretrained checkpoint
                res.append(system2particle(self._get_initial_system()))
                break  # ensure breaking the while loop
        return res

    def _integrate(self, tid, data_obj):
        ds = data_obj.get_buffer(tid)
        soup_config = {"cache_dir": f"{self._get_exp_root(tid)}/1st-merge"}
        executor = GreedySoupExecutor(
            soup_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=partial(self._eval_particle, ds=ds)
        )

        particles = self._load_particles(tid, mode="ft")
        merged_particle = executor.run(particles)
        merged_system = particle2system(merged_particle, ref_system=self._get_initial_system())
        merged_system.save(self._get_merged_path(tid))

        soup_config = {"cache_dir": f"{self._get_exp_root(tid)}/2nd-merge"}
        executor = GreedySoupExecutor(
            soup_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=partial(self._eval_particle, ds=ds)
        )

        particles = self._load_particles(tid, mode="merged")
        double_merged_particle = executor.run(particles)
        double_merged_system = particle2system(double_merged_particle, ref_system=self._get_initial_system())
        double_merged_system.save(self._get_checkpoint_path(tid))
