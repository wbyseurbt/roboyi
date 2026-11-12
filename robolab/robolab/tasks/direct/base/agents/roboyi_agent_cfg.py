"""PPO settings for roboyi; no RPO 23-joint mirror map is used."""

from isaaclab.utils import configclass

from .rpo_agent_cfg import RPOFlatAgentCfg


@configclass
class RoboyiFlatAgentCfg(RPOFlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "roboyi_flat"
        self.wandb_project = "roboyi_flat"
        self.logger = "tensorboard"
        self.save_interval = 500
        self.policy.init_noise_std = 0.5
        self.clip_actions = 3.0
        self.algorithm.symmetry_cfg = None
