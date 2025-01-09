import os

class inference():
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless) -> None:
        self.sim_params = sim_params
        self.device = 'cpu'

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.get_privileged_obs = cfg.env.get_privileged_obs
        self.num_actions = 5 if cfg.env.ctl_mode == "atti" else 4

        print(f'ctl_mode ================= {cfg.env.ctl_mode}')
        print(f'num_obs =================== {self.num_obs}')
        print(f'num_actions =================== {self.num_actions}')