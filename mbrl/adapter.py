"""mjlab bridge with terminal-state capture and explicit simulator context."""
import types
import torch
from .core import proprio
from .sampling import ReturnSampler


class Adapter:
    def __init__(self, kind, motion, nenv, seed, history, evaluation=False, simple=False, return_sampling=True):
        from mjlab.envs import ManagerBasedRlEnv
        if kind == 'tracking':
            from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
            cfg = unitree_g1_flat_tracking_env_cfg()
            # Use the same MuJoCo assets as the supplied motion data.
            from src.assets.robots import get_g1_robot_cfg, G1_ACTION_SCALE
            cfg.scene.entities = {'robot': get_g1_robot_cfg()}
            cfg.actions['joint_pos'].scale = G1_ACTION_SCALE
            cfg.commands['motion'].motion_file = motion
            # Root reward and termination now consistently refer to pelvis.
            cfg.commands['motion'].anchor_body_name = 'pelvis'
            cfg.rewards.pop('self_collisions', None)
            cfg.rewards['joint_limit'].weight = -2.
            cfg.terminations.pop('ee_body_pos', None)
        else:
            raise ValueError('Only motion tracking is supported')
        cfg.scene.num_envs, cfg.seed = nenv, seed
        # Unobserved velocity impulses invalidate a deterministic surrogate.
        cfg.events.pop('push_robot', None)
        cfg.curriculum = {}
        if evaluation or simple:
            for key in ['foot_friction', 'base_com', 'encoder_bias']:
                cfg.events.pop(key, None)
            if kind == 'tracking':
                cfg.commands['motion'].sampling_mode = 'start'
                cfg.commands['motion'].pose_range = {}
                cfg.commands['motion'].velocity_range = {}
                cfg.commands['motion'].joint_position_range = (0.,0.)
                cfg.episode_length_s = 100000.
        if simple and not evaluation:
            cfg.episode_length_s = 20.
        for group in cfg.observations.values():
            group.enable_corruption = False
            group.history_length = 1
            for term in group.terms.values():
                term.history_length = 0
        self.env = ManagerBasedRlEnv(cfg, 'cuda:0')
        self.kind, self.history = kind, history
        self.reward_scale = 1.
        self.obs, _ = self.env.reset()
        self.na = self.env.action_manager.total_action_dim
        if self.na != 29 or self.env.sim.data.qpos.shape[-1] != 36:
            raise ValueError('MBRL implementation requires a free-base 29-DoF G1')
        robot = self.env.scene['robot']
        if not torch.equal(robot.indexing.joint_q_adr, torch.arange(7,36,device='cuda:0')):
            raise ValueError('Robot joint ordering does not match qpos layout')
        self.default_pose = robot.data.default_joint_pos[0].clone()
        self.no = 93
        self.nf = 0
        self.noise = torch.tensor([.01]*29+[.5]*29+[.2]*3+[.05]*3+[0.]*29, device='cuda:0')
        self.obs['actor'] = self.proprio()
        self.h = self.noisy(self.obs['actor'])[:, None].repeat(1, history, 1)
        self.return_sampler = None
        self.terminal_state = None
        self.terminal_feat = None
        reset = self.env._reset_idx
        def capture(env, ids):
            # Captured before auto-reset. No transition is trained toward a new episode.
            self.terminal_state = self.state()
            return reset(ids)
        self.env._reset_idx = types.MethodType(capture, self.env)
        if kind == 'tracking':
            cmd = self.env.command_manager.get_term('motion')
            m = cmd.motion
            self.goal_table = torch.cat((m.joint_pos, m.joint_vel,
                m.body_pos_w.flatten(1), m.body_quat_w.flatten(1),
                m.body_lin_vel_w.flatten(1), m.body_ang_vel_w.flatten(1)), -1)
            if return_sampling and not evaluation:
                self.return_sampler = ReturnSampler(m.time_step_total, self.env.step_dt, 'cuda:0')
                cmd.cfg.sampling_mode = 'adaptive'
                sampler = self.return_sampler
                def sample_reset(command, ids):
                    command.time_steps[ids] = sampler.sample(len(ids))
                cmd._adaptive_sampling = types.MethodType(sample_reset, cmd)
        self.schema = {'joint_names': list(self.env.scene['robot'].joint_names),
                       'body_names': list(self.env.scene['robot'].body_names),
                       'actor_terms': ['joint_pos_rel','joint_vel','pelvis_ang_vel_local','pelvis_gravity','previous_action'],
                       'dt': self.env.step_dt, 'history': history}

    def proprio(self):
        return proprio(self.state(), self.default_pose, self.context()[..., -29:])

    def noisy(self, obs):
        return obs + (torch.rand_like(obs)*2-1)*self.noise

    def state(self):
        qpos = self.env.sim.data.qpos.clone()
        qpos[:, :3] -= self.env.scene.env_origins
        return torch.cat((qpos, self.env.sim.data.qvel.clone(),
                          self.env.action_manager.action.clone()), -1)

    def context(self):
        model, robot = self.env.sim.model, self.env.scene['robot']
        return torch.cat((model.geom_friction.clone().flatten(1),
                          model.body_ipos.clone().flatten(1), robot.data.encoder_bias.clone()), -1)

    def goal(self):
        if self.kind == 'tracking':
            idx = self.env.command_manager.get_term('motion').time_steps.clone()
            return self.goal_table[idx], idx.float().unsqueeze(-1)

    def goals(self, meta, horizon):
        if self.kind == 'tracking':
            idx = meta[:, 0].long()[None] + torch.arange(horizon+1, device=meta.device)[:, None]
            return self.goal_table[idx.clamp_max(len(self.goal_table)-1)]

    @torch.no_grad()
    def step(self, actor, action_noise=.1):
        g, meta = self.goal()
        row = dict(s=self.state(), c=self.context(), g=g, meta=meta, h=self.h.clone(),
                   o=self.obs['actor'].clone())
        goal_window = self.goals(meta, actor.goal_frames-1).transpose(0,1)
        action = actor(self.h, goal_window) + torch.randn(self.env.num_envs, self.na, device='cuda:0')*action_noise
        self.terminal_state = self.terminal_feat = None
        obs, reward, term, timeout, _ = self.env.step(action)
        nxt = self.state()
        reset = term | timeout
        if reset.any():
            assert self.terminal_state is not None
            nxt[reset] = self.terminal_state[reset]
        row['onext'] = proprio(nxt, self.default_pose, row['c'][..., -29:])
        row['gnext'] = self.goals(meta,1)[1]
        row.update(a=action, nxt=nxt, r=reward.clone()*self.reward_scale,
                   term=term.float().clone(), reset=reset.float().clone(),
                   valid=torch.ones_like(reward))
        if self.kind == 'tracking':
            # Motion-command end-of-clip resampling can teleport the robot
            # without passing through env._reset_idx. Exclude those transitions.
            after = self.env.command_manager.get_term('motion').time_steps
            jump = (after != meta[:,0].long()+1) & ~reset
            row['valid'][jump] = 0
            row['reset'][jump] = 1
            reset = reset | jump
        if self.return_sampler is not None:
            self.return_sampler.add(meta[:,0], row['r'], row['reset'], after)
        obs['actor'] = self.proprio()
        self.obs = obs
        frame = self.noisy(obs['actor'])
        self.h = torch.cat((self.h[:, 1:], frame[:, None]), 1)
        self.h[reset] = frame[reset, None].repeat(1, self.history, 1)
        return row


class Replay:
    """FIFO time x environment ring; contiguous sequences never cross overwritten data."""
    def __init__(self, capacity):
        self.capacity, self.count, self.buffers = capacity, 0, {}

    def add(self, row):
        if not self.buffers:
            self.buffers = {k: torch.empty((self.capacity, *v.shape), dtype=v.dtype, device=v.device) for k,v in row.items()}
        for k, v in row.items():
            self.buffers[k][self.count % self.capacity].copy_(v)
        self.count += 1

    def sample(self, length, batch, validation=False):
        size = min(self.count, self.capacity)
        if size < length:
            raise ValueError('Not enough chronological replay steps')
        nenv = self.buffers['s'].shape[1]
        # Hold out whole environment streams for model diagnostics.
        split = max(1, int(nenv*.8))
        lo, hi = (split, nenv) if validation else (0, split)
        env = torch.randint(lo, hi, (batch,), device=self.buffers['s'].device)
        first = torch.randint(0, size-length+1, (batch,), device=env.device)
        idx = (first[None]+torch.arange(length, device=env.device)[:, None]+max(0,self.count-self.capacity)) % self.capacity
        return {k:v[idx, env[None]] for k,v in self.buffers.items()}
