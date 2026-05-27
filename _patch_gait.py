import sys
path = 'MotrixLab/motrix_envs/src/motrix_envs/locomotion/k1/walk_ball_np.py'
with open(path, 'r') as f:
    content = f.read()

old = '''    def _moving_gait_mask(self, info: dict) -> np.ndarray:
        moving = np.linalg.norm(info["commands"][:, :2], axis=1) > 0.1
        dist_ratio = np.clip(info["ball_dist"] / max(self.cfg.ball_config.arrival_radius, 0.01), 0.3, 1.0)
        return (moving.astype(np.float32) * dist_ratio).astype(np.float32)'''

new = '''    def _moving_gait_mask(self, info: dict) -> np.ndarray:
        moving = np.linalg.norm(info["commands"][:, :2], axis=1) > 0.1
        approaching = info["ball_dist"] > self.cfg.ball_config.arrival_radius
        return (moving & approaching).astype(np.float32)'''

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('_moving_gait_mask reverted')
else:
    print('PATTERN NOT FOUND - may already be reverted')
