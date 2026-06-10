
cd /opt/sim_soccer2/MotrixLab
conda activate sim_soccer_rl

python -u scripts/train.py \
  --env k1-point-navigate \
  --rllib rslrl \
  --num-envs 2048 \
  --resume-policy runs/k1-flat-terrain-walk/rslrl/26-05-25_12-22-51-_91128_PPO/model_1675.pt \
  --resume-noise-std 0.12 \
  --max-iterations 500
