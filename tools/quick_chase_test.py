#!/usr/bin/env python3
"""Quick test: run sim + decider with ball rotation and observe the chase."""
import json, math, os, signal, subprocess, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SIM = PROJECT / "simulation" / "motrixsim" / "sim2sim_runner.py"
DECIDER = PROJECT / "decider" / "decider.py"
POLICY = PROJECT / "MotrixLab" / "exported" / "k1_point_navigate_torchscript.pt"
SIM_ENV = "/home/1ctnltug/miniconda3/envs/motrixsim0508/bin/python"
DEC_ENV = "/home/1ctnltug/miniconda3/envs/k1/bin/python"
SIM_PORT = 5556

ROT = os.environ.get("ROT", "90")
SIGN = os.environ.get("SIGN", "-1")
DURATION = int(os.environ.get("DUR", "30"))

DEC_LOG = PROJECT / "logs" / "quick_chase_decider.log"
DEC_LOG.parent.mkdir(parents=True, exist_ok=True)

CHASE_TRACE = Path("/tmp/chase_trace.txt")
GAME_TRACE = Path("/tmp/game_trace.txt")
CHASE_TRACE.unlink(missing_ok=True)
if GAME_TRACE.exists():
    GAME_TRACE.unlink(missing_ok=True)

def kill_port(p):
    subprocess.run(["fuser", "-k", f"{p}/tcp"], stderr=subprocess.DEVNULL)

# Clean ports
for p in [SIM_PORT]:
    kill_port(p)
time.sleep(1)

# Start sim
print(f"[*] Starting sim on ZMQ port {SIM_PORT} (no webview)...")
sim = subprocess.Popen(
    [SIM_ENV, str(SIM), "--team-size", "1", "--real-time",
     "--no-webview", "--port", str(SIM_PORT),
     "--k1-policy-flavor", "motrixlab",
     "--policy", str(POLICY),
     "--match-config", "/tmp/match_config_chase.json"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    preexec_fn=os.setsid,
)

# Wait for sim ZMQ to be ready
print("[*] Waiting for sim ZMQ to be ready...")
time.sleep(3)
print("[*] Sim should be ready.")

# Start decider
env = os.environ.copy()
env["K1_SIM_BALL_ROT"] = ROT
env["K1_FORWARD_SIGN"] = SIGN
print(f"[*] Starting decider (ROT={ROT}°, SIGN={SIGN})...")
dec_log_fh = open(str(DEC_LOG), "w")
dec = subprocess.Popen(
    [DEC_ENV, "-u", str(DECIDER), "--simulation", "--ip", "127.0.0.1", "--port", str(SIM_PORT),
     "--color", "red", "--id", "0", "--sim-hz", "10"],
    stdout=dec_log_fh, stderr=subprocess.STDOUT,
    preexec_fn=os.setsid, env=env,
)

# Monitor via trace files
print(f"[*] Running for {DURATION}s...")
start = time.time()
last_chase_lines = 0
last_game_lines = 0
while time.time() - start < DURATION:
    time.sleep(1)
    elapsed = time.time() - start
    # Check game trace
    if GAME_TRACE.exists():
        with open(GAME_TRACE) as f:
            game_lines = f.readlines()
            if len(game_lines) > last_game_lines:
                for line in game_lines[last_game_lines:]:
                    print(f"  [GAME] {line.rstrip()}")
                last_game_lines = len(game_lines)
    # Check chase trace
    if CHASE_TRACE.exists():
        with open(CHASE_TRACE) as f:
            chase_lines = f.readlines()
            if len(chase_lines) > last_chase_lines:
                for line in chase_lines[last_chase_lines:]:
                    print(f"  [CHASE] {line.rstrip()}")
                last_chase_lines = len(chase_lines)
    print(f"  t={elapsed:3.0f}s  game_calls={last_game_lines}  chase_calls={last_chase_lines}")

# Stop
print("[*] Stopping...")
os.killpg(os.getpgid(dec.pid), signal.SIGTERM)
os.killpg(os.getpgid(sim.pid), signal.SIGTERM)
dec.wait(timeout=5)
sim.wait(timeout=5)
print("[*] Done. Decider log at logs/quick_chase_decider.log")

# Dump decider log
print("\n--- Decider log ---")
try:
    with open(DEC_LOG) as f:
        for line in f:
            print(f"  {line.rstrip()}")
except FileNotFoundError:
    print("  (no log file)")
