"""学習済みの Go2 方策を、プラン(JSON)の速度コマンド列どおりに動かして mp4 に録画する"""
import argparse
import glob
import json
import os
import pickle
import re

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from go2_env import Go2Env

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", default="go2-omni")
parser.add_argument("--plan", default="plan.json")
parser.add_argument("--out", default="demo.mp4")
parser.add_argument("--ckpt", type=int, default=-1, help="-1 なら最新のチェックポイント")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--render_every", type=int, default=2, help="何ステップごとに1フレーム描画するか")
args = parser.parse_args()

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    FONT_L, FONT_S = ImageFont.truetype(FONT_PATH, 34), ImageFont.truetype(FONT_PATH, 24)
except OSError:
    FONT_L = FONT_S = ImageFont.load_default()
CAM_OFFSET = np.array([-2.2, -2.2, 1.3])  # ロボットの斜め後ろから追いかける


def first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def draw_overlay(rgb, seg, cmd, t):
    img = Image.fromarray(rgb).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    W, H = img.size
    d.rectangle([0, 0, W, 112], fill=(0, 0, 0, 150))
    d.text((24, 12), f"指示: {seg.get('instruction', '')}", font=FONT_L, fill=(255, 255, 255, 255))
    d.text((24, 66), f"→ {seg.get('caption', '')}    vx={cmd[0]:+.2f} m/s   vy={cmd[1]:+.2f} m/s   wz={cmd[2]:+.2f} rad/s",
           font=FONT_S, fill=(170, 225, 255, 255))
    d.text((W - 160, H - 44), f"t = {t:5.1f} s", font=FONT_S, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))


gs.init(backend=gs.gpu, logging_level="warning")

log_dir = f"logs/{args.exp_name}"
env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
reward_cfg["reward_scales"] = {}
env_cfg["episode_length_s"] = 10_000.0  # 録画中にタイムアウトでリセットされないように

# Go2Env はコンストラクタ内で scene.build() まで行うので、build 直前にカメラを差し込む
_orig_build = gs.Scene.build
CAM = {}


def _build_with_camera(self, *a, **kw):
    CAM["cam"] = self.add_camera(res=(args.width, args.height), pos=(2.5, -2.5, 1.5),
                                 lookat=(0, 0, 0.3), fov=45, GUI=False)
    return _orig_build(self, *a, **kw)


gs.Scene.build = _build_with_camera

env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
             command_cfg=command_cfg, show_viewer=False)
cam = CAM["cam"]
env._resample_commands = lambda *a, **kw: None  # ランダムなコマンド再抽選を止め、プランのコマンドだけを使う

ckpts = sorted(glob.glob(f"{log_dir}/model_*.pt"), key=lambda p: int(re.findall(r"model_(\d+)", p)[-1]))
ckpt_path = ckpts[-1] if args.ckpt < 0 else f"{log_dir}/model_{args.ckpt}.pt"
print("checkpoint:", ckpt_path)
runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
runner.load(ckpt_path)
policy = runner.get_inference_policy(device=gs.device)

# 学習した範囲外のコマンドは方策が対応できないので、範囲内にクリップする（安全制約）
ranges = [command_cfg["lin_vel_x_range"], command_cfg["lin_vel_y_range"], command_cfg["ang_vel_range"]]
plan = json.load(open(args.plan, encoding="utf-8"))

fps = round(1.0 / (env.dt * args.render_every))
writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8, macro_block_size=1)

obs = first(env.reset())
step, look = 0, None
with torch.no_grad():
    for i, seg in enumerate(plan):
        cmd = [float(np.clip(seg.get(k, 0.0), lo, hi)) for k, (lo, hi) in zip(("vx", "vy", "wz"), ranges)]
        cmd_t = torch.tensor(cmd, device=gs.device, dtype=torch.float32)
        print(f"[{i + 1}/{len(plan)}] {seg.get('caption', '')}  cmd={cmd}  {seg['duration']}s")
        for _ in range(int(seg["duration"] / env.dt)):
            env.commands[:] = cmd_t
            obs = first(env.step(policy(obs)))
            if step % args.render_every == 0:
                target = env.base_pos[0].cpu().numpy()
                look = target if look is None else 0.9 * look + 0.1 * target  # カメラの揺れを抑える
                cam.set_pose(pos=look + CAM_OFFSET, lookat=look)
                rgb = first(cam.render(rgb=True))
                rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
                rgb = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
                writer.append_data(draw_overlay(rgb, seg, cmd, step * env.dt))
            step += 1
writer.close()
print(f"saved: {args.out}  ({step * env.dt:.1f} s, {fps} fps)")
