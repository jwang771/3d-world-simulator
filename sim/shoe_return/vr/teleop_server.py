"""Teleoperate the shoe-return cell from a Meta Quest 3 (or any WebXR headset) through its web browser.

The Isaac process runs the Mimic env and a small web server. The Quest browser opens the page, enters VR, and sees a
stereo video panel rendered by a camera pair placed where the operator stands (in front of the table, facing the
robots). Controller poses come back over a WebSocket and drive the two arms' TCP targets; every episode is recorded as
a Mimic source demo (successful ones are exported).

Controls (both controllers, mirrored):
    grip (side button) held   clutch: the arm follows the controller's motion, relative to where the grip was pressed
    trigger                   close the gripper (released: open)
    A / X                     toggle the open width: full open / pinch width
    B                         end the episode: save it if the task is done, then reset
    Y                         discard the episode and reset

Connect:
    same Wi-Fi:  https://<this machine's IP>:8443   (self-signed certificate: accept the warning once)
    USB cable:   adb reverse tcp:8080 tcp:8080, then http://localhost:8080 on the Quest

    cd ../IsaacLab && OMNI_KIT_ACCEPT_EULA=YES uv run python ../3d-world-simulator/sim/shoe_return/vr/teleop_server.py \
        --out runs/vr/demos.hdf5 --shoe SUPERSTAR_CLR
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "mimic"))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="Mimic dataset (.hdf5) for the successful episodes")
parser.add_argument("--shoe", default="SUPERSTAR_CLR")
parser.add_argument("--http-port", type=int, default=8080, help="plain HTTP (localhost over USB with adb reverse)")
parser.add_argument("--https-port", type=int, default=8443, help="HTTPS with a self-signed certificate (Wi-Fi)")
parser.add_argument("--scale", type=float, default=1.0, help="robot motion per metre of hand motion")
parser.add_argument("--width", type=int, default=640, help="per-eye stream width")
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--jpeg-quality", type=int, default=80)
parser.add_argument("--max-seconds", type=float, default=0, help="stop after this long (0: run until Ctrl-C); for tests")
add_launcher_args(parser)
args = parser.parse_args()

# Operator frame: WebXR is y-up with -z forward; the operator stands at the front of the table facing +y (as the front
# camera does), so xr (x, y, z) -> sim (x, -z, y).
XR_TO_SIM = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
EYE = (0.0, -1.10, 0.80)       # stereo camera midpoint (sim frame): standing operator's eyes, both arms in view
LOOK_AT = (0.0, 0.05, 0.12)
IPD = 0.064
GRIP_OPEN, GRIP_PINCH, GRIP_CLOSED = 2.0, 1.0, 0.0


def make_cert(d):
    import subprocess

    d.mkdir(parents=True, exist_ok=True)
    crt, key = d / "cert.pem", d / "key.pem"
    if not crt.exists():
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650", "-subj", "/CN=shoe-sim-vr",
                        "-keyout", str(key), "-out", str(crt)], check=True, capture_output=True)
    return crt, key


class Bridge:
    """State shared between the sim (main thread) and the web server (its own thread and asyncio loop)."""

    def __init__(self):
        import threading

        self.lock = threading.Lock()
        self.hands = {}        # latest controller state per handedness
        self.events = []       # button presses not yet handled by the sim
        self.frames = (0, None, None)  # (id, left jpeg, right jpeg)
        self.status = {}
        self.clients = 0

    def push_input(self, msg):
        with self.lock:
            self.hands = msg.get("hands", {})
            self.events += msg.get("events", [])

    def take_input(self):
        with self.lock:
            ev, self.events = self.events, []
            return dict(self.hands), ev


def serve(bridge, web_dir, cert_dir):
    import asyncio
    import json
    import ssl
    import threading

    from aiohttp import WSMsgType, web

    async def index(_):
        return web.FileResponse(web_dir / "index.html")

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=1 << 20)
        await ws.prepare(request)
        bridge.clients += 1
        print(f"[vr] client connected from {request.remote}", flush=True)

        async def sender():
            last = 0
            while not ws.closed:
                fid, left, right = bridge.frames
                if fid != last and left is not None:
                    last = fid
                    await ws.send_bytes(b"\x00" + left)
                    await ws.send_bytes(b"\x01" + right)
                    await ws.send_str(json.dumps(bridge.status))
                await asyncio.sleep(0.01)

        task = asyncio.ensure_future(sender())
        try:
            async for m in ws:
                if m.type == WSMsgType.TEXT:
                    bridge.push_input(json.loads(m.data))
        finally:
            task.cancel()
            bridge.clients -= 1
            print("[vr] client disconnected", flush=True)
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", web_dir)

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", args.http_port).start())
        crt, key = make_cert(cert_dir)
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(str(crt), str(key))
        loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", args.https_port, ssl_context=ctx).start())
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()


class Hand:
    """Clutch teleop for one arm: while the grip is held, the TCP target moves (and turns) with the controller."""

    def __init__(self, side, xr_to_sim, scale):
        import numpy as np

        self.side, self.M, self.scale = side, np.asarray(xr_to_sim, float), scale
        self.reset(None)

    def reset(self, T_tcp):
        self.T = T_tcp            # 4x4 TCP target (env frame)
        self.c0 = None            # controller pose (sim frame) when the grip was pressed
        self.T0 = None
        self.open_level = GRIP_OPEN
        self.grip = GRIP_OPEN
        self.prev = {}

    def ctrl_pose(self, h):
        import numpy as np

        from geometry import quat_to_mat

        R = quat_to_mat([h["q"][3], *h["q"][:3]])  # WebXR (x, y, z, w) -> (w, x, y, z)
        return self.M @ np.asarray(h["p"], float), self.M @ R @ self.M.T

    def update(self, h):
        """h: controller state from the page, or None when that controller is not tracked. Returns button events."""
        import numpy as np

        events = []
        if h is None:
            self.c0 = None
            return events
        names = dict(a="a", b="b") if self.side == "right" else dict(a="x", b="y")  # Quest: A/B right, X/Y left
        for b in ("a", "b"):
            if h.get(b) and not self.prev.get(b):
                events.append(names[b])
        self.prev = h
        if names["a"] in events:
            self.open_level = GRIP_PINCH if self.open_level == GRIP_OPEN else GRIP_OPEN
        self.grip = GRIP_CLOSED if h.get("trigger", 0) > 0.5 else self.open_level
        if h.get("squeeze", 0) > 0.5:
            p, R = self.ctrl_pose(h)
            if self.c0 is None:
                self.c0, self.T0 = (p, R), self.T.copy()
            T = self.T0.copy()
            T[:3, 3] = self.T0[:3, 3] + self.scale * (p - self.c0[0])
            T[:3, :3] = R @ self.c0[1].T @ self.T0[:3, :3]
            self.T = T
        else:
            self.c0 = None
        return events


def main():
    import isaaclab.sim as sim_utils

    args.require_kit, args.enable_cameras, args.physics = True, True, "isaacsim_physx"
    with launch_simulation(sim_utils.SimulationCfg(dt=1 / 120, device=args.device), args):
        import time

        import cv2
        import gymnasium as gym
        import numpy as np
        import torch

        import isaaclab.utils.math as PoseUtils
        import shoe_return_mimic_env as M
        from isaaclab.managers import DatasetExportMode
        from repack_scene import camera_cfg

        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        M.configure(shoe=args.shoe, cache=str(out.parent / "_mimic_assets"), cameras=True, width=args.width, height=args.height)
        M.register()
        cfg = M.ShoeReturnMimicEnvCfg()
        sc = M.S()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device or "cuda:0"
        # only the operator's stereo pair is rendered (the dataset cameras would slow the loop down)
        for cid in (0, 1, 3, 4):
            setattr(cfg.scene, f"camera_{cid}", None)
        eye, target = np.asarray(EYE), np.asarray(LOOK_AT)
        fwd = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(fwd, [0, 0, 1])
        right /= np.linalg.norm(right)
        for name, s in (("vr_eye_left", -0.5), ("vr_eye_right", 0.5)):
            e = eye + s * IPD * right
            setattr(cfg.scene, name, camera_cfg("{ENV_REGEX_NS}/" + name, sc.static_cam_pose(e, e + (target - eye)), sc.STATIC_FOCAL_MM,
                                                args.width, args.height))
            getattr(cfg.scene, name).data_types = ["rgb"]
        cfg.recorders = M.ShoeMimicRecorderManagerCfg()
        cfg.recorders.dataset_export_dir_path = str(out.parent)
        cfg.recorders.dataset_filename = out.stem
        cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        cfg.terminations.time_out = None
        cfg.terminations.success = None
        env = gym.make(M.TASK_ID, cfg=cfg).unwrapped

        bridge = Bridge()
        serve(bridge, HERE / "web", out.parent / "_vr_cert")
        print(f"[vr] serving: Wi-Fi https://<this machine's IP>:{args.https_port}  |  USB (adb reverse) http://localhost:{args.http_port}",
              flush=True)

        hands = {side: Hand(side, XR_TO_SIM, args.scale) for side in M.ARMS}

        def new_episode():
            env.recorder_manager.reset()
            env.reset()
            for side, h in hands.items():
                h.reset(M.tcp_pose(env, side)[0].cpu().numpy().astype(float))

        def action():
            a = []
            for side in M.ARMS:
                T = torch.as_tensor(hands[side].T, dtype=torch.float32)
                q = PoseUtils.quat_from_matrix(T[None, :3, :3])[0].tolist()
                a += [*T[:3, 3].tolist(), *q, hands[side].grip]
            return torch.tensor([a], dtype=torch.float32, device=env.device)

        def end_episode(save):
            ok = bool(M.task_success(env)[0])
            if save:
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes([0], torch.tensor([[ok]], dtype=torch.bool, device=env.device))
                env.recorder_manager.export_episodes([0])
            return ok

        def rgb(name):
            x = env.scene[name].data.output["rgb"]
            x = (x.torch if hasattr(x, "torch") else x)[0, ..., :3].cpu().numpy()
            return cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])[1].tobytes()

        new_episode()
        n_saved, episode, step, t_start, t_rate, n_rate, sps = 0, 0, 0, time.time(), time.time(), 0, 0.0
        while True:
            state, _ = bridge.take_input()
            events = []
            for side in M.ARMS:
                events += hands[side].update(state.get(side))
            if "b" in events or "y" in events:
                ok = end_episode(save="b" in events)
                n_saved += "b" in events and ok
                print(f"[vr] episode {episode} {'saved (success)' if 'b' in events and ok else 'not saved' if 'b' in events else 'discarded'};"
                      f" {n_saved} saved so far", flush=True)
                episode += 1
                new_episode()
                step = 0
            env.step(action())
            step += 1
            n_rate += 1
            if step % cfg.sim.render_interval == 0:
                now = time.time()
                sps, t_rate, n_rate = n_rate / max(now - t_rate, 1e-6), now, 0
                bridge.status = dict(episode=episode, saved=n_saved, step=step, sim_speed=round(sps / M.PHYSICS_HZ, 2),
                                     success=bool(M.task_success(env)[0]), lid_deg=round(float(M.hinge_deg(env)[0]), 1),
                                     clutch={s: hands[s].c0 is not None for s in M.ARMS},
                                     grip={s: hands[s].grip for s in M.ARMS},
                                     tcp={s: [round(float(v), 3) for v in M.tcp_pose(env, s)[0, :3, 3]] for s in M.ARMS})
                bridge.frames = (bridge.frames[0] + 1, rgb("vr_eye_left"), rgb("vr_eye_right"))
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                break
        env.close()


if __name__ == "__main__":
    main()
