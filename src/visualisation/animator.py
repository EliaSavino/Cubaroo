'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from typing import Callable, Iterable, Optional

class CubeAnimator:
    """
    Animate face turns using temporary per-cubie transforms passed to plot_3d.
    """

    def __init__(
        self,
        cube,
        model,                                 # must expose .act(obs, epsilon=0, action_mask=None) -> int
        env,                                   # env providing .reset() -> obs, .step(action) -> (obs, reward, done, info)
        id2move: Callable[[int], str],         # maps model action index -> face letter / move token (e.g., 'U', 'R', "U'", "R2")
        frames_per_turn: int = 16,
        fps: int = 30,
    ):
        self.cube = cube
        self.model = model
        self.env = env
        self.id2move = id2move
        self.frames_per_turn = frames_per_turn
        self.fps = fps

        self.fig = plt.figure(figsize=(5.5,5.5))
        self.ax  = self.fig.add_subplot(111, projection='3d')

        # Rolling state for animation
        self.sequence: list[tuple[str, bool, int]] = []  # (face, clockwise, quarter_turns)  quarter_turns in {1,2,3}
        self.active_move: Optional[tuple[str, bool, int]] = None
        self.active_frame_idx: int = 0

        # Pre-cache cubie centers (updated at each discrete commit)
        self._update_cubie_centers()

    def _update_cubie_centers(self):
        centers = []
        for cubie in self.cube.cubies:
            # either use cubie.world_center() or compute from pose:
            R0, t0 = cubie.world_pose()
            centers.append(t0)  # assuming t0 is the center; adapt if needed
        self.cubie_centers = np.stack(centers, axis=0)

    def plan_with_model(self, max_steps: int = 60, epsilon: float = 0.0):
        """Run policy to build a move list (face, clockwise, k quarter-turns)."""
        obs = self.env.reset()
        self.sequence.clear()
        for _ in range(max_steps):
            a = self.model.act(obs, epsilon=epsilon, action_mask=getattr(self.env, "action_mask", None))
            move_token = self.id2move(a)  # e.g., 'U', "U'", 'U2'
            face, clockwise, k = self._parse_move(move_token)
            self.sequence.append((face, clockwise, k))
            obs, r, done, info = self.env.step(a)
            if done:
                break

    @staticmethod
    def _parse_move(tok: str) -> tuple[str, bool, int]:
        """
        Convert token to (face, clockwise, quarter_turns).
        Examples: 'U'->(U, True,1), "U'"->(U, False,1), 'U2'->(U, True,2).
        """
        face = tok[0]
        if tok.endswith("'"):
            return face, False, 1
        if tok.endswith("2"):
            return face, True, 2
        return face, True, 1

    def _intermediate_poses(self, face: str, clockwise: bool, theta: float):
        """
        Build pose_override for current tween angle theta (radians in [0, k*π/2]).
        """
        axis, sign, selector = face_spec(face)
        sign = sign if clockwise else -sign

        # Which cubies belong to the rotating layer?
        on_face = np.array([selector(c) for c in self.cubie_centers])

        # Rotation center = mean of rotating layer centers
        center = self.cubie_centers[on_face].mean(axis=0)

        R = R_axis_angle(axis, sign * theta)

        pose_override: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for idx, cubie in enumerate(self.cube.cubies):
            R0, t0 = cubie.world_pose()
            if on_face[idx]:
                # rotate about 'center': p' = center + R*(p - center)
                # For rigid body pose: R' = R * R0 ; t' = center + R*(t0 - center)
                Rn = R @ R0
                tn = center + R @ (t0 - center)
                pose_override[idx] = (Rn, tn)
            # else: leave as (R0, t0) by not overriding
        return pose_override

    def _commit_turn(self, face: str, clockwise: bool):
        """Apply the discrete state change to the cube after tween completes."""
        self.cube.rotate(face, clockwise=clockwise)
        self._update_cubie_centers()

    # ---------- Matplotlib animation glue ----------

    def _draw_frame(self, frame_idx: int):
        """Called by FuncAnimation for each global frame index."""
        self.ax.clear()

        if self.active_move is None and self.sequence:
            self.active_move = self.sequence.pop(0)
            self.active_frame_idx = 0

        if self.active_move is None:
            # nothing to do, just draw current cube
            self.cube.plot_3d(self.ax)
            return

        face, clockwise, k = self.active_move
        # progress in [0..k * 90°] over frames_per_turn * k frames
        total_frames = self.frames_per_turn * k
        alpha = min(1.0, self.active_frame_idx / total_frames)
        # ease-in-out for nicer look
        ease = 0.5 - 0.5*np.cos(np.pi * alpha)
        theta = ease * (k * np.pi/2)  # radians

        pose_override = self._intermediate_poses(face, clockwise, theta)
        self.cube.plot_3d(self.ax, pose_override=pose_override)

        self.active_frame_idx += 1
        if self.active_frame_idx > total_frames:
            # Finalize turn and move on
            self._commit_turn(face, clockwise)
            self.active_move = None
            self.active_frame_idx = 0

    def animate(self, outfile: str | None = None, duration_s: float | None = None):
        """
        If outfile is provided (.mp4 or .gif), save to disk. Otherwise show() live.
        If duration_s is given, cap total frames (useful when you have a long plan).
        """
        # Rough cap on frames if duration requested
        if duration_s is not None:
            max_frames = int(duration_s * self.fps)
        else:
            # upper bound: sum(frames per planned move)
            max_frames = sum(self.frames_per_turn * k for *_f, _cw, k in [self.active_move] + self.sequence if _f is not None) + 5*self.frames_per_turn

        anim = FuncAnimation(
            self.fig,
            self._draw_frame,
            frames=max_frames,
            interval=1000/self.fps,
            blit=False,
            repeat=False
        )

        if outfile:
            if outfile.endswith(".mp4"):
                try:
                    writer = FFMpegWriter(fps=self.fps, bitrate=4000)
                except Exception:
                    # fallback (still decent if Pillow is present)
                    writer = PillowWriter(fps=self.fps)
            elif outfile.endswith(".gif"):
                writer = PillowWriter(fps=self.fps)
            else:
                raise ValueError("outfile must end with .mp4 or .gif")
            anim.save(outfile, writer=writer)
            return outfile
        else:
            plt.show()
            return None