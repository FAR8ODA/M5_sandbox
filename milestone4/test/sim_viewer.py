"""
Watch the Milestone 4 nodes drive the closed-loop simulator (no ROS needed).

Runs the real fusion_node, gap_follow_node, safety_node and lap_counter_node from
this package through closed_loop.py (in this folder), records what the planner saw
on every scan, then animates it:

    grey dots   track walls (and boxes)
    blue dots   the LiDAR scan this step
    orange dots beams the disparity extender shortened (where it thinks the car can't go)
    red line    the target beam the planner steers for
    green       the car (rear axle at the back edge, LiDAR at the black dot), path trail

Setup (once):   pip install matplotlib numpy
Examples (run from milestone4/test):
    python sim_viewer.py                                # current code, ideal car, 1.5 m/s
    python sim_viewer.py --realistic --speed 1.5        # car with steering lag/error, LiDAR offset
    python sim_viewer.py --realistic --deep 0.7         # aim further from the inner wall
    python sim_viewer.py --boxes --realistic --speed 1.5
    python sim_viewer.py --blocked --realistic --speed 1.5   # lane blocked halfway: must stop
    python sim_viewer.py --many-boxes --realistic --speed 1.5  # slalom of 5 boxes on the straights
The nodes' log messages print in the terminal as the sim runs (--quiet to hide them).
Keys: space = pause/resume, left/right = step when paused, up/down = playback speed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

USED = {}                        # planner values actually used, for the title
# closed_loop.py lives beside this script in milestone4/test
TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def simulate(args) -> tuple[np.ndarray, dict, dict, float]:
    """
    Run the closed-loop sim with the planner instrumented for drawing.

    Args:
        args (argparse.Namespace): Parsed command-line options (see main).

    Returns:
        tuple: (world, result, records, offset): world is an Nx2 array of wall and box
        points, result is closed_loop.run's dict (its trace shifted to track coordinates),
        records maps sim time (ms) to what the planner saw, and offset is the LiDAR's
        distance ahead of the rear axle (m).
    """
    test_dir = os.path.abspath(args.code)
    if not os.path.isfile(os.path.join(test_dir, 'closed_loop.py')):
        sys.exit(f'{test_dir} has no closed_loop.py: run from milestone4/test, '
                 f'or pass --code <a copy of it>/milestone4/test')
    sys.path.insert(0, test_dir)
    import closed_loop as cl
    if not args.quiet:
        print_node_logs(cl.ros_stubs)

    records = {}                     # sim time (ms) -> what the planner saw
    base = cl.GapFollowNode

    class RecordingPlanner(base):
        """The real planner, with its intermediate results recorded for drawing."""

        def __init__(self) -> None:
            """Build the real planner, applying --deep and noting the values used."""
            super().__init__()
            if args.deep is not None:
                self.DEEP_FRACTION = args.deep
            USED['deep'] = self.DEEP_FRACTION
            USED['margin'] = self.MARGIN

        def drive_on(self, msg) -> None:
            """Start a record for this scan, run the real planner, then store the record."""
            self._rec = {'scan': np.asarray(msg.ranges, float),
                         'amin': msg.angle_min, 'inc': msg.angle_increment}
            super().drive_on(msg)
            records[int(round(cl.ros_stubs.CLOCK.seconds * 1000))] = self._rec

        def extend_disparities(self, ranges, boxes, inc) -> np.ndarray:
            """Run the real disparity extender and record its input and output."""
            out = super().extend_disparities(ranges, boxes, inc)
            self._rec['clean'], self._rec['ext'] = ranges.copy(), out.copy()
            return out

        def find_target(self, extended, angles) -> int | None:
            """Run the real target search and record the chosen beam (angle, range)."""
            i = super().find_target(extended, angles)
            if i is not None:
                self._rec['target'] = (float(angles[i]), float(extended[i]))
            return i

        def corner_check(self, steering, beside) -> float:
            """Run the real corner check and record the final steering angle."""
            out = super().corner_check(steering, beside)
            self._rec['steer'] = out
            return out

    # closed_loop builds the planner by this name, so it now builds the recording one
    cl.GapFollowNode = RecordingPlanner

    track, lap = cl.stadium_track(straight=args.straight, radius=args.radius,
                                  half_lane=args.lane / 2.0)
    world = track
    if args.boxes:
        world = np.vstack([track, cl.box(1.5, 0.25), cl.box(-1.0, 2 * args.radius - 0.25)])
    centres = None
    if args.many_boxes:
        centres = many_box_centres(args.straight, args.radius, args.lane)
        world = np.vstack([world] + [cl.box(cx, cy) for cx, cy in centres])
    if args.blocked:
        world = np.vstack([world, cl.blocking_boxes(args.radius)])
    kw = dict(cl.REALISTIC_CAR) if args.realistic else {}
    if args.margin is not None:
        kw['margin'] = args.margin
    # blocked: no lap ever finishes, so stop the sim a few seconds after the car has stopped
    max_time = min(args.time, 25.0) if args.blocked else args.time
    if centres is not None:
        # camera labels the boxes too, like the real car (in the sim's car-start frame)
        kw['box_centres'] = [(cx - args.start_x, cy) for cx, cy in centres]
    # the sim always starts the car at its origin heading +x; shift the world so that origin
    # is the real start spot on the near straight, then shift the trace back to track coords
    start = np.array([args.start_x, 0.0])
    result = cl.run(world - start, num_laps=args.laps, max_speed=args.speed,
                    min_lap_distance=0.75 * lap, max_time=max_time, **kw)
    result['trace'] = [(t, x + start[0], y + start[1], v, m, n)
                       for t, x, y, v, m, n in result['trace']]
    offset = kw.get('lidar_offset', 0.0)
    return world, result, records, offset


def print_node_logs(stubs) -> None:
    """
    Print every node's log messages as they happen, like ROS does in the terminal.

    The test stub Logger only stores (level, message) in .records for the unit tests. Each
    node's records list is swapped for one that also prints, with the sim time and node
    name. (Wrapping Logger._log instead would break its per-call-site severity check.)
    """
    class PrintingRecords(list):
        """A node's log record list that also prints each non-debug message."""

        def __init__(self, name) -> None:
            """Create an empty list for the node called name."""
            super().__init__()
            self.name = name

        def append(self, record) -> None:
            """Print a (level, message) record with the sim time, then store it."""
            level, message = record
            if level != 'debug':
                print(f'[{stubs.CLOCK.seconds:7.2f}s] {self.name:16s} '
                      f'{level.upper():5s} {message}')
            super().append(record)

    init = stubs.Node.__init__

    def node_init(self, node_name, **kwargs) -> None:
        """Run the stub Node constructor, then give the node a printing record list."""
        init(self, node_name, **kwargs)
        self._logger.records = PrintingRecords(node_name)

    stubs.Node.__init__ = node_init


def many_box_centres(straight, radius, lane) -> list[tuple[float, float]]:
    """
    Centres of 5 boxes (0.4 m) on the two straights (none on the turns), alternating sides.

    Each box sits 5 cm off a wall, leaving a gap of lane - 0.45 m (0.75 m on a 1.2 m lane)
    beside it: enough for the 0.3 m car plus the planner's 0.2 m margin each side, so the lap
    is always passable but the car has to weave. Boxes on a straight are 0.4 * straight apart
    (2.4 m on a 6 m straight). The car starts near (-2, 0) heading +x, so the nearest box is
    2 m ahead of the start.
    """
    off = lane / 2.0 - 0.25           # box centre from the centreline
    far = 2.0 * radius
    d = 0.4 * straight
    return [
        # near straight (y = 0, car heading +x): left, then right
        (0.0, +off), (d, -off),
        # far straight (y = 2R, car heading -x): right, left, right
        (d, far + off), (0.0, far - off), (-d, far + off),
    ]


def side_gaps(world, x, y, yaw) -> tuple[float, float]:
    """Distance from the car's centre line to the nearest wall on its left and right (m)."""
    dx, dy = world[:, 0] - x, world[:, 1] - y
    ahead = dx * math.cos(yaw) + dy * math.sin(yaw)
    left = -dx * math.sin(yaw) + dy * math.cos(yaw)
    beside = np.abs(ahead - 0.15) < 0.3
    lefts, rights = left[beside & (left > 0)], -left[beside & (left < 0)]
    return (lefts.min() if lefts.size else math.inf), (rights.min() if rights.size else math.inf)


def main() -> None:
    """Parse the options, run the sim, print a summary, then animate it (unless --check)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--code', default=TEST_DIR,
                   help='milestone4/test folder to run (default: the folder of this script)')
    p.add_argument('--speed', type=float, default=1.5, help='max_speed (m/s)')
    p.add_argument('--realistic', action='store_true',
                   help='steering lag/delay, steering 0.8x command, LiDAR 0.27 m ahead, noise')
    p.add_argument('--deep', type=float, default=None,
                   help='deep_fraction (default: the value in the code, 0.95; '
                        'try 0.7 to aim further from inner walls)')
    p.add_argument('--margin', type=float, default=None, help='margin (code default 0.20)')
    p.add_argument('--boxes', action='store_true', help='add two boxes to the track')
    p.add_argument('--many-boxes', action='store_true',
                   help='slalom of 5 boxes on the straights, alternating sides (always passable)')
    p.add_argument('--blocked', action='store_true',
                   help='two boxes side by side across the lane halfway round: the car must stop')
    p.add_argument('--start-x', type=float, default=-2.0,
                   help='where the car starts on the near straight (track x, m; y is 0)')
    p.add_argument('--lane', type=float, default=1.2, help='lane width (m)')
    p.add_argument('--radius', type=float, default=2.0, help='corner radius of the centreline (m)')
    p.add_argument('--straight', type=float, default=6.0, help='straight length (m)')
    p.add_argument('--laps', type=int, default=3, help='num_laps (same as the launch default)')
    p.add_argument('--time', type=float, default=120.0, help='max sim time (s)')
    p.add_argument('--quiet', action='store_true', help="don't print the nodes' log messages")
    p.add_argument('--check', action='store_true', help='simulate and print only (no window)')
    args = p.parse_args()

    print('simulating...')
    world, result, records, offset = simulate(args)
    trace = result['trace']
    print(f"{result['laps']} laps in {result['time']:.1f} s, final mode {result['mode']}, "
          f"closest wall {result['closest']:.2f} m, top speed {result['top_speed']:.2f} m/s")
    t, x, y = trace[-1][:3]
    print(f"car ended at ({x:.2f}, {y:.2f}) at t = {t:.1f} s")
    if args.check:
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon

    # heading from consecutive positions (the trace has no yaw)
    xs = np.array([s[1] for s in trace])
    ys = np.array([s[2] for s in trace])
    yaws = np.zeros(len(trace))
    for k in range(1, len(trace)):
        dx, dy = xs[k] - xs[k - 1], ys[k] - ys[k - 1]
        yaws[k] = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-5 else yaws[k - 1]

    # track on the left, info in its own panel on the right (never covers the track)
    fig, (ax, panel) = plt.subplots(1, 2, figsize=(13, 6),
                                    gridspec_kw={'width_ratios': [3, 0.8], 'wspace': 0.04})
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.07)
    panel.axis('off')
    ax.set_aspect('equal')
    ax.plot(world[:, 0], world[:, 1], '.', color='0.6', ms=1)
    trail, = ax.plot([], [], '-', color='tab:green', lw=1, alpha=0.6)
    scan_pts, = ax.plot([], [], '.', color='tab:blue', ms=2)
    ext_pts, = ax.plot([], [], '.', color='tab:orange', ms=3)
    target_ln, = ax.plot([], [], '-', color='tab:red', lw=1.5)
    lidar_dot, = ax.plot([], [], 'k.', ms=6)
    car = Polygon(np.zeros((4, 2)), closed=True, fc='tab:green', ec='k', alpha=0.8)
    ax.add_patch(car)
    info = panel.text(0.0, 0.98, '', transform=panel.transAxes, va='top', family='monospace',
                      fontsize=9)
    panel.text(0.0, 0.02,
               'grey    walls (and boxes)\nblue    LiDAR scan\n'
               'orange  disparity-extended beams\nred     target beam\n'
               'green   car, black dot = LiDAR\n\n'
               'space pause  </> step  ^/v speed',
               transform=panel.transAxes, va='bottom', family='monospace', fontsize=8,
               color='0.35')
    title = (f"speed {args.speed}  realistic={args.realistic}  deep={USED['deep']:.2f}  "
             f"margin={USED['margin']:.2f}  lane={args.lane} m  "
             f"boxes={'5' if args.many_boxes else args.boxes}  "
             f"blocked={args.blocked}")
    ax.set_title(title)

    state = {'k': 0, 'paused': False, 'skip': 2}

    def draw(k) -> tuple:
        """Draw trace step k (car, trail, scan, extended beams, target, info panel)."""
        t, x, y, v, mode, laps = trace[k]
        yaw = yaws[k]
        c, s = math.cos(yaw), math.sin(yaw)
        # car body: rear axle at x,y; 0.5 m long (0.05 behind axle), 0.3 m wide
        body = np.array([[-0.05, -0.15], [0.45, -0.15], [0.45, 0.15], [-0.05, 0.15]])
        car.set_xy(np.c_[x + body[:, 0] * c - body[:, 1] * s, y + body[:, 0] * s + body[:, 1] * c])
        lx, ly = x + offset * c, y + offset * s
        lidar_dot.set_data([lx], [ly])
        lo = max(0, k - 400)
        trail.set_data(xs[lo:k + 1], ys[lo:k + 1])

        rec = records.get(int(round(t * 1000)))
        steer_txt = ''
        if rec is not None:
            r = rec['scan']
            a = rec['amin'] + np.arange(r.size) * rec['inc']
            ok = np.isfinite(r)
            scan_pts.set_data(lx + r[ok] * np.cos(a[ok] + yaw), ly + r[ok] * np.sin(a[ok] + yaw))
            if 'ext' in rec:
                cut = (rec['ext'] < rec['clean'] - 1e-6) & (rec['ext'] > 0)
                e = rec['ext'][cut]
                ext_pts.set_data(lx + e * np.cos(a[cut] + yaw), ly + e * np.sin(a[cut] + yaw))
            if 'target' in rec:
                ta, tr = rec['target']
                target_ln.set_data([lx, lx + tr * math.cos(ta + yaw)],
                                   [ly, ly + tr * math.sin(ta + yaw)])
            if 'steer' in rec:
                steer_txt = f"steer   {math.degrees(rec['steer']):+6.1f} deg\n"
        gl, gr = side_gaps(world, x, y, yaw)
        info.set_text(f"t       {t:6.2f} s\nspeed   {v:6.2f} m/s\n{steer_txt}"
                      f"car centre to wall:\n"
                      f"  left  {gl:6.2f} m\n  right {gr:6.2f} m\n"
                      f"  (car side is 0.15 m less)\n"
                      f"mode    {mode}\nlaps    {laps}")
        return car, trail, scan_pts, ext_pts, target_ln, lidar_dot, info

    def update(_) -> tuple:
        """Advance playback by the current skip (unless paused) and redraw."""
        if not state['paused']:
            state['k'] = min(state['k'] + state['skip'], len(trace) - 1)
        return draw(state['k'])

    def on_key(event) -> None:
        """Handle the playback keys: pause, step and speed."""
        if event.key == ' ':
            state['paused'] = not state['paused']
        elif event.key == 'right':
            state['k'] = min(state['k'] + 1, len(trace) - 1)
        elif event.key == 'left':
            state['k'] = max(state['k'] - 1, 0)
        elif event.key == 'up':
            state['skip'] = min(state['skip'] * 2, 32)
        elif event.key == 'down':
            state['skip'] = max(state['skip'] // 2, 1)

    fig.canvas.mpl_connect('key_press_event', on_key)
    ax.set_xlim(world[:, 0].min() - 0.5, world[:, 0].max() + 0.5)
    ax.set_ylim(world[:, 1].min() - 0.5, world[:, 1].max() + 0.5)
    # keep a reference, or the animation is garbage-collected and never runs
    _anim = FuncAnimation(fig, update, interval=25, blit=False,  # noqa: F841
                          cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    main()
