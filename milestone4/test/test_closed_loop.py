"""
Closed-loop tests: all the driving nodes together on a simulated track.

fusion_node, gap_follow_node, safety_node and lap_counter_node run together (see
closed_loop.py). The car is "crashed" if any wall or box comes within the car's half-width
(0.15 m) of its centre line. These take a few seconds each.
"""

import unittest

import numpy as np

import closed_loop as cl  # noqa: I100  (imports ros_stubs first)

CAR_HALF_WIDTH = 0.15


class ClosedLoopTest(unittest.TestCase):
    """Laps, obstacles, a moving obstacle and a LiDAR dropout, end to end."""

    @classmethod
    def setUpClass(cls):
        cls.track, cls.lap = cl.stadium_track()
        cls.gate = 0.75 * cls.lap          # more than half a lap: the far straight looks alike

    def assert_no_crash(self, result):
        self.assertGreater(result['closest'], CAR_HALF_WIDTH)

    def test_laps_on_an_empty_track(self):
        result = cl.run(self.track, num_laps=3, min_lap_distance=self.gate, max_time=120)
        self.assert_no_crash(result)
        self.assertEqual(result['laps'], 3)
        self.assertEqual(result['mode'], 'stopped')            # latched stop, not AEB
        self.assertAlmostEqual(result['distance'] / 3, self.lap, delta=0.2 * self.lap)
        # the whole terminal output: one line per lap, then the safety node's stop line
        logs = result['logs']
        self.assertEqual(logs['fusion'], [])
        self.assertEqual(logs['planner'], [])
        self.assertEqual([m.split(' (')[0] for _, m in logs['laps']],
                         ['Lap 1/3', 'Lap 2/3', 'Lap 3/3'])
        self.assertTrue(logs['laps'][-1][1].endswith(': stopping'))
        self.assertEqual(logs['safety'], [('error', 'AEB engaged: stop requested (laps done)')])

    def test_variable_speed_is_faster_and_stays_inside_the_safety_band(self):
        # the old piecewise-linear speed averaged 0.77 m/s here with the same 1.5 m/s cap
        result = cl.run(self.track, num_laps=2, min_lap_distance=self.gate, max_speed=1.5,
                        max_time=120)
        self.assertEqual(result['laps'], 2)
        self.assertGreater(result['distance'] / result['time'], 1.2)
        modes = [mode for t, x, y, v, mode, laps in result['trace'] if mode != 'stopped']
        # the planner keeps plan_factor x the needed TTC, so safety rarely has to slow it
        self.assertLess(modes.count('slow') / len(modes), 0.05)
        self.assertNotIn('aeb', modes)

    def test_default_top_speed_is_held_around_the_track(self):
        # at the default 1.0 m/s cap the car keeps close to its top speed the whole lap
        result = cl.run(self.track, num_laps=2, min_lap_distance=self.gate, max_time=120)
        self.assertEqual(result['laps'], 2)
        self.assertEqual(result['mode'], 'stopped')
        self.assertGreater(result['distance'] / result['time'], 0.9)
        self.assertLessEqual(result['top_speed'], 1.0 + 1e-9)

    def test_laps_past_boxes_at_two_metres_per_second(self):
        world = np.vstack([self.track, cl.box(1.5, 0.25), cl.box(-1.0, 3.75)])
        result = cl.run(world, num_laps=2, min_lap_distance=self.gate, max_speed=2.0,
                        max_time=120)
        self.assert_no_crash(result)
        self.assertEqual(result['laps'], 2)
        self.assertEqual(result['mode'], 'stopped')
        self.assertGreater(result['top_speed'], 1.7)

    def test_box_driving_towards_the_car(self):
        ys = np.arange(-0.25, 0.25, 0.005)

        def world(t):
            # a box pushed towards the car at 1 m/s for 1.2 s (from t = 1 s), across the
            # whole lane except gaps too narrow for the car; then it stays where it is
            x = 3.0 - 1.0 * min(max(0.0, t - 1.0), 1.2)
            return np.vstack([self.track, np.stack([np.full_like(ys, x), ys], axis=1)])

        result = cl.run(world, num_laps=1, min_lap_distance=self.gate, max_time=5.0)
        self.assert_no_crash(result)
        self.assertEqual(result['mode'], 'aeb')                # latched, stopped before it

    def test_stops_safely_when_the_lane_is_blocked(self):
        # halfway round, two boxes side by side fill the middle of the lane and leave gaps
        # narrower than the car. The car must stop in front of them, with room to spare, and
        # stay stopped (latched) instead of squeezing through or driving into them.
        world = np.vstack([self.track, cl.blocking_boxes()])
        box_face = 0.2                                         # near face of the boxes (x)
        for kw in ({}, dict(cl.REALISTIC_CAR, max_speed=1.5)):
            result = cl.run(world, num_laps=3, min_lap_distance=self.gate, max_time=25, **kw)
            self.assertGreater(result['closest'], CAR_HALF_WIDTH + 0.1)   # 10 cm to spare
            self.assertEqual(result['mode'], 'aeb')            # latched stop, not the lap stop
            self.assertEqual(result['laps'], 0)
            self.assertGreater(result['distance'], 9.0)        # drove to the boxes first
            trace = result['trace']
            t, x, y, v, mode, laps = trace[-1]
            self.assertAlmostEqual(y, 4.0, delta=0.3)          # on the far straight
            self.assertGreater(x, box_face)                    # before the boxes (heading -x)
            stopped_at = min(t for t, x, y, v, mode, laps in trace if mode == 'aeb')
            held = [v for t, x, y, v, mode, laps in trace if t > stopped_at + 0.5]
            self.assertGreater(len(held), 100)                 # watched for over 2.5 s
            self.assertLess(max(held), 0.01)                   # and it stayed stopped

    def test_blocked_lane_with_weak_late_brakes(self):
        # the braking model assumes a_brake = 2.0 m/s^2 and no motor delay. The real car may
        # brake weaker and later: here at a third of that (0.7 m/s^2) and 0.2 s late. The
        # planner starts slowing well before the boxes (plan_factor), so the body (bumper
        # 0.15 m ahead of the LiDAR) must still keep 10 cm from walls and boxes, and stop.
        world = np.vstack([self.track, cl.blocking_boxes()])
        for speed in (1.0, 1.5):
            result = cl.run(world, num_laps=3, min_lap_distance=self.gate, max_time=25,
                            car_brake=0.7, drive_delay=0.2, lidar_to_bumper=0.15,
                            **dict(cl.REALISTIC_CAR, max_speed=speed))
            self.assertGreater(result['body_clearance'], 0.10)
            self.assertEqual(result['mode'], 'aeb')
            self.assertLess(result['trace'][-1][3], 0.01)     # stopped

    def test_obstacle_speed_tracking_prevents_a_crash(self):
        # on a long straight, a box is pushed at the moving car, then stops. Its own speed in
        # the variable TTC makes the car brake earlier; ignoring it leaves less room.
        from milestone4 import safety_node
        track, lap = cl.stadium_track(straight=14.0)
        ys = np.arange(-0.25, 0.25, 0.005)

        def gap(box_speed, seconds, tracking):
            def world(t):
                x = 6.5 - box_speed * min(max(0.0, t - 2.0), seconds)
                return np.vstack([track, np.stack([np.full_like(ys, x), ys], axis=1)])

            original = safety_node.SafetyNode.__init__

            def init(node):
                original(node)
                node.TRACK_OBSTACLES = tracking

            safety_node.SafetyNode.__init__ = init
            try:
                return cl.run(world, num_laps=1, min_lap_distance=0.75 * lap, max_speed=1.5,
                              max_time=2.0 + seconds + 2.5)
            finally:
                safety_node.SafetyNode.__init__ = original

        for box_speed, seconds in ((2.0, 1.4), (2.5, 1.2)):
            tracked = gap(box_speed, seconds, True)            # track_obstacles:=true
            ignored = gap(box_speed, seconds, False)           # static obstacles only
            self.assertGreater(tracked['closest'] - ignored['closest'], 0.05)
            self.assertEqual(tracked['mode'], 'aeb')
        self.assert_no_crash(tracked)                          # 2.5 m/s: avoided with tracking
        self.assertLess(ignored['closest'], CAR_HALF_WIDTH)    # and hit without it

    def test_servo_lag_and_flickering_box_labels(self):
        # the car-test failure: a lagging steering servo (odometry yaw from the command) and
        # box labels that come and go. The car must still lap, with no AEB and few toggles.
        centres = [(1.5, 0.25), (-1.0, 3.75)]
        world = np.vstack([self.track] + [cl.box(*c) for c in centres])
        for probability in (0.3, 0.6):
            result = cl.run(world, num_laps=2, min_lap_distance=self.gate, max_time=90,
                            servo_lag=0.12, box_centres=centres,
                            label_probability=probability, track_obstacles=True)
            self.assert_no_crash(result)
            self.assertEqual(result['laps'], 2)
            self.assertEqual(result['mode'], 'stopped')        # the lap stop, not AEB
            self.assertLess(result['passing_starts'], 20)      # was over 100 without debounce

    def test_depth_camera_avoids_what_the_lidar_cannot_see(self):
        # a low box below the LiDAR's scan plane, 2 m ahead on the left half of the lane:
        # only the depth camera sees it, through fusion_node
        low = cl.box(2.0, 0.3)
        fused = cl.run(self.track, num_laps=1, min_lap_distance=self.gate, max_time=6.0,
                       low_obstacles=low, use_camera=True)
        blind = cl.run(self.track, num_laps=1, min_lap_distance=self.gate, max_time=6.0,
                       low_obstacles=low, use_camera=False)
        self.assert_no_crash(fused)
        self.assertGreater(fused['distance'], 4.0)             # it went around, not stopped
        self.assertLess(blind['closest'], CAR_HALF_WIDTH)      # LiDAR alone drives into it

    def test_scan_lag_spike_does_not_stop_the_car(self):
        # the second car test: when the camera starts, the Jetson falls behind and scans
        # reach the safety node up to 0.4 s late, then in a burst; a static box is ahead
        track, lap = cl.stadium_track(straight=14.0)
        world = np.vstack([track, cl.box(3.4, 0.35)])

        def lag(t):
            return 0.4 * (t - 1.0) / 0.3 + 0.02 if 1.0 <= t < 1.3 else 0.02

        result = cl.run(world, num_laps=1, min_lap_distance=0.75 * lap, max_time=8.0,
                        scan_lag=lag, track_obstacles=True)
        self.assert_no_crash(result)
        self.assertNotIn('aeb', [mode for t, x, y, v, mode, laps in result['trace']])
        self.assertGreater(result['distance'], 6.0)            # drove on past the box
        # nothing moves here; pairing late scans with the pose on arrival read ~2.4 m/s
        self.assertLess(result['obstacle_peak'], 0.3)

    def test_realistic_car_laps_narrow_tracks(self):
        # the car as it really is (servo lag, steering 80% of the command plus an offset, the
        # LiDAR ahead of the rear axle, range noise) on 1.0 m and 1.2 m wide tracks
        for half_lane in (0.5, 0.6):
            track, lap = cl.stadium_track(half_lane=half_lane)
            result = cl.run(track, num_laps=1, min_lap_distance=0.75 * lap, max_time=40,
                            **cl.REALISTIC_CAR)
            self.assert_no_crash(result)
            self.assertEqual(result['laps'], 1)
            self.assertEqual(result['mode'], 'stopped')        # the lap stop, not AEB
            # with the old 0.12 m margin (and pure pursuit from the LiDAR, as then) it cuts the
            # corners: the body passes at least 5 cm nearer the inner wall (AEB stopped it
            # on the car, and still does here on the 1.2 m track)
            old = cl.run(track, num_laps=1, min_lap_distance=0.75 * lap, max_time=40,
                         margin=0.12, planner={'lidar_x': 0.0}, **cl.REALISTIC_CAR)
            self.assertLess(old['body_clearance'], result['body_clearance'] - 0.05)

    def test_steering_mismatch_with_tracking_on(self):
        # the third car test: the car steers 70% of what it is told. With obstacle tracking
        # on, barriers seen at a shallow angle read 1.9 m/s without the facing-surface gate
        mismatch = cl.run(self.track, num_laps=1, min_lap_distance=self.gate, max_time=40,
                          track_obstacles=True, steer_gain=0.7)
        self.assertEqual(mismatch['laps'], 1)
        self.assertLess(mismatch['obstacle_peak'], 0.3)
        # the whole realistic car still laps with tracking on. Its 0.1 s steering delay leaves
        # some false speed (2.1 m/s without the gate), one reason tracking is off by default
        car = dict(cl.REALISTIC_CAR, steer_gain=0.7)
        result = cl.run(self.track, num_laps=1, min_lap_distance=self.gate, max_time=40,
                        track_obstacles=True, **car)
        self.assert_no_crash(result)
        self.assertEqual(result['laps'], 1)
        self.assertEqual(result['mode'], 'stopped')
        self.assertLess(result['obstacle_peak'], 1.0)

    def test_realistic_car_passes_boxes_with_calm_steering(self):
        # the fourth car test: the steering twitched. With the real car's steering delay and
        # noisy odometry, smoothing 0.6 laps cleanly and steers calmer than the old 0.3.
        # (0.3 used to run into AEB at the first box. So did 0.6 once the simulated LiDAR
        # saw the box's side face: the planner kept its clearance from the far end of that
        # face, not the near corner, and corner_check cancelled the turn at full speed.)
        world = np.vstack([self.track, cl.box(1.5, 0.25), cl.box(-1.0, 3.75)])
        calm = cl.run(world, num_laps=2, min_lap_distance=self.gate, max_time=120,
                      **cl.REALISTIC_CAR)
        self.assert_no_crash(calm)
        self.assertEqual(calm['laps'], 2)
        self.assertEqual(calm['mode'], 'stopped')
        self.assertGreater(calm['body_clearance'], 0.10)       # the box's near corner
        twitchy = cl.run(world, num_laps=2, min_lap_distance=self.gate, max_time=120,
                         smoothing=0.3, **cl.REALISTIC_CAR)
        self.assertLess(calm['steer_jitter'], 0.8 * twitchy['steer_jitter'])

    def test_lidar_dropout_stops_then_resumes(self):
        result = cl.run(self.track, num_laps=1, min_lap_distance=self.gate, max_time=6.5,
                        lidar_dropout=(2.0, 3.5))
        self.assert_no_crash(result)
        trace = result['trace']
        stale_at = min(t for t, x, y, v, mode, laps in trace if mode == 'stale')
        stopped = [v for t, x, y, v, mode, laps in trace if 3.4 <= t < 3.5]
        after = [v for t, x, y, v, mode, laps in trace if t > 5.0]
        self.assertLess(stale_at - 2.0, 0.35)                  # watchdog: 0.25 s + one period
        self.assertLess(max(stopped), 0.05)                    # braked to a stop
        self.assertGreater(min(after), 0.3)                    # and drove on once scans came back


if __name__ == '__main__':
    unittest.main()
