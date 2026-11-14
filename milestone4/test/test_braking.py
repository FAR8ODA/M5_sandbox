"""Tests for the shared braking model: variable TTC, speed caps and the path corridor."""

import math
import unittest

import numpy as np
import ros_stubs

from milestone4.braking import (car_motion, path_distances, required_ttc,  # noqa: I100
                                        safe_speed, slow_speed_cap, stopping_distance)

T_REACT, A_BRAKE, TTC_MARGIN = 0.15, 2.0, 0.1         # the node defaults


def old_min_ttc(ranges, angles, speed):
    """Milestone 3 TTC: every beam ahead, range / (speed * cos(angle))."""
    closing = speed * np.cos(angles)
    ok = np.isfinite(ranges) & (closing > 0.0)
    return float(np.min(ranges[ok] / closing[ok]))


def needed(speed, obstacle=0.0):
    """Return the required TTC with the default settings."""
    return float(required_ttc(speed, obstacle, T_REACT, A_BRAKE, TTC_MARGIN))


class VariableTtcTest(unittest.TestCase):
    """The AEB threshold is the time the car needs to stop, so it changes with speed."""

    def test_threshold_values(self):
        # the table in the README
        for speed, expected in ((0.5, 0.375), (1.0, 0.5), (1.5, 0.625), (2.0, 0.75)):
            self.assertAlmostEqual(needed(speed), expected)

    def test_threshold_rises_with_speed(self):
        values = [needed(v) for v in np.linspace(0.0, 3.0, 13)]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))

    def test_threshold_rises_when_the_obstacle_comes_towards_us(self):
        self.assertGreater(needed(1.0, 1.0), needed(1.0, 0.0))
        self.assertGreater(needed(1.0, 3.0), needed(1.0, 1.0))
        # a stopped car only needs to react
        self.assertAlmostEqual(needed(0.0, 2.0), T_REACT + TTC_MARGIN)

    def test_threshold_is_exactly_enough_to_stop(self):
        # at ttc = required TTC, the stopping distance plus the obstacle's travel uses up
        # the whole gap, with ttc_margin x closing speed to spare
        for v, e in ((1.0, 0.0), (1.5, 1.0), (2.0, 2.5)):
            gap = (v + e) * needed(v, e)
            used = (stopping_distance(v, T_REACT, A_BRAKE, 0.0)
                    + e * (T_REACT + v / A_BRAKE))
            self.assertAlmostEqual(gap - used, TTC_MARGIN * (v + e))

    def test_works_on_arrays(self):
        out = required_ttc(1.0, np.array([0.0, 1.0, 2.0]), T_REACT, A_BRAKE, TTC_MARGIN)
        self.assertEqual(out.shape, (3,))

    def test_slow_cap_keeps_the_ttc_at_the_factor(self):
        # plugging the cap back in: ttc at that speed is exactly factor x the required TTC
        for gap in (0.5, 1.0, 2.0, 4.0):
            for factor in (1.0, 2.0, 2.5):
                v = float(slow_speed_cap(gap, 0.0, factor, T_REACT, A_BRAKE, TTC_MARGIN))
                self.assertAlmostEqual(gap / v, factor * needed(v), places=9)

    def test_slow_cap_is_zero_without_a_gap(self):
        self.assertEqual(float(slow_speed_cap(-0.1, 0.0, 2.0, T_REACT, A_BRAKE, TTC_MARGIN)),
                         0.0)

    def test_slow_cap_drops_for_an_approaching_obstacle(self):
        still = float(slow_speed_cap(2.0, 0.0, 2.0, T_REACT, A_BRAKE, TTC_MARGIN))
        coming = float(slow_speed_cap(2.0, 0.5, 2.0, T_REACT, A_BRAKE, TTC_MARGIN))
        self.assertAlmostEqual(still - coming, 0.5)


class BrakingModelTest(unittest.TestCase):
    """Stopping distance, safe speed, car motion and the path corridor."""

    def test_stopping_distance_grows_with_speed_squared(self):
        braking = [stopping_distance(v, 0.0, 2.0, 0.0) for v in (1.0, 2.0)]
        self.assertAlmostEqual(braking[1] / braking[0], 4.0)

    def test_safe_speed_inverts_stopping_distance(self):
        for v in (0.5, 1.0, 2.0):
            d = stopping_distance(v, 0.15, 2.0, 0.0)
            self.assertAlmostEqual(safe_speed(d, 0.15, 2.0), v)
        self.assertEqual(safe_speed(-0.1, 0.15, 2.0), 0.0)

    def test_side_walls_are_off_a_straight_path(self):
        # a narrow straight (walls 0.5 m each side); Milestone 3 TTC would brake at 2 m/s
        a = ros_stubs.lidar_angles()
        r = ros_stubs.corridor_ranges(0.5)
        self.assertLess(old_min_ttc(r, a, 2.0), 0.65)
        index, along = path_distances(r, a, 0.0, 0.33, 0.2)
        self.assertGreater(along.min(), 9.0)                  # only the far end is on the path

    def test_turning_path_ignores_a_wall_straight_ahead(self):
        a = ros_stubs.lidar_angles()
        r = ros_stubs.add_box(np.full(a.size, np.inf), 1.2, -1.0, 0.0)   # ahead and to the right
        index, _ = path_distances(r, a, 0.4, 0.33, 0.2)       # full left lock
        self.assertEqual(index.size, 0)
        index, along = path_distances(r, a, 0.0, 0.33, 0.2)   # straight: it is on the path
        self.assertAlmostEqual(float(along.min()), 1.2, places=2)

    def test_car_motion_on_an_arc(self):
        ahead, left, turn = car_motion(1.0, 1.0, math.pi / 2)    # quarter circle, radius 1
        self.assertAlmostEqual(ahead, 1.0)
        self.assertAlmostEqual(left, 1.0)
        self.assertAlmostEqual(turn, math.pi / 2)
        self.assertEqual(car_motion(2.0, 0.0, 0.5), (1.0, 0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
