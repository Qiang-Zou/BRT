import math
import unittest
from unittest.mock import Mock, patch

import numpy as np
from OCC.Core.Geom2d import Geom2d_Circle
from OCC.Core.gp import gp_Ax2d, gp_Dir2d, gp_Pnt2d

from triangles3 import (
    Intersection, Rectangle, Triangle, boundary_interval, chordErrorCheckInRect,
    pointOnLine, splitBoundaryRectangle,
)
from utils import bezier2


class BoundaryGeometryTests(unittest.TestCase):
    def test_finite_segment(self):
        a, b = np.array([0., 0.]), np.array([1., 0.])
        for x, y, expected in [
            (-2., 0., False), (-.001, 0., True), (.5, 0., True),
            (.5, .02, False), (1.001, 0., True), (5., 0., False),
        ]:
            with self.subTest(x=x, y=y):
                self.assertEqual(pointOnLine(a, b, np.array([x, y])), expected)
        self.assertFalse(pointOnLine(a, a, b))

    def test_arc_across_seam_keeps_endpoint_order(self):
        circle = Geom2d_Circle(gp_Ax2d(gp_Pnt2d(0, 0), gp_Dir2d(1, 0)), 1.)
        rect = Rectangle()
        height = math.sin(.1)
        rect.points = [[.9, -height], [1.1, -height], [.9, height], [1.1, height]]
        period = 2 * math.pi
        for params in ([.1, period - .1], [period - .1, .1]):
            points = [np.array(circle.Value(u).Coord()) for u in params]
            intersection = Intersection(2, points, list(params), circle, (0, period))
            for _ in range(2):
                self.assertTrue(chordErrorCheckInRect(intersection, rect, tol=.98))
                self.assertEqual(intersection.Parameters, params)
            start, end = boundary_interval(intersection, rect)
            self.assertAlmostEqual(abs(end - start), .2)
            np.testing.assert_allclose(circle.Value(start).Coord(), points[0])
            np.testing.assert_allclose(circle.Value(end).Coord(), points[1])
            for u in np.linspace(start, end, 11):
                self.assertTrue(rect.contains(np.array(circle.Value(u).Coord())))
            cell = Rectangle()
            cell.points = rect.points
            cell.leaf_info = [intersection]
            face = Mock()
            face.visibility_status.return_value = 0
            leaf = Triangle(points[0], cell.points[0], points[1])
            with patch("triangles3.make_boundary_rect", return_value=[leaf]) as cut:
                splitBoundaryRectangle(face, cell, None, None)
            cut.assert_called_once()
            start, end = cut.call_args.kwargs["interval"]
            endpoints = cut.call_args.kwargs["end_points"]
            self.assertAlmostEqual(abs(end - start), .2)
            np.testing.assert_allclose(circle.Value(start).Coord(), endpoints[0])
            np.testing.assert_allclose(circle.Value(end).Coord(), endpoints[-1])
            self.assertEqual(intersection.Parameters, params)

    def test_custom_grid_uses_its_own_basis(self):
        default = bezier2.FIT_UVS
        custom = default[::-1].copy()
        cache = []
        for uv in (default, custom, custom[:-8], default):
            u, v = uv.T
            points = np.column_stack((u, v, u**2 + v**2))
            controls = bezier2.fit_bezier_surface2(points, uv, bn_cache=cache)
            basis = bezier2.bernstein_polynomial_all_multi(6, u, v)
            np.testing.assert_allclose(basis @ controls[:, :3], points, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
