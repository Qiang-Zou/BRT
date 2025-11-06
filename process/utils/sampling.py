from OCC.Core.gp import gp_Pnt, gp_Vec
import numpy as np
from OCC.Core.Geom import Geom_BSplineCurve
from occwl.geometry import geom_utils
from occwl.geometry.box import Box

from utils.bezier2 import bernstein_polynomial_all_multi, getControlPointsFromApproximation,pointOnFace,normalOnFace
from OCC.Core.BRepTools import breptools

def uv_bounds(face):
    """
    Get the UV-domain bounds of this face's surface geometry

    Returns:
        Box: UV-domain bounds
    """
    umin, umax, vmin, vmax = breptools.UVBounds(face.topods_shape())
    bounds = Box(np.array([umin, vmin]))
    bounds.encompass_point(np.array([umax, vmax]))
    return bounds

def randn_uvgrid(face, num=100, uvs=True, method="point", given_uvs=None, bounds=None):
    """
    Creates a 2D UV-grid of samples from the given face
    """
    if given_uvs is not None:
        uv_values = given_uvs
    else:
        assert num >= 2
        uv_values = np.random.uniform(size=(num, 2)).astype(np.float32)

    if bounds is not None:
        umin, umax, vmin, vmax = bounds
        uv_box = Box(np.array([umin, vmin]))
        uv_box.encompass_point(np.array([umax, vmax]))
    else:
        uv_box = uv_bounds(face)


    if method=='point':
        fn=lambda uv:pointOnFace(face,uv)
    elif method=='normal':
        fn=lambda uv:normalOnFace(face,uv)
    else:
        fn = getattr(face, method)

    data = []

    for i in range(num):
        u = uv_box.intervals[0].interpolate(uv_values[i][0])
        v = uv_box.intervals[1].interpolate(uv_values[i][1])
        uv = np.array([u, v])
        val = fn(uv)
        data.append(val)
    data = np.asarray(data)

    if uvs:
        return data, uv_values

    return data

def _uvgrid_reverse_u(grid):
    reversed_grid = grid[::-1, :, :]
    return reversed_grid


def ugrid(curve:Geom_BSplineCurve,u_range, num_u: int = 10,us=False, method="point", reverse_order_with_edge=True):
    """ 
    Creates a 1D UV-grid of samples from the given edge
        edge (occwl.edge.Edge): A B-rep edge
        num_u (int): Number of samples along the curve. Defaults to 10/
        us (bool): Return the u values at which the quantity were evaluated
        method (str): Name of the method in the occwl.edge.Edge object to be called 
                      (the method has to accept the u value as argument). Defaults to "point".
    Returns:
        np.ndarray: 1D array of quantity evaluated on the edge geometry
        np.ndarray (optional): 1D array of u-values where evaluation was done
    """
    assert num_u >= 2
    ugrid = []

    u_values = np.linspace(u_range[0],u_range[1],num_u)

    if method=='point':
        fn = lambda u:geom_utils.gp_to_numpy(curve.Value(u))
    elif method=='tangent':
        fn = lambda u:tangent(curve,u)

    for u in u_values:
        val = fn(u)
        ugrid.append(val)

    ugrid = np.asarray(ugrid).reshape((num_u, -1))
    if us:
        return ugrid, u_values
    return ugrid

def tangent(curve:Geom_BSplineCurve, u):
    """
    Compute the tangent of the edge geometry at given parameter

    Args:
        u (float): Curve parameter

    Returns:
        np.ndarray: 3D unit vector
    """
    pt = gp_Pnt()
    der = gp_Vec()
    curve.D1(u, pt, der)
    der.Normalize()
    tangent = geom_utils.gp_to_numpy(der)
    return tangent