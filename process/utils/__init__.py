import math
from .sampling import randn_uvgrid
import numpy as np

def chordErrorCheck(t0,t1,t2,eval_fn,tol=10/math.sqrt(101)):
    return chordError(t0,t1,t2,eval_fn)>tol
def chordError(t0,t1,t2,eval_fn):
    dist=eval_fn
    return dist(t0,t2)/(dist(t0,t1)+dist(t1,t2))


class PlainFaceError(Exception):
    pass
def are_points_on_same_plane(points):
    if len(points) != 5:
        raise ValueError("Exactly 5 points are required")

    # Convert points to numpy array for easier manipulation
    points = np.array(points)

    # Create vectors from the first point to the other points
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    v3 = points[3] - points[0]
    v4 = points[4] - points[0]

    # Calculate the normal vector to the plane defined by the first three vectors
    normal = np.cross(v1, v2)

    # Check if the fourth and fifth vectors are in the same plane
    return np.isclose(np.dot(normal, v3),0) and np.isclose(np.dot(normal, v4),0)
def may_be_plain_face(face):
    # sample_params = np.array([[0, 0], [0.2, 0.3], [1, 0], [1, 1], [0.5, 0.5]]) 
    # points=[]
    # for uv in sample_params:
    #     points.append(pointOnFace(face,uv))
    points=randn_uvgrid(face, num=5, uvs=False, method="point", given_uvs=None, bounds=None)
    return are_points_on_same_plane(points)