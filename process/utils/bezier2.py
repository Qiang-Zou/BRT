import numpy as np
import logging
from utils.triangle import Triangle,BoundaryEdge,isClose,Rectangular2TriangularBezier
from occwl.face import Face
from occwl.geometry import geom_utils
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps

def pointOnFace(face, uv):
    """
    Evaluate the face geometry at given parameter

    Args:
        uv (np.ndarray or tuple): Surface parameter
    
    Returns:
        np.ndarray: 3D Point
    """
    self=face
    loc = TopLoc_Location()
    surf = BRep_Tool().Surface(self.topods_shape(), loc)
    pt = surf.Value(uv[0], uv[1])
    pt = pt.Transformed(loc.Transformation())
    return geom_utils.gp_to_numpy(pt)

def getCenterAndScale(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    bbox = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    bbox = np.array(bbox)

    diag = bbox[1] - bbox[0]
    scale = [(2.0 / d) if d>1e-6 else 1 for d in diag]
    scale= np.array(scale)
    center = 0.5 * (bbox[0] + bbox[1])

    return center,scale

def bernstein_polynomial_all(degree,u,v):
    w=1-u-v
    init_array=np.array([[w,0],[u,v]],dtype=np.float32)
    if degree>1:
        for i in range(degree-1):
            new_array=np.zeros(((i+3),(i+3)),dtype=np.float32)
            for j in range(i+2):
                for k in range(j+1):
                    new_array[j,k]+=init_array[j,k]*w
                    new_array[j+1,k]+=init_array[j,k]*u
                    new_array[j+1,k+1]+=init_array[j,k]*v
            init_array=new_array

    nodes=[]
    for i in range(degree+1):
        for j in range(degree+1-i):
            nodes.append(init_array[degree-i,j])
    return np.array(nodes)
def bernstein_polynomial_all_multi(degree,u,v):
    nodes=[bernstein_polynomial_all(degree,u[i],v[i]) for i in range(len(u))]
    nodes=np.stack(nodes)
    return nodes
def fit_bezier_surface2(points,uvs, initial_control_points=None,bn_cache=[]):
    """Fit a triangular Bézier surface to the given points."""
    if bn_cache is None or len(bn_cache)==0:
        bn=bernstein_polynomial_all_multi(6,uvs[:,0],uvs[:,1])
        if bn_cache is not None:
            bn_cache.append(bn)
    else:
        bn=bn_cache[0]
    result=np.linalg.lstsq(bn.T@bn,bn.T@points,rcond=None)
    ctrl=result[0] 
    residuals=result[1]

    if residuals is None:
        logging.warning("Residuals are None")
    else:
        err=np.linalg.norm(residuals)
        if err>1e-3:
            logging.warning(f"Large error: {err}")

    weight=np.ones_like(ctrl[:,0:1])
    ctrl=np.concatenate((ctrl,weight),axis=-1)
    return ctrl


def generate_uvw(num_points):
    points = []
    for _ in range(num_points):
        x = np.random.rand()
        y = np.random.rand()
        if x + y > 1:
            u = x - (x + y - 1)
            v = y - (x + y - 1)
        else:
            u = x
            v = y

        w = 1 - u - v
        
        points.append((u, v, w))

    points=np.array(points)
    
    return points
def getControlPointsFromApproximation(triangle:Triangle,face:Face,edge=None,edge_idx=0,uvs_random=generate_uvw(100)[:,:2]):
    v1=triangle.v1
    v2=triangle.v2
    v3=triangle.v3

    uvs_base=np.array([[0,0],[0,1],[1,0],[0,1/3],[0,2/3],[1/3,2/3],[2/3,1/3],[2/3,0],[1/3,0]])

    uvs=np.concatenate([uvs_base,uvs_random],axis=0)

    params=[[v3[i]*u+v2[i]*v+v1[i]*(1-u-v) for i in range(2)] for (u,v) in uvs]
    points=[pointOnFace(face,p) for p in params]
    if edge:
        edge:BoundaryEdge
        old_points=[params[edge_idx],params[(edge_idx+1)%3]]

        if isClose(edge.start_point,old_points[0]):
            params_=[edge.params[0]*2/3+edge.params[1]*1/3,edge.params[0]*1/3+edge.params[1]*2/3]

        else:
            params_=[edge.params[0]*1/3+edge.params[1]*2/3,edge.params[0]*2/3+edge.params[1]*1/3]
            
        params_=[geom_utils.gp_to_numpy(edge.crv.Value(p)) for p in params_]
        replace_points=[pointOnFace(face,p) for p in params_]

        points[3+edge_idx*2:3+edge_idx*2+2]=replace_points

    points=np.array(points)

    center,scale=getCenterAndScale(points)

    points=points-center
    points=points*scale

    ctrl_pts=points

    ctrl_pts=fit_bezier_surface2(points,uvs,ctrl_pts)

    ctrl_pts[...,:3]/=scale
    ctrl_pts[...,:3]+=center

    return ctrl_pts

def getControlPointsFromRect(patch,surface,loc):
    tri_converter=Rectangular2TriangularBezier()
    nU_ctrl_pts, nV_ctrl_pts = 4, 4
    channel_size=4
    
    max_degree=3
    patch.Increase(max_degree, max_degree)
    poles = np.zeros([nU_ctrl_pts, nV_ctrl_pts, channel_size])

    for u, v in np.ndindex((nU_ctrl_pts, nV_ctrl_pts)):
        p = patch.Pole(u+1, v+1)
        w = patch.Weight(u+1, v+1)

        p=p.Transformed(loc.Transformation())
        poles[u, v] = [p.X(), p.Y(), p.Z(), w]

    if poles[...,-1].sum()<1e-6:
        poles[...,-1]=1

    deg,node1,node2=tri_converter.convert(poles,rational=True)

    return node1,node2
