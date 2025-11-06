from scipy.special import binom
import numpy as np

from typing import Optional,List
import math

def isClose(pt1,pt2,tol=1e-6):
    return np.linalg.norm(pt1-pt2)<=tol
class Triangle:
    def __init__(self,v1,v2,v3) -> None:
        self.v1=v1
        self.v2=v2
        self.v3=v3
        self.control_points=None

class BoundaryEdge:
    def __init__(self,start_point,end_point,uv,crv,edge3d) -> None:
        self.start_point=start_point
        self.end_point=end_point
        self.params = uv
        self.crv=crv
        self.edge3d=edge3d

class Rectangular2TriangularBezier:

    def __init__(self) -> None:
        pass

    def convert_(self,control_pts,invert=False):
        '''
            control_pts: m x n array
        '''
        m,n,_=control_pts.shape
        m-=1
        n-=1
        nodes=[]

        degree=m+n
        const1=1./binom(degree,n)

        for s in range(degree,-1,-1):
            for b in range(s+1):
                a=s-b
                v=[]
                for j in range(a+1):
                    const_aj=binom(a,j)

                    item_sum=[]

                    for k in range(max(0,b-m+j),min(b,n-a+j)+1):
                        if invert:
                            item=control_pts[m-j,n-k]*binom(b,k)*binom(m+n-a-b,m+k-j-b)
                        else:
                            item=control_pts[j,k]*binom(b,k)*binom(m+n-a-b,m+k-j-b)
                        item_sum.append(item)

                    item_sum=sum(item_sum)
                    item_sum*=const_aj

                    v.append(item_sum)
                v=sum(v)
                nodes.append(v)

        nodes=np.stack(nodes)
        nodes*=const1

        return degree,nodes

    def convert(self,control_pts,rational=False):
        if not rational:
            deg,nodes1=self.convert_(control_pts)
            deg,nodes2=self.convert_(control_pts,invert=True)
        else:
            control_pts=np.array(control_pts)
            control_pts[...,:-1]*=control_pts[...,[-1]]

            deg,nodes1=self.convert_(control_pts)
            deg,nodes2=self.convert_(control_pts,invert=True)
            nodes1[...,:-1]/=nodes1[...,[-1]]
            nodes2[...,:-1]/=nodes2[...,[-1]]

        return deg,nodes1,nodes2

class PointInfo:
    def __init__(self) -> None:
        self.coord=None
        self.boundary_status = None # 0: inside, 1: outside, 2: on
        self.is_on_rectangle = False
        self.param_on_curve=None
        self.belong_edges=[]
        self.belong_rectangle_indices=[] # [(i,j),...]

class RectInfo:
    def __init__(self) -> None:
        self.end_points=None
        self.isBroken=False
        self.upper_triangle=None
        self.lower_triangle=None


class PointsManager:
    def __init__(self,tolerance=1e-4) -> None:
        self.tolerance = tolerance
        self.points_dict = dict()
        self.scale = 1 / tolerance  # Scaling factor for quantization
        self.point_data=[]
    
    def getPointInfomation(self,point):
        """
            args: point: (x,y)
            return: PointInfo or None if not found
        """
        h=self.getHash(point)
        idx=self.points_dict.get(h)
        if idx is None:
            return None
        return self.point_data[idx]

    def addPointInfo(self,point_info:PointInfo):
        """
            args: point_info: PointInfo
        """
        h=self.getHash(point_info.coord)
        idx=self.points_dict.get(h)
        if idx is None:
            idx=len(self.point_data)
            self.points_dict[h]=idx
            self.point_data.append(point_info)

        return idx
    def addPoint(self,point):
        """
            args: point: (x,y)
        """
        h=self.getHash(point)
        idx=self.points_dict.get(h)
        if idx is None:
            idx=len(self.point_data)
            self.points_dict[h]=idx
            point_info=PointInfo()
            point_info.coord=point
            self.point_data.append(point_info)

        return idx
    def getPointId(self,point):
        """
            args: point: (x,y)
        """
        h=self.getHash(point)
        return self.points_dict.get(h)

    def points(self)->List[PointInfo]:
        """
            return Iterable of PointInfos
        """
        return self.point_data
    def getHash(self,point):
        return (int(round(point[0] * self.scale)),int(round(point[1] * self.scale)))


class TraingleEdgeManager:
    def __init__(self) -> None:
        self.data=dict()

    def get_adjacent_triangles(self,edge):
        """
            args: edge: (idx1,idx2)
            return: adjacent triangles: [Trangle1,Trangle2]
        """

        if edge[1]>edge[0]:
            edge=(edge[1],edge[0])
        return self.data[edge]
    def add_adjacent_triangles(self,edge,triangle):
        """
        """
        if edge[1]>edge[0]:
            edge=(edge[1],edge[0])
        
        lst=self.data.get(edge,[])
        lst.append(triangle)
        if len(lst)==1:
            self.data[edge]=lst

    def connections(self):
        """
            return: Iterable of (edge,triangle_id)
        """
        result=[]
        for values in self.data.values():
            if len(values)<2:
                continue
            u,v = values
            result.append((u,v))
            result.append((v,u))
        return result