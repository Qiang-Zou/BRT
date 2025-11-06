from typing import List,Optional
import math
import logging
from OCC.Core.TColStd import TColStd_Array1OfReal as ArrReal
from utils import chordErrorCheck
import numpy as np
from occwl.geometry import geom_utils
from OCC.Core.Geom2d import Geom2d_Curve,Geom2d_Line
from OCC.Core.gp import gp_Pnt2d,gp_Dir2d,gp_Vec2d
from OCC.Core.Geom2dAPI import Geom2dAPI_InterCurveCurve,Geom2dAPI_ProjectPointOnCurve
from occwl.face import Face
from utils.triangle import BoundaryEdge,Triangle,PointsManager,TraingleEdgeManager
from utils.bezier2 import getControlPointsFromApproximation,getControlPointsFromRect
from OCC.Core.GeomConvert\
    import GeomConvert_BSplineSurfaceToBezierSurface as Converter
from contextlib import contextmanager

REPORT_ERROR=True

class Intersection:
    def __init__(self,nbpoints,points,params,crv:Geom2d_Curve,interval,valid:bool=True,two_point_on_same_line=False) -> None:
        self._crv=crv
        self._interval=interval
        self._valid=valid
        self._nbpoints=nbpoints
        self._points=points
        self._params=params
        self._two_point_on_same_line=two_point_on_same_line
    @property
    def TwoPointOnSameLine(self):
        return self._two_point_on_same_line
    @property
    def NbPoints(self)->int:
        '''
            number of intersection points
        '''
        if self._valid:
            return self._nbpoints
        return 0

    @property
    def Points(self)->List:
        if not self._valid:
            return []
        return self._points

    @property
    def Parameters(self)->List:
        '''
            parameters of the intersection points on the curve
        '''
        if not self._valid:
            return []
        return self._params
    @property
    def Curve(self):
        '''
            the curve that the intersection points are on
            return curve,interval
        '''
        return self._crv,self._interval
    def deleteMiddlePoints(self):
        if self._nbpoints>2:
            self._nbpoints=2
            self._points=[self._points[0],self._points[-1]]
            self._params=[self._params[0],self._params[-1]]

class Rectangle:
    def __init__(self) -> None:
        self._points=None
        self.is_leaf=True
        self.leaf_info=None
        self.sub_rects=None
        self.discarded=False
        self.level=0
    @property
    def points(self):
        return self._points

    @points.setter
    def points(self,points:List[float]):
        self._points=np.array(points)

    def isCorner(self,point:List[float],tol=9e-3):
        return any([np.linalg.norm(point-p)<tol for p in self.points])
    def area(self):
        return np.linalg.norm(np.cross(self.points[1]-self.points[0],self.points[2]-self.points[0]))
    
    def center(self):
        return np.mean(self.points,axis=0)

    def split(self,split_point=None,auto_ajust=True,check_area=True,tol=9e-3,discarded=None):
        if discarded is None:
            discarded=self.discarded
        self.sub_rects=[Rectangle() for _ in range(4)]
        if split_point is None:
            self.sub_rects[0].points=[self.points[0],(self.points[0]+self.points[1])/2,(self.points[0]+self.points[2])/2,(self.points[0]+self.points[3])/2]
            self.sub_rects[1].points=[(self.points[0]+self.points[1])/2,self.points[1],(self.points[1]+self.points[2])/2,(self.points[1]+self.points[3])/2]
            self.sub_rects[2].points=[(self.points[0]+self.points[2])/2,(self.points[1]+self.points[2])/2,self.points[2],(self.points[2]+self.points[3])/2]
            self.sub_rects[3].points=[(self.points[0]+self.points[3])/2,(self.points[1]+self.points[3])/2,(self.points[2]+self.points[3])/2,self.points[3]]
        else:
            if auto_ajust:
                min_x,max_x,min_y,max_y=self.getBoundary()
                if split_point[0] is None or math.fabs(split_point[0]-min_x)<tol or math.fabs(split_point[0]-max_x)<tol:
                    split_point[0]=(min_x+max_x)/2
                if split_point[1] is None or math.fabs(split_point[1]-min_y)<tol or math.fabs(split_point[1]-max_y)<tol:
                    split_point[1]=(min_y+max_y)/2

            if not self.contains(split_point):
                raise ValueError("split point not in the rectangle")

            self.sub_rects[0].points=[self.points[0],(split_point[0],self.points[0][1]),(self.points[0][0],split_point[1]),split_point]
            self.sub_rects[1].points=[(split_point[0],self.points[1][1]),self.points[1],split_point,(self.points[1][0],split_point[1])]
            self.sub_rects[2].points=[(self.points[2][0],split_point[1]),split_point,self.points[2],(split_point[0],self.points[2][1])]
            self.sub_rects[3].points=[split_point,(self.points[3][0],split_point[1]),(split_point[0],self.points[3][1]),self.points[3]]

        for sub_rect in self.sub_rects:
            sub_rect.level=self.level+1
            sub_rect.discarded=discarded
        if check_area:
            for sub_rect in self.sub_rects:
                if sub_rect.area()<tol*tol:
                    sub_rect.discarded=True
    def getBoundary(self):
        max_x=max(self.points,key=lambda x:x[0])[0]
        min_x=min(self.points,key=lambda x:x[0])[0]
        max_y=max(self.points,key=lambda x:x[1])[1]
        min_y=min(self.points,key=lambda x:x[1])[1]
        return min_x,max_x,min_y,max_y
    def contains(self,point:List[float],boundary=True,tol=1e-4)->bool:
        min_x,max_x,min_y,max_y=self.getBoundary()
        # return min_x<=point[0]<=max_x and min_y<=point[1]<=max_y
        if boundary:
            return point[0]-min_x>=-tol and max_x-point[0]>=-tol and point[1]-min_y>=-tol and max_y-point[1]>=-tol
        return point[0]-min_x>tol and max_x-point[0]>tol and point[1]-min_y>tol and max_y-point[1]>tol

    def onBoundary(self,point:List[float],tol=1e-4):
        min_x,max_x,min_y,max_y=self.getBoundary()
        return math.fabs(point[0]-min_x)<tol or math.fabs(point[0]-max_x)<tol or math.fabs(point[1]-min_y)<tol or math.fabs(point[1]-max_y)<tol

    def allCornersInFace(self,face:Face):
        for point in self.points:
            vis=face.visibility_status(point)
            if vis==1 or vis==3:
                return False
        return True
    def anyCornersInFace(self,face:Face):
        for point in self.points:
            vis=face.visibility_status(point)
            if vis==0 or vis==2:
                return True
        return False

class Intersector:
    def intersect(self,rect:Rectangle,curve:Geom2d_Curve,interval,tol=9e-3)->Intersection:
        rect_points=rect.points
        x_line1=constructLineXDir(rect_points[0][1])
        x_line2=constructLineXDir(rect_points[2][1])
        y_line1=constructLineYDir(rect_points[0][0])
        y_line2=constructLineYDir(rect_points[1][0])

        points=[]
        params=[]
        two_points_on_same_line=False
        for line in [x_line1,x_line2,y_line1,y_line2]:
            intersection=self.intersectWithLine(line,rect,curve,interval,tol=tol)
            if intersection is not None:
                if len(intersection[0])>1:
                    two_points_on_same_line=True
                points.extend(intersection[0])
                params.extend(intersection[1])

        # logging.debug("intersection points:"+str(len(points)))
        if len(points)>0:
            # sort params and points by params
            sorted_params_points=sorted(zip(params,points),key=lambda x:x[0])
            params=[x[0] for x in sorted_params_points]
            points=[x[1] for x in sorted_params_points]

            # remove duplicate points
            i=0
            new_points=[]
            new_params=[]
            multi_cnts=[1]
            while i<len(points)-1:
                if np.linalg.norm(points[i]-points[i+1])<tol:
                    multi_cnts[-1]+=1
                else:
                    new_points.append(points[i])
                    new_params.append(params[i])
                    multi_cnts.append(1)
                i+=1
            
            new_points.append(points[-1])
            new_params.append(params[-1])

            points=[]
            params=[]

            tangent_points=[]
            tangent_points_param=[]

            for i in range(len(new_points)):
                point=new_points[i]
                if rect.isCorner(point) and multi_cnts[i]<2:
                    tangent_points.append(point)
                    tangent_points_param.append(new_params[i])
                    continue
                points.append(point)
                params.append(new_params[i])

            if len(points)==1:
                    p0=geom_utils.gp_to_numpy(curve.Value(interval[0]))
                    p1=geom_utils.gp_to_numpy(curve.Value(interval[1]))

                    # no intersection
                    # if np.linalg.norm(p0-points[0])<1e-6 or np.linalg.norm(p1-points[0])<1e-6:
                    #     return Intersection(0,[],[],curve,interval)

                    # logging.debug(str(p0))
                    # logging.debug(str(p1))

                    points.extend(tangent_points)
                    params.extend(tangent_points_param)

                    # tol=9e-3
                    if rect.onBoundary(p0,tol):
                        repeat=False
                        for p in points:
                            if np.linalg.norm(p-p0)<tol:
                                repeat=True
                                break
                        if not repeat:
                            points.append(p0)
                            params.append(interval[0])
                    if rect.onBoundary(p1,tol):
                        repeat=False
                        for p in points:
                            if np.linalg.norm(p-p1)<tol:
                                repeat=True
                                break
                        if not repeat:
                            points.append(p1)
                            params.append(interval[1])

                    if len(points)==1:
                        l0=rect.contains(p0,boundary=False)
                        l1=rect.contains(p1,boundary=False)
                        if (not l0 and not l1) or (l0 and l1):
                        # if (not l0 and not l1):
                            # logging.debug(f"with 1 intersection point, but goes through the rectangle,{l0},{l1}")
                            return Intersection(0,[],[],curve,interval)
                    else:
                        logging.debug("collect close intersection point ignored before")
        return Intersection(nbpoints=len(points),
                            points=points,
                            params=params,
                            crv=curve,
                            interval=interval,
                            valid=True,
                            two_point_on_same_line=two_points_on_same_line)

    def intersectWithLine(self,line:Geom2d_Line,rect:Rectangle,curve:Geom2d_Curve,interval,tol=1e-4)->Intersection:
        intersector=Geom2dAPI_InterCurveCurve(curve,line,tol)
        points=[]
        params=[]
        for i in range(intersector.NbPoints()):
            point=intersector.Point(i+1)
            # logging.debug("point:"+str(geom_utils.gp_to_numpy(point)))
            projection=Geom2dAPI_ProjectPointOnCurve(point,curve,interval[0],interval[1])
            if projection.NbPoints()==0 or projection.LowerDistance()>tol:
                continue
            point=geom_utils.gp_to_numpy(point)
            
            if not rect.contains(point,tol=tol):
                continue

            # logging.debug("point accepted")
            points.append(point)
            params.append(projection.LowerDistanceParameter())
        return points,params

def constructLineXDir(value):
    p1=gp_Pnt2d(0,value)
    p2=gp_Pnt2d(1,value)
    vec=gp_Vec2d(p1,p2)
    dir=gp_Dir2d(vec)
    return Geom2d_Line(p1,dir)
def constructLineYDir(value):
    p1=gp_Pnt2d(value,0)
    p2=gp_Pnt2d(value,1)
    vec=gp_Vec2d(p1,p2)
    dir=gp_Dir2d(vec)
    return Geom2d_Line(p1,dir)

def curveInRect(crv:Geom2d_Curve,interval,rect:Rectangle,intersection:Intersection,tol=9e-3):
    if intersection.NbPoints>1:
        return False
    umin,umax=interval
    return rect.contains(geom_utils.gp_to_numpy(crv.Value(umin)),boundary=False,tol=tol) or rect.contains(geom_utils.gp_to_numpy(crv.Value(umax)),boundary=False,tol=tol)
def chordErrorCheckInRect(intersection:Intersection,rect:Rectangle,tol=0.995,edge_sample_num=20):
    crv,interval=intersection.Curve
    crv:Geom2d_Curve
    interval=intersection.Parameters
    if intersection.NbPoints>1:
        if intersection.NbPoints!=2:
            raise ValueError("intersection points are not 2, but {}".format(intersection.NbPoints))
        if crv.IsPeriodic():
            u=(interval[0]+interval[1])/2
            middle_point=geom_utils.gp_to_numpy(crv.Value(u))
            if not rect.contains(middle_point):
                interval[0],interval[1]=interval[1],interval[0]
                interval[1]+=crv.Period()


        t=np.linspace(interval[0],interval[1],edge_sample_num)
        points=[geom_utils.gp_to_numpy(crv.Value(value)) for value in t]

        dist_fn=lambda x,y:np.linalg.norm(x-y)

        for i in range(1,len(t)-1):
            if not chordErrorCheck(points[0],points[i],points[-1],dist_fn,tol):
                return False
    return True


def hasSamePoint(crv1:Geom2d_Curve,interval1,crv2:Geom2d_Curve,interval2,tol=9e-3):
    '''
        if has same point on the curve,retrun that point
        else return None
    '''
    p1=geom_utils.gp_to_numpy(crv1.Value(interval1[0]))
    p2=geom_utils.gp_to_numpy(crv1.Value(interval1[1]))
    p3=geom_utils.gp_to_numpy(crv2.Value(interval2[0]))
    p4=geom_utils.gp_to_numpy(crv2.Value(interval2[1]))
    if np.linalg.norm(p1-p3)<tol:
        return p1
    if np.linalg.norm(p1-p4)<tol:
        return p1
    if np.linalg.norm(p2-p3)<tol:
        return p2
    if np.linalg.norm(p2-p4)<tol:
        return p2
    return None

def make_triangles(face:Face,curve,interval,end_points):
    '''
        make triangles from face and curves and 3 end points
    '''
    edge=BoundaryEdge(end_points[0],end_points[2],interval,curve,None)
    tri=Triangle(end_points[0],end_points[1],end_points[2])
    tri.control_points=getControlPointsFromApproximation(tri,face,edge,2)
    return [tri]
def make_boundary_rect(face:Face,curve,interval,end_points):
    '''
        make rects from face and curves and 4 end points
    '''
    tri1=Triangle(end_points[0],end_points[1],end_points[2])
    tri1.control_points=getControlPointsFromApproximation(tri1,face)

    tri2=Triangle(end_points[0],end_points[2],end_points[3])
    edge=BoundaryEdge(end_points[0],end_points[3],interval,curve,None)
    tri2.control_points=getControlPointsFromApproximation(tri2,face,edge,2)

    return [tri1,tri2]

def make_rect(face:Face,rect:Rectangle,nurbs_surface,loc):
    '''
        make rects from face and curves and 4 end points
    '''
    end_points=rect.points
    x_min,x_max,y_min,y_max=rect.getBoundary()
    try:
        converter=Converter(nurbs_surface,x_min,x_max,y_min,y_max,1e-4)
    except RuntimeError as e:
        logging.error(f"failed to convert surface to bezier surface: {e}\n with following boundary: {end_points}")
        rect.discarded=True
        return

    uNumPatches = converter.NbUPatches()
    vNumPatches = converter.NbVPatches()

    if uNumPatches==0 or vNumPatches==0:
        logging.error("failed to convert surface to bezier surface: no patches")
        rect.discarded=True
        return
    # assert uNumPatches==1 and vNumPatches==1

    # uKnots = ArrReal(1, uNumPatches+1)
    # vKnots = ArrReal(1, vNumPatches+1)

    # converter.UKnots(uKnots)
    # converter.VKnots(vKnots)
    tris=[]

    for (i,j) in np.ndindex((uNumPatches,vNumPatches)):
        patch=converter.Patch(i+1,j+1)

        node1,nodes2=getControlPointsFromRect(patch,nurbs_surface,loc)

        tri1=Triangle(end_points[0],end_points[1],end_points[2])
        tri1.control_points=node1

        tri2=Triangle(end_points[1],end_points[2],end_points[3])
        tri2.control_points=nodes2

        tris.append(tri1)
        tris.append(tri2)

    return tris

def CollectTris(rectangle:Rectangle,edgeManager:TraingleEdgeManager,pointsManager:PointsManager,tris_lst:List[Triangle]):
    if rectangle.discarded:
        return
    if rectangle.is_leaf:
        for tri in rectangle.leaf_info:
            if type(tri)!=Triangle:
                raise ValueError("triangle is not Triangle")

            v1,v2,v3=(tri.v1,tri.v2,tri.v3)
            v1=pointsManager.addPoint(v1)
            v2=pointsManager.addPoint(v2)
            v3=pointsManager.addPoint(v3)
            idx=len(tris_lst)

            edgeManager.add_adjacent_triangles((v1,v2),idx)
            edgeManager.add_adjacent_triangles((v2,v3),idx)
            edgeManager.add_adjacent_triangles((v1,v3),idx)

            tris_lst.append(tri)
    else:
        for sub_rect in rectangle.sub_rects:
            CollectTris(sub_rect,edgeManager,pointsManager,tris_lst)

def CollectTrisInLine(rectangle:Rectangle,tris_lst:List[Triangle],face,surface,loc):
    if rectangle.discarded:
        # tris_lst.append(None)
        if rectangle.area()>1e-5:
            tris=make_rect(face,rectangle,surface,loc)
            for tri in tris:
                tris_lst.append((None,tri))
        return
    if rectangle.is_leaf:
        for tri in rectangle.leaf_info:
            if type(tri)!=Triangle:
                if rectangle.area()>1e-5:
                    tris=make_rect(face,rectangle,surface,loc)
                    for tri in tris:
                        tris_lst.append((None,tri))
                # rectangle.discarded=True
                # tris_lst.append(None)
                return
                # raise ValueError("triangle is not Triangle")

            tris_lst.append(tri)
    else:
        # order=[0,1,3,2]
        # for idx in order:
        #     sub_rect=rectangle.sub_rects[idx]
        #     CollectTrisInLine(sub_rect,tris_lst)
        for sub_rect in rectangle.sub_rects:
            CollectTrisInLine(sub_rect,tris_lst,face,surface,loc)

def HandleLeaves(face:Face,rectangle:Rectangle,surface,loc):
    if rectangle.discarded:
        return
    if rectangle.is_leaf:
        splitBoundaryRectangle(face,rectangle,surface,loc)
    else:
        for sub_rect in rectangle.sub_rects:
            HandleLeaves(face,sub_rect,surface,loc)

def HandleLeavesSimple(face:Face,rectangle:Rectangle,surface,loc):
    # if rectangle.discarded:
    #     return
    if rectangle.is_leaf:
        # splitBoundaryRectangle(face,rectangle,surface,loc)
        rectangle.leaf_info=make_rect(face,rectangle,surface,loc)
    else:
        for sub_rect in rectangle.sub_rects:
            HandleLeavesSimple(face,sub_rect,surface,loc)

def splitBoundaryRectangle(face:Face,rectangle:Rectangle,surface,loc,tol=9e-3):
    intersections=rectangle.leaf_info
    if intersections is None or len(intersections)==0:
        rectangle.leaf_info=make_rect(face,rectangle,surface,loc)
        return
    # logging.debug("splitting boundary rectangle")
    # logging.debug("rectangle points:{}".format(rectangle.points))
    intersection=intersections[0]
    if type(intersection)==Triangle:
        # spliting has done
        return
    if len(intersections) >1:
        if report_error():
            raise ValueError("too many intersections!")
        else:
            logging.warning("too many intersections!")
            # rectangle.leaf_info=make_rect(face,rectangle,surface,loc)
            return
    if intersection.NbPoints!=2:
        if report_error():
            raise ValueError("intersection points are not 2, but {}".format(intersection.NbPoints))
        else:
            logging.warning("intersection points are not 2, but {}".format(intersection.NbPoints))
            # rectangle.leaf_info=make_rect(face,rectangle,surface,loc)
            # return
            if intersection.NbPoints==1:
                rectangle.leaf_info=make_rect(face,rectangle,surface,loc)
                return
            intersection.deleteMiddlePoints()

    point_on_line_01=0
    point_on_line_13=0
    point_on_line_32=0
    point_on_line_21=0

    for idx,point in enumerate(intersection.Points):
        if pointOnLine(rectangle.points[0],rectangle.points[1],point,tol=tol):
            point_on_line_01+=(1<<idx)
        if pointOnLine(rectangle.points[1],rectangle.points[3],point,tol=tol):
            point_on_line_13+=(1<<idx)
        if pointOnLine(rectangle.points[3],rectangle.points[2],point,tol=tol):
            point_on_line_32+=(1<<idx)
        if pointOnLine(rectangle.points[2],rectangle.points[0],point,tol=tol):
            point_on_line_21+=(1<<idx)

    status=[point_on_line_01,point_on_line_13,point_on_line_32,point_on_line_21]
    # logging.debug("two points:{},{}".format(intersection.Points[0],intersection.Points[1]))
    # logging.debug("status:{}".format(status))
    # print(status)

    hit_flag=0
    for item in status:
        hit_flag|=item
        if item==0x3:
            logging.warning("intersection points are on the same line")
            rectangle.leaf_info=make_rect(face,rectangle,surface,loc)
            # rectangle.leaf_info=make_rect(face,rectangle,surface,loc)
            # raise NotImplementedError("intersection points are on the same line")
            return
    if hit_flag!=0x03:
        logging.warning("not enough intersection points are on the boundary")
        # rectangle.leaf_info=make_rect(face,rectangle,surface,loc)
        # raise NotImplementedError("intersection points are on the same line")
        return
    clockwise_index=[0,1,3,2]
    # next_index=lambda x:(x+1)%4
    turned=False
    try:
        for idx in range(4):
            item=status[idx]
            if item>0:
                corner_point=False

                reverse=(item==0x02)
                if status[(idx+1)%4]>0:
                    if item!=status[(idx+1)%4]:
                        # intersection on ajacent edges

                        end_index=clockwise_index[(idx+1)%4]
                        end_point=rectangle.points[end_index]
                        vis=face.visibility_status(end_point)
                        if vis==1 or vis==3:
                            points=intersection.Points
                            oppo_point=points[0]+points[1]-end_point

                            rectangle.split(oppo_point,auto_ajust=False,tol=tol)
                            rectangle.is_leaf=False

                            rectangle.sub_rects[end_index].leaf_info=\
                                make_triangles(face=face,
                                            curve=intersection.Curve[0],
                                            interval=intersection.Parameters[::-1] if reverse else intersection.Parameters,
                                            end_points=(intersection.Points[reverse],oppo_point,intersection.Points[not reverse]))
                        else:
                            points=intersection.Points
                            oppo_point=points[0]+points[1]-end_point

                            rectangle.split(oppo_point,auto_ajust=False,tol=tol)
                            rectangle.is_leaf=False

                            for rec in rectangle.sub_rects:
                                rec.discarded=True

                            rectangle.sub_rects[end_index].discarded=False
                            rectangle.sub_rects[end_index].leaf_info=\
                                make_triangles(face=face,
                                            curve=intersection.Curve[0],
                                            interval=intersection.Parameters[::-1] if reverse else intersection.Parameters,
                                            end_points=(intersection.Points[reverse],rectangle.points[end_index],intersection.Points[not reverse]))

                        # logging.debug("intersection on ajacent edges")
                        break
                    else:
                        corner_point=True
                if item==status[(idx+3)%4]:
                    corner_point=True
                if not corner_point and status[(idx+2)%4]>0 and item!=status[(idx+2)%4]:
                    # intersection on opposite edges
                    end_index=clockwise_index[(idx+1)%4]
                    end_point=rectangle.points[end_index]
                    vis=face.visibility_status(end_point)
                    # if not turned and (vis==1 or vis==3):
                    #     turned=True
                    #     continue
                    if vis==1 or vis==3:
                        turned=True
                        continue
                    else:
                        other_index=clockwise_index[(idx+2)%4]
                        other_point=rectangle.points[other_index]
                        point1=intersection.Points[item-1]
                        point2=intersection.Points[item%2]
                        if np.linalg.norm(point1-end_point)-np.linalg.norm(point2-other_point)>tol:
                            oppo_point=point2-other_point+end_point
                            rectangle.split(oppo_point,auto_ajust=False,tol=tol)
                            rectangle.is_leaf=False
                            rectangle.sub_rects[clockwise_index[(idx+3)%4]].leaf_info=\
                                make_triangles(face=face,
                                                curve=intersection.Curve[0],
                                                interval=intersection.Parameters[::-1] if reverse else intersection.Parameters,
                                                end_points=(point1,oppo_point,point2))
                        elif np.linalg.norm(point1-end_point)-np.linalg.norm(point2-other_point)<-tol:
                            oppo_point=point1-end_point+other_point
                            rectangle.split(oppo_point,auto_ajust=False,tol=tol)
                            rectangle.is_leaf=False
                            rectangle.sub_rects[clockwise_index[idx]].leaf_info=\
                                make_triangles(face=face,
                                                curve=intersection.Curve[0],
                                                interval=intersection.Parameters[::-1] if reverse else intersection.Parameters,
                                                end_points=(point1,oppo_point,point2))
                        else:
                            oppo_point=point2
                            rectangle.split(oppo_point,auto_ajust=False,tol=tol)
                            rectangle.is_leaf=False
                            rectangle.sub_rects[end_index].leaf_info=\
                                make_boundary_rect(face=face,
                                                curve=intersection.Curve[0],
                                                interval=intersection.Parameters[::-1] if reverse else intersection.Parameters,
                                                end_points=(point1,end_point,other_point,oppo_point))
                            for idx,rect in enumerate(rectangle.sub_rects):
                                if idx!=end_index:
                                    rect.discarded=True
                    break
    except Exception as e:
        logging.error("error in splitBoundaryRectangle:{}".format(e))
        return
    if not rectangle.is_leaf:
        for sub_rect in rectangle.sub_rects:
            if not sub_rect.discarded:
                splitBoundaryRectangle(face,sub_rect,surface,loc)
    else:
        if type(rectangle.leaf_info[0])!=Triangle:
            logging.warning("no triangles are generated, status:{}".format(status))
            # rectangle.leaf_info=make_rect(face,rectangle,surface,loc)
            # rectangle.discarded=True

        

def pointOnLine(p1,p2,point,tol=9e-3):
    return (np.linalg.norm(p2-point)<tol or np.linalg.norm(p1-point)<tol) or np.dot(p2-p1,point-p1)>=0 and distance_point_to_line(p1,p2,point)<tol

def distance_point_to_line(A, B, P):
    A = np.array(A)
    B = np.array(B)
    P = np.array(P)
    
    # 计算向量 AB 和 AP
    AB = B - A
    AP = P - A
    
    # 计算叉积的模
    cross_product = np.abs(np.cross(AB, AP))
    
    # 计算 AB 的模长
    AB_length = np.linalg.norm(AB)
    
    # 计算距离
    distance = cross_product / AB_length
    return distance
# define a context manager to suppress error

@contextmanager
def suppress_subdivsion_err():
    global REPORT_ERROR
    REPORT_ERROR=False
    yield
    REPORT_ERROR=True

def report_error():
    return REPORT_ERROR

def splitRectangle(face:Face,rectangle,curves,max_split=7,tol=0.7,distance_tol=1e-4,split_all=False):
    intersector=Intersector()
    stack=[rectangle]
    # logging.debug("splitting rectangle")
    # logging.debug("rectangle region:{}".format(rectangle.points))
    while len(stack)>0:
        # logging.debug("stack length:{}".format(len(stack)))
        rect:Rectangle=stack.pop()
        
        if rect.discarded:
            continue

        intersections=[]
        split_flag=False
        split_point=None
        for curve,interval in curves:
            intersection=intersector.intersect(rect,curve,interval,tol=distance_tol)

            if intersection.NbPoints>0:
                intersections.append(intersection)

            # logging.debug("intersection points:{}".format(intersection.NbPoints))
            # logging.debug("intersection status:{}".format(intersection.TwoPointOnSameLine))
            # logging.debug("curve in rect:{}".format(curveInRect(curve,interval,rect,intersection,tol=distance_tol)))
            # if intersection.NbPoints==2:
            #     logging.debug("chord error check:{}".format(chordErrorCheckInRect(intersection,rect,tol=tol)))

            if curveInRect(curve,interval,rect,intersection,tol=distance_tol) or\
                      intersection.NbPoints>2 or\
                      intersection.TwoPointOnSameLine or\
                      not chordErrorCheckInRect(intersection,rect,tol=tol):
                split_flag=True
                break

        # logging.debug("intersection number:{}".format(len(intersections)))

        if len(intersections)>1:
            split_flag=True
            if len(intersections)==2:
                point=hasSamePoint(*intersections[0].Curve,*intersections[1].Curve,tol=distance_tol) 
                if point is not None and rect.contains(point,tol=distance_tol) and not rect.isCorner(point,tol=distance_tol):
                    split_point=point

        if split_flag or split_all:
            if rect.level<max_split:
                rect.split(split_point,auto_ajust=False,tol=distance_tol)
                rect.is_leaf=False
                stack.extend(rect.sub_rects)
            else:
                rect.leaf_info=intersections
        else:
            if len(intersections)==0 and ((not rect.anyCornersInFace(face)) and face.visibility_status(rect.center())==1):
                rect.discarded=True
                continue
            rect.leaf_info=intersections
    return rectangle
