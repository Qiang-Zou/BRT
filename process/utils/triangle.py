from typing import List


class Triangle:
    def __init__(self, v1, v2, v3) -> None:
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.control_points = None


class BoundaryEdge:
    def __init__(self, start_point, end_point, uv, crv, edge3d) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.params = uv
        self.crv = crv
        self.edge3d = edge3d


class PointInfo:
    def __init__(self) -> None:
        self.coord = None
        self.boundary_status = None  # 0: inside, 1: outside, 2: on
        self.is_on_rectangle = False
        self.param_on_curve = None
        self.belong_edges = []
        self.belong_rectangle_indices = []  # [(i,j),...]


class PointsManager:
    def __init__(self, tolerance=1e-4) -> None:
        self.tolerance = tolerance
        self.points_dict = dict()
        self.scale = 1 / tolerance  # Scaling factor for quantization
        self.point_data = []

    def getPointInfomation(self, point):
        """
        args: point: (x,y)
        return: PointInfo or None if not found
        """
        h = self.getHash(point)
        idx = self.points_dict.get(h)
        if idx is None:
            return None
        return self.point_data[idx]

    def addPointInfo(self, point_info: PointInfo):
        """
        args: point_info: PointInfo
        """
        h = self.getHash(point_info.coord)
        idx = self.points_dict.get(h)
        if idx is None:
            idx = len(self.point_data)
            self.points_dict[h] = idx
            self.point_data.append(point_info)

        return idx

    def addPoint(self, point):
        """
        args: point: (x,y)
        """
        h = self.getHash(point)
        idx = self.points_dict.get(h)
        if idx is None:
            idx = len(self.point_data)
            self.points_dict[h] = idx
            point_info = PointInfo()
            point_info.coord = point
            self.point_data.append(point_info)

        return idx

    def getPointId(self, point):
        """
        args: point: (x,y)
        """
        h = self.getHash(point)
        return self.points_dict.get(h)

    def points(self) -> List[PointInfo]:
        """
        return Iterable of PointInfos
        """
        return self.point_data

    def getHash(self, point):
        return (int(round(point[0] * self.scale)), int(round(point[1] * self.scale)))


class TraingleEdgeManager:
    def __init__(self) -> None:
        self.data = dict()

    def get_adjacent_triangles(self, edge):
        """
        args: edge: (idx1,idx2)
        return: adjacent triangles: [Trangle1,Trangle2]
        """

        if edge[1] > edge[0]:
            edge = (edge[1], edge[0])
        return self.data[edge]

    def add_adjacent_triangles(self, edge, triangle):
        """ """
        if edge[1] > edge[0]:
            edge = (edge[1], edge[0])

        lst = self.data.get(edge, [])
        lst.append(triangle)
        if len(lst) == 1:
            self.data[edge] = lst

    def connections(self):
        """
        return: Iterable of (edge,triangle_id)
        """
        result = []
        for values in self.data.values():
            if len(values) < 2:
                continue
            u, v = values
            result.append((u, v))
            result.append((v, u))
        return result
