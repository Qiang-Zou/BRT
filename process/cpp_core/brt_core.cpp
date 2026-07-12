// brt_core: C++ port of the BRT tokenizer face hot path (see README.md).
//
// Line-by-line port of process/triangles3.py quadtree subdivision, boundary
// rectangle cutting and triangle collection, plus the OCC-evaluation part of
// process/utils/bezier2.py. OCC objects are shared with pythonocc by raw
// pointer (SWIG `this`), so every OCC-derived quantity is bitwise identical
// to the Python implementation. Compiled with -ffp-contract=off so scalar
// arithmetic rounds exactly like numpy scalar ops.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Standard_Failure.hxx>
#include <TopoDS_Face.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopLoc_Location.hxx>
#include <BRep_Tool.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <Geom_Surface.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BezierSurface.hxx>
#include <GeomConvert_BSplineSurfaceToBezierSurface.hxx>
#include <GeomLProp_SLProps.hxx>
#include <Geom2d_Curve.hxx>
#include <Geom2d_Line.hxx>
#include <Geom2dAPI_InterCurveCurve.hxx>
#include <Geom2dAPI_ProjectPointOnCurve.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Vec2d.hxx>
#include <gp_Dir2d.hxx>
#include <gp_Dir.hxx>
#include <gp_Trsf.hxx>

namespace py = pybind11;

namespace {

struct V2 {
    double x = 0.0, y = 0.0;
};

// np.linalg.norm(a - b) for 2-vectors: sqrt(dx*dx + dy*dy) (ddot order).
static inline double dist2(const V2& a, const V2& b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

struct Curve {
    Handle(Geom2d_Curve) crv;
    double u0 = 0.0, u1 = 0.0;
};

// Mirror of triangles3.Intersection (valid is always True at construction in
// the paths we port; the early-return "empty" intersection is nbpoints=0).
struct Isect {
    int nbpoints = 0;
    std::vector<V2> pts;
    std::vector<double> params;  // chordErrorCheckInRect mutates these in place
    const Curve* curve = nullptr;
    bool two_same_line = false;
};

struct Tri {
    V2 v1, v2, v3;
    int mode = 0;  // 0 = ctrl ready (make_rect), 1 = deferred python lstsq fit
    std::array<double, 28 * 4> ctrl{};
    std::vector<double> fitpts;  // 109*3 when mode == 1
};

struct Rect {
    V2 p[4];
    bool is_leaf = true;
    bool discarded = false;
    int level = 0;
    int leaf_state = 0;  // 0 = None, 1 = intersections list, 2 = triangle list
    std::vector<Isect> isects;
    std::vector<Tri> tris;
    std::vector<Rect> subs;  // size 4 after split()
};

struct Ctx {
    TopoDS_Face face;
    Handle(Geom_Surface) face_surf;  // BRep_Tool::Surface(face, loc), as pointOnFace
    TopLoc_Location face_loc;
    gp_Trsf trsf;
    Handle(Geom_BSplineSurface) nurbs;  // getNURBS + doKnotInsertion result
    std::unique_ptr<BRepTopAdaptor_FClass2d> classifier;  // occwl Face._trimmed
    std::vector<Curve> curves;
    std::vector<V2> uvs;  // 109 barycentric fit samples (base 9 + random 100)
    bool reversed_face = false;
};

// ---------------------------------------------------------------- Rectangle

static inline void rect_boundary(const Rect& r, double& minx, double& maxx,
                                 double& miny, double& maxy) {
    minx = maxx = r.p[0].x;
    miny = maxy = r.p[0].y;
    for (int i = 1; i < 4; i++) {
        minx = std::min(minx, r.p[i].x);
        maxx = std::max(maxx, r.p[i].x);
        miny = std::min(miny, r.p[i].y);
        maxy = std::max(maxy, r.p[i].y);
    }
}

// norm(cross(p1-p0, p2-p0)) with 2D cross -> scalar, norm -> abs.
static inline double rect_area(const Rect& r) {
    double ax = r.p[1].x - r.p[0].x, ay = r.p[1].y - r.p[0].y;
    double bx = r.p[2].x - r.p[0].x, by = r.p[2].y - r.p[0].y;
    return std::fabs(ax * by - ay * bx);
}

// np.mean(points, axis=0): sequential accumulation then /4.
static inline V2 rect_center(const Rect& r) {
    double sx = ((r.p[0].x + r.p[1].x) + r.p[2].x) + r.p[3].x;
    double sy = ((r.p[0].y + r.p[1].y) + r.p[2].y) + r.p[3].y;
    return V2{sx / 4.0, sy / 4.0};
}

static inline bool rect_contains(const Rect& r, const V2& pt, bool boundary,
                                 double tol) {
    double minx, maxx, miny, maxy;
    rect_boundary(r, minx, maxx, miny, maxy);
    if (boundary) {
        return pt.x - minx >= -tol && maxx - pt.x >= -tol &&
               pt.y - miny >= -tol && maxy - pt.y >= -tol;
    }
    return pt.x - minx > tol && maxx - pt.x > tol && pt.y - miny > tol &&
           maxy - pt.y > tol;
}

static inline bool rect_on_boundary(const Rect& r, const V2& pt, double tol) {
    double minx, maxx, miny, maxy;
    rect_boundary(r, minx, maxx, miny, maxy);
    return std::fabs(pt.x - minx) < tol || std::fabs(pt.x - maxx) < tol ||
           std::fabs(pt.y - miny) < tol || std::fabs(pt.y - maxy) < tol;
}

static inline bool rect_is_corner(const Rect& r, const V2& pt, double tol) {
    for (int i = 0; i < 4; i++)
        if (dist2(pt, r.p[i]) < tol) return true;
    return false;
}

// Rectangle.split(split_point, auto_ajust=False, check_area=True, tol).
// The explicit-split containment check uses contains() with its DEFAULT tol
// 1e-4, not the passed tol (Python quirk). sub_rects are assigned before the
// containment check can raise, exactly as in Python.
static void rect_split(Rect& r, const V2* sp, double tol) {
    bool disc = r.discarded;
    r.subs.clear();
    r.subs.resize(4);
    if (sp == nullptr) {
        auto mid = [](const V2& a, const V2& b) {
            return V2{(a.x + b.x) / 2.0, (a.y + b.y) / 2.0};
        };
        V2 m01 = mid(r.p[0], r.p[1]), m02 = mid(r.p[0], r.p[2]);
        V2 m03 = mid(r.p[0], r.p[3]), m12 = mid(r.p[1], r.p[2]);
        V2 m13 = mid(r.p[1], r.p[3]), m23 = mid(r.p[2], r.p[3]);
        r.subs[0].p[0] = r.p[0]; r.subs[0].p[1] = m01; r.subs[0].p[2] = m02; r.subs[0].p[3] = m03;
        r.subs[1].p[0] = m01;    r.subs[1].p[1] = r.p[1]; r.subs[1].p[2] = m12; r.subs[1].p[3] = m13;
        r.subs[2].p[0] = m02;    r.subs[2].p[1] = m12; r.subs[2].p[2] = r.p[2]; r.subs[2].p[3] = m23;
        r.subs[3].p[0] = m03;    r.subs[3].p[1] = m13; r.subs[3].p[2] = m23; r.subs[3].p[3] = r.p[3];
    } else {
        V2 s = *sp;
        if (!rect_contains(r, s, true, 1e-4))
            throw std::runtime_error("split point not in the rectangle");
        r.subs[0].p[0] = r.p[0];
        r.subs[0].p[1] = V2{s.x, r.p[0].y};
        r.subs[0].p[2] = V2{r.p[0].x, s.y};
        r.subs[0].p[3] = s;
        r.subs[1].p[0] = V2{s.x, r.p[1].y};
        r.subs[1].p[1] = r.p[1];
        r.subs[1].p[2] = s;
        r.subs[1].p[3] = V2{r.p[1].x, s.y};
        r.subs[2].p[0] = V2{r.p[2].x, s.y};
        r.subs[2].p[1] = s;
        r.subs[2].p[2] = r.p[2];
        r.subs[2].p[3] = V2{s.x, r.p[2].y};
        r.subs[3].p[0] = s;
        r.subs[3].p[1] = V2{r.p[3].x, s.y};
        r.subs[3].p[2] = V2{s.x, r.p[3].y};
        r.subs[3].p[3] = r.p[3];
    }
    for (int i = 0; i < 4; i++) {
        r.subs[i].level = r.level + 1;
        r.subs[i].discarded = disc;
    }
    for (int i = 0; i < 4; i++)
        if (rect_area(r.subs[i]) < tol * tol) r.subs[i].discarded = true;
}

// ------------------------------------------------------------ classification

static inline int vis_status(Ctx& c, const V2& pt) {
    return static_cast<int>(c.classifier->Perform(gp_Pnt2d(pt.x, pt.y)));
}

static inline bool any_corners_in_face(Ctx& c, const Rect& r) {
    for (int i = 0; i < 4; i++) {
        int v = vis_status(c, r.p[i]);
        if (v == 0 || v == 2) return true;
    }
    return false;
}

// ------------------------------------------------------------- intersection

static Handle(Geom2d_Line) construct_line_x(double value) {
    gp_Pnt2d p1(0.0, value), p2(1.0, value);
    gp_Vec2d vec(p1, p2);
    gp_Dir2d dir(vec);
    return new Geom2d_Line(p1, dir);
}

static Handle(Geom2d_Line) construct_line_y(double value) {
    gp_Pnt2d p1(value, 0.0), p2(value, 1.0);
    gp_Vec2d vec(p1, p2);
    gp_Dir2d dir(vec);
    return new Geom2d_Line(p1, dir);
}

// Intersector.intersectWithLine
static void intersect_with_line(const Handle(Geom2d_Line)& line, const Rect& r,
                                const Curve& cur, double tol,
                                std::vector<V2>& out_pts,
                                std::vector<double>& out_params) {
    Geom2dAPI_InterCurveCurve inter(cur.crv, line, tol);
    int nb = inter.NbPoints();
    for (int i = 0; i < nb; i++) {
        gp_Pnt2d gp = inter.Point(i + 1);
        Geom2dAPI_ProjectPointOnCurve proj(gp, cur.crv, cur.u0, cur.u1);
        if (proj.NbPoints() == 0 || proj.LowerDistance() > tol) continue;
        V2 q{gp.X(), gp.Y()};
        if (!rect_contains(r, q, true, tol)) continue;
        out_pts.push_back(q);
        out_params.push_back(proj.LowerDistanceParameter());
    }
}

// Intersector.intersect (tol = distance_tol = 1e-4 from splitRectangle)
static Isect intersect_rect_curve(Ctx& c, const Rect& r, const Curve& cur,
                                  double tol) {
    std::vector<V2> points;
    std::vector<double> params;
    bool two_same = false;
    for (int li = 0; li < 4; li++) {
        Handle(Geom2d_Line) line;
        if (li == 0) line = construct_line_x(r.p[0].y);
        else if (li == 1) line = construct_line_x(r.p[2].y);
        else if (li == 2) line = construct_line_y(r.p[0].x);
        else line = construct_line_y(r.p[1].x);
        size_t before = points.size();
        intersect_with_line(line, r, cur, tol, points, params);
        if (points.size() - before > 1) two_same = true;
    }

    Isect res;
    res.curve = &cur;

    if (!points.empty()) {
        // sorted(zip(params, points), key=lambda x: x[0]) -- stable
        std::vector<size_t> order(points.size());
        std::iota(order.begin(), order.end(), size_t(0));
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return params[a] < params[b]; });
        std::vector<V2> spts;
        std::vector<double> spar;
        for (size_t k : order) {
            spts.push_back(points[k]);
            spar.push_back(params[k]);
        }

        // consecutive dedup at < tol with multiplicity counting
        std::vector<V2> npts;
        std::vector<double> npar;
        std::vector<int> multi{1};
        size_t i = 0;
        while (i + 1 < spts.size()) {
            if (dist2(spts[i], spts[i + 1]) < tol) {
                multi.back() += 1;
            } else {
                npts.push_back(spts[i]);
                npar.push_back(spar[i]);
                multi.push_back(1);
            }
            i++;
        }
        npts.push_back(spts.back());
        npar.push_back(spar.back());

        // corner-tangency filter (isCorner default tol 9e-3)
        std::vector<V2> fpts, tpts;
        std::vector<double> fpar, tpar;
        for (size_t k = 0; k < npts.size(); k++) {
            if (rect_is_corner(r, npts[k], 9e-3) && multi[k] < 2) {
                tpts.push_back(npts[k]);
                tpar.push_back(npar[k]);
                continue;
            }
            fpts.push_back(npts[k]);
            fpar.push_back(npar[k]);
        }

        if (fpts.size() == 1) {
            gp_Pnt2d g0 = cur.crv->Value(cur.u0);
            gp_Pnt2d g1 = cur.crv->Value(cur.u1);
            V2 p0{g0.X(), g0.Y()}, p1{g1.X(), g1.Y()};

            for (size_t k = 0; k < tpts.size(); k++) {
                fpts.push_back(tpts[k]);
                fpar.push_back(tpar[k]);
            }

            if (rect_on_boundary(r, p0, tol)) {
                bool repeat = false;
                for (const V2& p : fpts)
                    if (dist2(p, p0) < tol) { repeat = true; break; }
                if (!repeat) {
                    fpts.push_back(p0);
                    fpar.push_back(cur.u0);
                }
            }
            if (rect_on_boundary(r, p1, tol)) {
                bool repeat = false;
                for (const V2& p : fpts)
                    if (dist2(p, p1) < tol) { repeat = true; break; }
                if (!repeat) {
                    fpts.push_back(p1);
                    fpar.push_back(cur.u1);
                }
            }

            if (fpts.size() == 1) {
                bool l0 = rect_contains(r, p0, false, 1e-4);
                bool l1 = rect_contains(r, p1, false, 1e-4);
                if ((!l0 && !l1) || (l0 && l1)) {
                    // Intersection(0, [], [], curve, interval): drops the
                    // two_point_on_same_line flag (Python default False).
                    res.nbpoints = 0;
                    return res;
                }
            }
        }
        res.pts = std::move(fpts);
        res.params = std::move(fpar);
    }
    res.nbpoints = static_cast<int>(res.pts.size());
    res.two_same_line = two_same;
    return res;
}

// curveInRect (tol = distance_tol = 1e-4)
static bool curve_in_rect(const Curve& cur, const Rect& r, const Isect& is,
                          double tol) {
    if (is.nbpoints > 1) return false;
    gp_Pnt2d a = cur.crv->Value(cur.u0);
    gp_Pnt2d b = cur.crv->Value(cur.u1);
    return rect_contains(r, V2{a.X(), a.Y()}, false, tol) ||
           rect_contains(r, V2{b.X(), b.Y()}, false, tol);
}

// exact replica of np.linspace(start, stop, num) rounding for scalars
static void py_linspace(double start, double stop, int num, double* out) {
    double delta = stop - start;
    int div = num - 1;
    double step = delta / div;
    if (step != 0.0) {
        for (int i = 0; i < num; i++) out[i] = static_cast<double>(i) * step;
    } else {
        for (int i = 0; i < num; i++)
            out[i] = (static_cast<double>(i) / static_cast<double>(div)) * delta;
    }
    for (int i = 0; i < num; i++) out[i] += start;
    out[num - 1] = stop;
}

// chordErrorCheckInRect(intersection, rect, tol=0.7, edge_sample_num=20)
// NOTE: the periodic branch mutates the stored intersection parameters in
// place (swap + period shift), exactly like the Python list aliasing.
static bool chord_error_check_in_rect(Isect& is, const Rect& r, double tol) {
    constexpr int kSamples = 20;
    if (is.nbpoints > 1) {
        if (is.nbpoints != 2)
            throw std::runtime_error("intersection points are not 2");
        const Handle(Geom2d_Curve)& crv = is.curve->crv;
        if (crv->IsPeriodic()) {
            double u = (is.params[0] + is.params[1]) / 2.0;
            gp_Pnt2d mid = crv->Value(u);
            if (!rect_contains(r, V2{mid.X(), mid.Y()}, true, 1e-4)) {
                std::swap(is.params[0], is.params[1]);
                is.params[1] += crv->Period();
            }
        }
        double t[kSamples];
        py_linspace(is.params[0], is.params[1], kSamples, t);
        V2 pts[kSamples];
        for (int i = 0; i < kSamples; i++) {
            gp_Pnt2d g = crv->Value(t[i]);
            pts[i] = V2{g.X(), g.Y()};
        }
        for (int i = 1; i < kSamples - 1; i++) {
            double num = dist2(pts[0], pts[kSamples - 1]);
            double den = dist2(pts[0], pts[i]) + dist2(pts[i], pts[kSamples - 1]);
            if (!(num / den > tol)) return false;
        }
    }
    return true;
}

// hasSamePoint (tol = 1e-4)
static bool has_same_point(const Isect& a, const Isect& b, double tol, V2& out) {
    gp_Pnt2d g1 = a.curve->crv->Value(a.curve->u0);
    gp_Pnt2d g2 = a.curve->crv->Value(a.curve->u1);
    gp_Pnt2d g3 = b.curve->crv->Value(b.curve->u0);
    gp_Pnt2d g4 = b.curve->crv->Value(b.curve->u1);
    V2 p1{g1.X(), g1.Y()}, p2{g2.X(), g2.Y()}, p3{g3.X(), g3.Y()}, p4{g4.X(), g4.Y()};
    if (dist2(p1, p3) < tol) { out = p1; return true; }
    if (dist2(p1, p4) < tol) { out = p1; return true; }
    if (dist2(p2, p3) < tol) { out = p2; return true; }
    if (dist2(p2, p4) < tol) { out = p2; return true; }
    return false;
}

// pointOnLine, reproducing `(near p1 or p2) or (dot >= 0 and dist < tol)`
static bool point_on_line(const V2& p1, const V2& p2, const V2& pt, double tol) {
    if (dist2(p2, pt) < tol || dist2(p1, pt) < tol) return true;
    double abx = p2.x - p1.x, aby = p2.y - p1.y;
    double apx = pt.x - p1.x, apy = pt.y - p1.y;
    double dot = abx * apx + aby * apy;
    if (!(dot >= 0.0)) return false;
    double cross = std::fabs(abx * apy - aby * apx);
    double ab_len = std::sqrt(abx * abx + aby * aby);
    return (cross / ab_len) < tol;
}

// -------------------------------------------------- control point machinery

// pointOnFace: BRep_Tool.Surface value transformed by the face location.
static inline void point_on_face(Ctx& c, double u, double v, double out[3]) {
    gp_Pnt p = c.face_surf->Value(u, v);
    p = p.Transformed(c.trsf);
    out[0] = p.X();
    out[1] = p.Y();
    out[2] = p.Z();
}

// getControlPointsFromApproximation, OCC-evaluation part only: returns the
// 109 sampled 3D points (rows 7..8 replaced through the boundary edge when
// present). The lstsq tail runs in Python for bitwise equality.
struct EdgeInfo {
    V2 start;
    double params[2];
    Handle(Geom2d_Curve) crv;
};

static std::vector<double> approx_fit_points(Ctx& c, const V2& v1, const V2& v2,
                                             const V2& v3, const EdgeInfo* edge) {
    const size_t n = c.uvs.size();
    std::vector<double> pts(n * 3);
    std::vector<V2> params(n);
    for (size_t i = 0; i < n; i++) {
        double u = c.uvs[i].x, v = c.uvs[i].y;
        double w = (1.0 - u) - v;
        params[i].x = v3.x * u + v2.x * v + v1.x * w;
        params[i].y = v3.y * u + v2.y * v + v1.y * w;
        point_on_face(c, params[i].x, params[i].y, &pts[i * 3]);
    }
    if (edge != nullptr) {
        // old_points[0] = params[edge_idx] with edge_idx == 2
        const V2& old0 = params[2];
        bool close = dist2(edge->start, old0) <= 1e-6;  // isClose tol
        // python `e0*2/3 + e1*1/3` is left-associative: ((e0*2)/3) + ((e1*1)/3)
        double t0, t1;
        if (close) {
            t0 = (edge->params[0] * 2.0) / 3.0 + (edge->params[1] * 1.0) / 3.0;
            t1 = (edge->params[0] * 1.0) / 3.0 + (edge->params[1] * 2.0) / 3.0;
        } else {
            t0 = (edge->params[0] * 1.0) / 3.0 + (edge->params[1] * 2.0) / 3.0;
            t1 = (edge->params[0] * 2.0) / 3.0 + (edge->params[1] * 1.0) / 3.0;
        }
        double ts[2] = {t0, t1};
        for (int k = 0; k < 2; k++) {
            gp_Pnt2d q = edge->crv->Value(ts[k]);
            point_on_face(c, q.X(), q.Y(), &pts[(7 + k) * 3]);  // rows 3+2*2 .. +2
        }
    }
    return pts;
}

// _conv_matrix(3, 3, invert): 28x16, exact binomial arithmetic (identical
// doubles to the scipy/numpy version; all intermediates are small integers).
static double binom_d(int nn, int kk) {
    if (kk < 0 || kk > nn) return 0.0;
    double r = 1.0;
    for (int i = 0; i < kk; i++) r = r * (nn - i) / (i + 1);
    return r;
}

static const std::array<double, 28 * 16>& conv_matrix(bool invert) {
    static std::array<double, 28 * 16> m_norm, m_inv;
    static bool init = false;
    if (!init) {
        const int m = 3, n = 3, degree = 6;
        for (int pass = 0; pass < 2; pass++) {
            auto& M = pass ? m_inv : m_norm;
            bool inv = pass != 0;
            M.fill(0.0);
            int row = 0;
            for (int s = degree; s >= 0; s--) {
                for (int b = 0; b <= s; b++) {
                    int a = s - b;
                    for (int j = 0; j <= a; j++) {
                        double caj = binom_d(a, j);
                        int k_lo = std::max(0, b - m + j);
                        int k_hi = std::min(b, n - a + j);
                        for (int k = k_lo; k <= k_hi; k++) {
                            double cc = caj * binom_d(b, k) *
                                        binom_d(m + n - a - b, m + k - j - b);
                            int col = inv ? (m - j) * (n + 1) + (n - k)
                                          : j * (n + 1) + k;
                            M[row * 16 + col] += cc;
                        }
                    }
                    row++;
                }
            }
            double d = binom_d(degree, n);  // 20
            for (auto& x : M) x = x / d;
        }
        init = true;
    }
    return invert ? m_inv : m_norm;
}

// getControlPointsFromRect: elevate patch to (3,3), read the 4x4 pole grid
// transformed by loc, weight fixup, rational rect->tri Bezier conversion.
static void rect_ctrl_from_patch(Ctx& c, const Handle(Geom_BezierSurface)& patch,
                                 std::array<double, 112>& out1,
                                 std::array<double, 112>& out2) {
    patch->Increase(3, 3);  // over-degree patches raise, as in Python
    double poles[16][4];
    for (int u = 0; u < 4; u++) {
        for (int v = 0; v < 4; v++) {
            gp_Pnt p = patch->Pole(u + 1, v + 1);
            double w = patch->Weight(u + 1, v + 1);
            p = p.Transformed(c.trsf);
            double* q = poles[u * 4 + v];
            q[0] = p.X();
            q[1] = p.Y();
            q[2] = p.Z();
            q[3] = w;
        }
    }
    double wsum = 0.0;
    for (int i = 0; i < 16; i++) wsum += poles[i][3];
    if (wsum < 1e-6)
        for (int i = 0; i < 16; i++) poles[i][3] = 1.0;

    // control_pts[..., :-1] *= control_pts[..., [-1]]
    for (int i = 0; i < 16; i++) {
        poles[i][0] *= poles[i][3];
        poles[i][1] *= poles[i][3];
        poles[i][2] *= poles[i][3];
    }
    const auto& M0 = conv_matrix(false);
    const auto& M1 = conv_matrix(true);
    for (int r = 0; r < 28; r++) {
        for (int col = 0; col < 4; col++) {
            double s0 = 0.0, s1 = 0.0;
            for (int k = 0; k < 16; k++) {
                s0 += M0[r * 16 + k] * poles[k][col];
                s1 += M1[r * 16 + k] * poles[k][col];
            }
            out1[r * 4 + col] = s0;
            out2[r * 4 + col] = s1;
        }
    }
    for (int r = 0; r < 28; r++) {
        double w1 = out1[r * 4 + 3], w2 = out2[r * 4 + 3];
        out1[r * 4 + 0] /= w1;
        out1[r * 4 + 1] /= w1;
        out1[r * 4 + 2] /= w1;
        out2[r * 4 + 0] /= w2;
        out2[r * 4 + 1] /= w2;
        out2[r * 4 + 2] /= w2;
    }
}

// make_rect: returns false when Python would return None (converter failure
// or zero patches); sets r.discarded in those cases, like the original.
static bool make_rect(Ctx& c, Rect& r, std::vector<Tri>& out) {
    double xmin, xmax, ymin, ymax;
    rect_boundary(r, xmin, xmax, ymin, ymax);
    std::unique_ptr<GeomConvert_BSplineSurfaceToBezierSurface> conv;
    try {
        conv.reset(new GeomConvert_BSplineSurfaceToBezierSurface(
            c.nurbs, xmin, xmax, ymin, ymax, 1e-4));
    } catch (const Standard_Failure&) {
        r.discarded = true;
        return false;
    }
    int U = conv->NbUPatches();
    int V = conv->NbVPatches();
    if (U == 0 || V == 0) {
        r.discarded = true;
        return false;
    }
    for (int i = 0; i < U; i++) {
        for (int j = 0; j < V; j++) {
            Handle(Geom_BezierSurface) patch = conv->Patch(i + 1, j + 1);
            Tri t1, t2;
            rect_ctrl_from_patch(c, patch, t1.ctrl, t2.ctrl);
            t1.v1 = r.p[0]; t1.v2 = r.p[1]; t1.v3 = r.p[2];
            t2.v1 = r.p[1]; t2.v2 = r.p[2]; t2.v3 = r.p[3];
            out.push_back(std::move(t1));
            out.push_back(std::move(t2));
        }
    }
    return true;
}

// make_triangles: single triangle with a boundary edge on (ep0 -> ep2),
// control points deferred to the Python lstsq tail.
static std::vector<Tri> make_triangles(Ctx& c, const Handle(Geom2d_Curve)& crv,
                                       const double interval[2], const V2& ep0,
                                       const V2& ep1, const V2& ep2) {
    EdgeInfo edge;
    edge.start = ep0;
    edge.params[0] = interval[0];
    edge.params[1] = interval[1];
    edge.crv = crv;
    Tri tri;
    tri.mode = 1;
    tri.v1 = ep0; tri.v2 = ep1; tri.v3 = ep2;
    tri.fitpts = approx_fit_points(c, ep0, ep1, ep2, &edge);
    return {std::move(tri)};
}

// make_boundary_rect: interior triangle + boundary triangle.
static std::vector<Tri> make_boundary_rect(Ctx& c, const Handle(Geom2d_Curve)& crv,
                                           const double interval[2], const V2& ep0,
                                           const V2& ep1, const V2& ep2,
                                           const V2& ep3) {
    Tri t1;
    t1.mode = 1;
    t1.v1 = ep0; t1.v2 = ep1; t1.v3 = ep2;
    t1.fitpts = approx_fit_points(c, ep0, ep1, ep2, nullptr);

    EdgeInfo edge;
    edge.start = ep0;
    edge.params[0] = interval[0];
    edge.params[1] = interval[1];
    edge.crv = crv;
    Tri t2;
    t2.mode = 1;
    t2.v1 = ep0; t2.v2 = ep2; t2.v3 = ep3;
    t2.fitpts = approx_fit_points(c, ep0, ep2, ep3, &edge);

    std::vector<Tri> res;
    res.push_back(std::move(t1));
    res.push_back(std::move(t2));
    return res;
}

// ------------------------------------------------------- boundary splitting

static void set_leaf_make_rect(Ctx& c, Rect& r) {
    std::vector<Tri> tris;
    if (make_rect(c, r, tris)) {
        r.leaf_state = 2;
        r.tris = std::move(tris);
    } else {
        r.leaf_state = 0;  // leaf_info = None (r.discarded set by make_rect)
    }
}

// splitBoundaryRectangle (tol = 9e-3, REPORT_ERROR suppressed by caller)
static void split_boundary_rectangle(Ctx& c, Rect& r) {
    const double tol = 9e-3;

    if (r.leaf_state == 0 || (r.leaf_state == 1 && r.isects.empty())) {
        set_leaf_make_rect(c, r);
        return;
    }
    if (r.leaf_state == 2) return;  // splitting has been done
    if (r.isects.size() > 1) return;  // "too many intersections!" (warning)

    Isect& isect = r.isects[0];
    if (isect.nbpoints != 2) {
        if (isect.nbpoints == 1) {
            set_leaf_make_rect(c, r);
            return;
        }
        // deleteMiddlePoints: keep first and last
        V2 pa = isect.pts.front(), pb = isect.pts.back();
        double qa = isect.params.front(), qb = isect.params.back();
        isect.pts = {pa, pb};
        isect.params = {qa, qb};
        isect.nbpoints = 2;
    }

    // status bitmask against edges (p0,p1),(p1,p3),(p3,p2),(p2,p0)
    const int E[4][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0}};
    int status[4] = {0, 0, 0, 0};
    for (int pi = 0; pi < 2; pi++) {
        for (int e = 0; e < 4; e++) {
            if (point_on_line(r.p[E[e][0]], r.p[E[e][1]], isect.pts[pi], tol))
                status[e] += (1 << pi);
        }
    }

    int hit_flag = 0;
    for (int e = 0; e < 4; e++) {
        hit_flag |= status[e];
        if (status[e] == 0x3) {
            // "intersection points are on the same line"
            set_leaf_make_rect(c, r);
            return;
        }
    }
    if (hit_flag != 0x03) return;  // "not enough intersection points ..."

    const int CW[4] = {0, 1, 3, 2};

    try {
        for (int idx = 0; idx < 4; idx++) {
            int item = status[idx];
            if (item <= 0) continue;
            bool corner_point = false;
            bool reverse = (item == 0x02);
            double pr[2];
            if (reverse) {
                pr[0] = isect.params[1];
                pr[1] = isect.params[0];
            } else {
                pr[0] = isect.params[0];
                pr[1] = isect.params[1];
            }
            const V2& A = isect.pts[reverse ? 1 : 0];   // Points[reverse]
            const V2& B = isect.pts[reverse ? 0 : 1];   // Points[not reverse]

            if (status[(idx + 1) % 4] > 0) {
                if (item != status[(idx + 1) % 4]) {
                    // intersection on adjacent edges
                    int end_index = CW[(idx + 1) % 4];
                    V2 end_point = r.p[end_index];
                    int v = vis_status(c, end_point);
                    V2 oppo{(isect.pts[0].x + isect.pts[1].x) - end_point.x,
                            (isect.pts[0].y + isect.pts[1].y) - end_point.y};
                    if (v == 1 || v == 3) {
                        rect_split(r, &oppo, tol);
                        r.is_leaf = false;
                        r.subs[end_index].tris =
                            make_triangles(c, isect.curve->crv, pr, A, oppo, B);
                        r.subs[end_index].leaf_state = 2;
                    } else {
                        rect_split(r, &oppo, tol);
                        r.is_leaf = false;
                        for (int k = 0; k < 4; k++) r.subs[k].discarded = true;
                        r.subs[end_index].discarded = false;
                        r.subs[end_index].tris = make_triangles(
                            c, isect.curve->crv, pr, A, r.p[end_index], B);
                        r.subs[end_index].leaf_state = 2;
                    }
                    break;
                } else {
                    corner_point = true;
                }
            }
            if (item == status[(idx + 3) % 4]) corner_point = true;
            if (!corner_point && status[(idx + 2) % 4] > 0 &&
                item != status[(idx + 2) % 4]) {
                // intersection on opposite edges
                int end_index = CW[(idx + 1) % 4];
                V2 end_point = r.p[end_index];
                int v = vis_status(c, end_point);
                if (v == 1 || v == 3) continue;  // quirk Q7
                int other_index = CW[(idx + 2) % 4];
                V2 other_point = r.p[other_index];
                V2 point1 = isect.pts[item - 1];
                V2 point2 = isect.pts[item % 2];
                double d = dist2(point1, end_point) - dist2(point2, other_point);
                if (d > tol) {
                    V2 oppo{(point2.x - other_point.x) + end_point.x,
                            (point2.y - other_point.y) + end_point.y};
                    rect_split(r, &oppo, tol);
                    r.is_leaf = false;
                    int tgt = CW[(idx + 3) % 4];
                    r.subs[tgt].tris = make_triangles(c, isect.curve->crv, pr,
                                                      point1, oppo, point2);
                    r.subs[tgt].leaf_state = 2;
                } else if (d < -tol) {
                    V2 oppo{(point1.x - end_point.x) + other_point.x,
                            (point1.y - end_point.y) + other_point.y};
                    rect_split(r, &oppo, tol);
                    r.is_leaf = false;
                    int tgt = CW[idx];
                    r.subs[tgt].tris = make_triangles(c, isect.curve->crv, pr,
                                                      point1, oppo, point2);
                    r.subs[tgt].leaf_state = 2;
                } else {
                    V2 oppo = point2;
                    rect_split(r, &oppo, tol);
                    r.is_leaf = false;
                    r.subs[end_index].tris =
                        make_boundary_rect(c, isect.curve->crv, pr, point1,
                                           end_point, other_point, point2);
                    r.subs[end_index].leaf_state = 2;
                    for (int k = 0; k < 4; k++)
                        if (k != end_index) r.subs[k].discarded = true;
                }
                break;
            }
        }
    } catch (...) {
        // "error in splitBoundaryRectangle" -- swallowed, leaf keeps isects
        return;
    }

    if (!r.is_leaf) {
        for (int k = 0; k < 4; k++)
            if (!r.subs[k].discarded) split_boundary_rectangle(c, r.subs[k]);
    }
    // else: warning "no triangles are generated" only
}

// HandleLeaves
static void handle_leaves(Ctx& c, Rect& r) {
    if (r.discarded) return;
    if (r.is_leaf) {
        split_boundary_rectangle(c, r);
    } else {
        for (int k = 0; k < 4; k++) handle_leaves(c, r.subs[k]);
    }
}

// ------------------------------------------------------------ splitRectangle

static void split_rectangle(Ctx& c, Rect& root, int max_split, double chord_tol,
                            double dtol) {
    std::vector<Rect*> stack{&root};
    while (!stack.empty()) {
        Rect* rect = stack.back();
        stack.pop_back();
        if (rect->discarded) continue;

        std::vector<Isect> isects;
        isects.reserve(c.curves.size());
        bool split_flag = false;
        bool has_sp = false;
        V2 split_point;

        for (const Curve& cur : c.curves) {
            Isect local = intersect_rect_curve(c, *rect, cur, dtol);
            Isect* ref = &local;
            if (local.nbpoints > 0) {
                isects.push_back(std::move(local));
                ref = &isects.back();
            }
            if (curve_in_rect(cur, *rect, *ref, dtol) || ref->nbpoints > 2 ||
                ref->two_same_line ||
                !chord_error_check_in_rect(*ref, *rect, chord_tol)) {
                split_flag = true;
                break;
            }
        }

        if (isects.size() > 1) {
            split_flag = true;
            if (isects.size() == 2) {
                V2 sp;
                if (has_same_point(isects[0], isects[1], dtol, sp) &&
                    rect_contains(*rect, sp, true, dtol) &&
                    !rect_is_corner(*rect, sp, dtol)) {
                    has_sp = true;
                    split_point = sp;
                }
            }
        }

        if (split_flag) {
            if (rect->level < max_split) {
                rect_split(*rect, has_sp ? &split_point : nullptr, dtol);
                rect->is_leaf = false;
                for (int k = 0; k < 4; k++) stack.push_back(&rect->subs[k]);
            } else {
                rect->leaf_state = 1;
                rect->isects = std::move(isects);
            }
        } else {
            if (isects.empty() && !any_corners_in_face(c, *rect) &&
                vis_status(c, rect_center(*rect)) == 1) {
                rect->discarded = true;
                continue;
            }
            rect->leaf_state = 1;
            rect->isects = std::move(isects);
        }
    }
}

// --------------------------------------------------------------- collection

struct OutRec {
    bool in = false;
    Tri tri;
};

static void collect_tris(Ctx& c, Rect& r, std::vector<OutRec>& out) {
    if (r.discarded) {
        if (rect_area(r) > 1e-5) {
            std::vector<Tri> tris;
            if (!make_rect(c, r, tris))
                throw std::runtime_error(
                    "make_rect returned None while collecting discarded rect "
                    "(python TypeError)");
            for (Tri& t : tris) out.push_back(OutRec{false, std::move(t)});
        }
        return;
    }
    if (r.is_leaf) {
        if (r.leaf_state == 2) {
            for (const Tri& t : r.tris) out.push_back(OutRec{true, t});
        } else if (r.leaf_state == 1) {
            if (!r.isects.empty()) {
                if (rect_area(r) > 1e-5) {
                    std::vector<Tri> tris;
                    if (!make_rect(c, r, tris))
                        throw std::runtime_error(
                            "make_rect returned None while collecting leaf "
                            "with intersections (python TypeError)");
                    for (Tri& t : tris) out.push_back(OutRec{false, std::move(t)});
                }
                return;
            }
            // empty list: python for-loop body never runs
        } else {
            throw std::runtime_error(
                "leaf_info is None on non-discarded leaf (python TypeError)");
        }
    } else {
        for (int k = 0; k < 4; k++) collect_tris(c, r.subs[k], out);
    }
}

// ------------------------------------------------------------------- driver

py::dict process_face(uintptr_t face_addr, uintptr_t surf_addr,
                      std::vector<double> uknots, std::vector<double> vknots,
                      std::vector<std::tuple<uintptr_t, double, double>> curves_in,
                      py::array_t<double, py::array::c_style | py::array::forcecast> uvs_in) {
    try {
        Ctx c;
        c.face = *reinterpret_cast<TopoDS_Face*>(face_addr);
        c.nurbs = Handle(Geom_BSplineSurface)(
            reinterpret_cast<Geom_BSplineSurface*>(surf_addr));
        c.face_surf = BRep_Tool::Surface(c.face, c.face_loc);
        c.trsf = c.face_loc.Transformation();
        c.classifier.reset(new BRepTopAdaptor_FClass2d(c.face, 1e-9));
        c.reversed_face = (c.face.Orientation() == TopAbs_REVERSED);

        for (auto& t : curves_in) {
            Curve cur;
            cur.crv = Handle(Geom2d_Curve)(
                reinterpret_cast<Geom2d_Curve*>(std::get<0>(t)));
            cur.u0 = std::get<1>(t);
            cur.u1 = std::get<2>(t);
            c.curves.push_back(std::move(cur));
        }

        auto uv = uvs_in.unchecked<2>();
        if (uv.shape(1) != 2) throw std::runtime_error("uvs must be (n,2)");
        c.uvs.resize(uv.shape(0));
        for (py::ssize_t i = 0; i < uv.shape(0); i++)
            c.uvs[i] = V2{uv(i, 0), uv(i, 1)};

        const int U = static_cast<int>(uknots.size()) - 1;
        const int V = static_cast<int>(vknots.size()) - 1;

        std::vector<OutRec> out;
        for (int u = 0; u < U; u++) {
            for (int v = 0; v < V; v++) {
                Rect root;
                root.p[0] = V2{uknots[u], vknots[v]};
                root.p[1] = V2{uknots[u + 1], vknots[v]};
                root.p[2] = V2{uknots[u], vknots[v + 1]};
                root.p[3] = V2{uknots[u + 1], vknots[v + 1]};
                split_rectangle(c, root, 5, 0.7, 1e-4);
                handle_leaves(c, root);
                collect_tris(c, root, out);
            }
        }

        const py::ssize_t n = static_cast<py::ssize_t>(out.size());
        if (n == 0)
            throw std::runtime_error("no triangles collected (np.stack raises)");

        py::array_t<bool> in_mask(n);
        py::array_t<double> ctrl({n, py::ssize_t(28), py::ssize_t(4)});
        py::array_t<double> normals({n, py::ssize_t(3)});
        auto im = in_mask.mutable_unchecked<1>();
        auto ct = ctrl.mutable_unchecked<3>();
        auto nm = normals.mutable_unchecked<2>();

        std::vector<py::ssize_t> fit_rows;
        for (py::ssize_t i = 0; i < n; i++)
            if (out[i].tri.mode == 1) fit_rows.push_back(i);

        py::array_t<py::ssize_t> fit_idx(py::ssize_t(fit_rows.size()));
        py::array_t<double> fitpts(
            {py::ssize_t(fit_rows.size()),
             py::ssize_t(c.uvs.size()), py::ssize_t(3)});
        auto fi = fit_idx.mutable_unchecked<1>();
        auto fp = fitpts.mutable_unchecked<3>();

        py::ssize_t slot = 0;
        for (py::ssize_t i = 0; i < n; i++) {
            const Tri& t = out[i].tri;
            im(i) = out[i].in;

            // getTriNormal: face.normal at the uv centroid (occwl semantics,
            // no location transform, negated for reversed faces)
            double x = ((t.v1.x + t.v2.x) + t.v3.x) / 3.0;
            double y = ((t.v1.y + t.v2.y) + t.v3.y) / 3.0;
            GeomLProp_SLProps props(c.face_surf, x, y, 1, 1e-9);
            if (!props.IsNormalDefined()) {
                nm(i, 0) = 0.0;
                nm(i, 1) = 0.0;
                nm(i, 2) = 0.0;
            } else {
                gp_Dir d = props.Normal();
                double nx = d.X(), ny = d.Y(), nz = d.Z();
                if (c.reversed_face) {
                    nx = -nx;
                    ny = -ny;
                    nz = -nz;
                }
                nm(i, 0) = nx;
                nm(i, 1) = ny;
                nm(i, 2) = nz;
            }

            if (t.mode == 0) {
                for (int r2 = 0; r2 < 28; r2++)
                    for (int c2 = 0; c2 < 4; c2++)
                        ct(i, r2, c2) = t.ctrl[r2 * 4 + c2];
            } else {
                for (int r2 = 0; r2 < 28; r2++) {
                    ct(i, r2, 0) = 0.0;
                    ct(i, r2, 1) = 0.0;
                    ct(i, r2, 2) = 0.0;
                    ct(i, r2, 3) = 1.0;  // weight column (python appends ones)
                }
                fi(slot) = i;
                for (size_t r2 = 0; r2 < c.uvs.size(); r2++) {
                    fp(slot, py::ssize_t(r2), 0) = t.fitpts[r2 * 3 + 0];
                    fp(slot, py::ssize_t(r2), 1) = t.fitpts[r2 * 3 + 1];
                    fp(slot, py::ssize_t(r2), 2) = t.fitpts[r2 * 3 + 2];
                }
                slot++;
            }
        }

        py::dict res;
        res["in_mask"] = in_mask;
        res["ctrl"] = ctrl;
        res["normals"] = normals;
        res["fit_idx"] = fit_idx;
        res["fitpts"] = fitpts;
        return res;
    } catch (const Standard_Failure& f) {
        const char* msg = f.GetMessageString();
        throw std::runtime_error(std::string("OCC failure: ") +
                                 (msg ? msg : "<no message>"));
    }
}

// ================================================================= phase 2

// randn_uvgrid evaluation stage (RNG stays in Python; uv01 is the float32
// np.random.uniform draw). Replicates occwl Interval.interpolate exactly:
// `(1.0 - t) * a + t * b`, all in float64 (numpy scalar-scalar promotion
// widens the float32 t before any arithmetic).
py::dict sample_face(uintptr_t face_addr,
                     py::array_t<float, py::array::c_style | py::array::forcecast> uv01,
                     double umin, double umax, double vmin, double vmax) {
    try {
        Ctx c;
        c.face = *reinterpret_cast<TopoDS_Face*>(face_addr);
        c.face_surf = BRep_Tool::Surface(c.face, c.face_loc);
        c.trsf = c.face_loc.Transformation();
        c.classifier.reset(new BRepTopAdaptor_FClass2d(c.face, 1e-9));
        c.reversed_face = (c.face.Orientation() == TopAbs_REVERSED);

        auto uv = uv01.unchecked<2>();
        if (uv.shape(1) != 2) throw std::runtime_error("uv01 must be (n,2)");
        const py::ssize_t n = uv.shape(0);

        // Box(np.array([umin, vmin])); encompass_point([umax, vmax])
        double au = umin, bu = umin, av = vmin, bv = vmin;
        if (au > umax) au = umax;
        if (bu < umax) bu = umax;
        if (av > vmax) av = vmax;
        if (bv < vmax) bv = vmax;

        py::array_t<double> points({n, py::ssize_t(3)});
        py::array_t<double> normals({n, py::ssize_t(3)});
        py::array_t<int64_t> vis(n);
        auto P = points.mutable_unchecked<2>();
        auto N = normals.mutable_unchecked<2>();
        auto S = vis.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < n; i++) {
            // scalar-scalar promotion: np.float32 with python float promotes
            // to float64, so the whole interpolation is double arithmetic on
            // the exactly-widened float32 sample
            double tu = static_cast<double>(uv(i, 0));
            double tv = static_cast<double>(uv(i, 1));
            double u = (1.0 - tu) * au + tu * bu;
            double v = (1.0 - tv) * av + tv * bv;

            double p3[3];
            point_on_face(c, u, v, p3);
            P(i, 0) = p3[0];
            P(i, 1) = p3[1];
            P(i, 2) = p3[2];

            // bezier2.normalOnFace (loc transform result discarded, quirk Q8)
            GeomLProp_SLProps props(c.face_surf, u, v, 1, 1e-9);
            if (!props.IsNormalDefined()) {
                N(i, 0) = 0.0;
                N(i, 1) = 0.0;
                N(i, 2) = 0.0;
            } else {
                gp_Dir d = props.Normal();
                double nx = d.X(), ny = d.Y(), nz = d.Z();
                if (c.reversed_face) {
                    nx = -nx;
                    ny = -ny;
                    nz = -nz;
                }
                N(i, 0) = nx;
                N(i, 1) = ny;
                N(i, 2) = nz;
            }

            S(i) = static_cast<int64_t>(vis_status(c, V2{u, v}));
        }

        py::dict res;
        res["points"] = points;
        res["normals"] = normals;
        res["vis"] = vis;
        return res;
    } catch (const Standard_Failure& f) {
        const char* msg = f.GetMessageString();
        throw std::runtime_error(std::string("OCC failure: ") +
                                 (msg ? msg : "<no message>"));
    }
}

// ---- rotation / normalization loop (convertFaceToTriangles epilogue) ----

// numpy minimum/maximum.reduce semantics (NaN propagates via the accumulator)
static inline double np_min(double acc, double v) {
    return (acc < v || std::isnan(acc)) ? acc : v;
}
static inline double np_max(double acc, double v) {
    return (acc > v || std::isnan(acc)) ? acc : v;
}

// rotation_matrix_from_axis_angle: R = eye + sin(angle)*K + (1-cos(angle))*K@K
static void rot_from_axis_angle(const double axis_in[3], double angle,
                                double R[9]) {
    double n = std::sqrt((axis_in[0] * axis_in[0] + axis_in[1] * axis_in[1]) +
                         axis_in[2] * axis_in[2]);
    double a0 = axis_in[0] / n, a1 = axis_in[1] / n, a2 = axis_in[2] / n;
    double K[9] = {0.0, -a2, a1, a2, 0.0, -a0, -a1, a0, 0.0};
    double s = std::sin(angle);
    double c1 = 1.0 - std::cos(angle);
    double KK[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            KK[i * 3 + j] = (K[i * 3 + 0] * K[0 + j] + K[i * 3 + 1] * K[3 + j]) +
                            K[i * 3 + 2] * K[6 + j];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double eye = (i == j) ? 1.0 : 0.0;
            R[i * 3 + j] = (eye + s * K[i * 3 + j]) + c1 * KK[i * 3 + j];
        }
    }
}

// np.isclose(a, b) with default rtol=1e-5, atol=1e-8 (finite b)
static inline bool np_isclose(double a, double b) {
    return std::fabs(a - b) <= (1e-8 + 1e-5 * std::fabs(b));
}

// rotation_matrix_to_z_axis (solid_to_triangles2.py:33), exact op order
static void rot_to_z(const double v[3], double R[9]) {
    double nv = std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]);
    if (!np_isclose(nv, 1.0))
        throw std::runtime_error("v must be unit vector (python AssertionError)");
    // dot = np.dot(v, [0,0,1]) with ddot accumulation order
    double dot = (v[0] * 0.0 + v[1] * 0.0) + v[2] * 1.0;
    if (np_isclose(dot, 1.0)) {
        for (int i = 0; i < 9; i++) R[i] = 0.0;
        R[0] = R[4] = R[8] = 1.0;
        return;
    }
    if (np_isclose(dot, -1.0)) {
        double axis[3];
        if (!np_isclose(v[0], 1.0)) {
            axis[0] = 1.0; axis[1] = 0.0; axis[2] = 0.0;
        } else {
            axis[0] = 0.0; axis[1] = 1.0; axis[2] = 0.0;
        }
        double ad = (axis[0] * v[0] + axis[1] * v[1]) + axis[2] * v[2];
        for (int k = 0; k < 3; k++) axis[k] = axis[k] - ad * v[k];
        double an = std::sqrt((axis[0] * axis[0] + axis[1] * axis[1]) +
                              axis[2] * axis[2]);
        for (int k = 0; k < 3; k++) axis[k] = axis[k] / an;
        rot_from_axis_angle(axis, M_PI, R);
        return;
    }
    // axis = np.cross(v, [0,0,1]) (numpy component formula, literal zeros)
    double b0 = 0.0, b1 = 0.0, b2 = 1.0;
    double ax0 = v[1] * b2 - v[2] * b1;
    double ax1 = v[2] * b0 - v[0] * b2;
    double ax2 = v[0] * b1 - v[1] * b0;
    double an = std::sqrt((ax0 * ax0 + ax1 * ax1) + ax2 * ax2);
    double axis[3] = {ax0 / an, ax1 / an, ax2 / an};
    double angle = std::acos(dot);
    rot_from_axis_angle(axis, angle, R);
}

// The per-triangle rotate + cumulative re-center/re-scale loop from
// convertFaceToTriangles (rotated_and_normalized branch), verbatim
// semantics: iteration i rotates nodes[i] in the CURRENT cumulative frame,
// then re-centers and re-scales ALL nodes by the bbox of the mixed
// (rotated 0..i, unrotated i+1..) set. O(T^2) by construction; kept
// sequential for bit-fidelity. Mutates `nodes` in place, returns the last
// iteration's (center, scale) for new_feature[:, 3:7].
py::tuple rotate_normalize(py::array_t<double, py::array::c_style> nodes,
                           py::array_t<double, py::array::c_style | py::array::forcecast> tri_normals) {
    auto nb = nodes.mutable_unchecked<3>();
    auto tn = tri_normals.unchecked<2>();
    const py::ssize_t T = nb.shape(0);
    const py::ssize_t P = nb.shape(1);
    if (nb.shape(2) != 4 || tn.shape(0) != T || tn.shape(1) != 3)
        throw std::runtime_error("rotate_normalize: bad shapes");

    double cen[3] = {0.0, 0.0, 0.0};
    double scale_out = 1.0;

    for (py::ssize_t i = 0; i < T; i++) {
        double v[3] = {tn(i, 0), tn(i, 1), tn(i, 2)};
        double R[9];
        rot_to_z(v, R);

        // nodes[i][..., :3] = (R @ nodes[i][..., :3].T).T
        for (py::ssize_t p = 0; p < P; p++) {
            double a = nb(i, p, 0), b = nb(i, p, 1), c2 = nb(i, p, 2);
            nb(i, p, 0) = (R[0] * a + R[1] * b) + R[2] * c2;
            nb(i, p, 1) = (R[3] * a + R[4] * b) + R[5] * c2;
            nb(i, p, 2) = (R[6] * a + R[7] * b) + R[8] * c2;
        }

        // bbox over ALL nodes (numpy min/max with NaN-sticky accumulator)
        double mn[3] = {nb(0, 0, 0), nb(0, 0, 1), nb(0, 0, 2)};
        double mx[3] = {nb(0, 0, 0), nb(0, 0, 1), nb(0, 0, 2)};
        for (py::ssize_t j = 0; j < T; j++) {
            for (py::ssize_t p = 0; p < P; p++) {
                for (int k = 0; k < 3; k++) {
                    double val = nb(j, p, k);
                    mn[k] = np_min(mn[k], val);
                    mx[k] = np_max(mx[k], val);
                }
            }
        }

        double diag[3] = {mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]};
        // python builtin max(d0, d1, d2)
        double mxd = diag[0];
        if (diag[1] > mxd) mxd = diag[1];
        if (diag[2] > mxd) mxd = diag[2];
        double scale = 2.0 / mxd;
        double center[3] = {0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]),
                            0.5 * (mn[2] + mx[2])};

        for (py::ssize_t j = 0; j < T; j++)
            for (py::ssize_t p = 0; p < P; p++)
                for (int k = 0; k < 3; k++) nb(j, p, k) -= center[k];
        if (!std::isnan(scale)) {
            for (py::ssize_t j = 0; j < T; j++)
                for (py::ssize_t p = 0; p < P; p++)
                    for (int k = 0; k < 3; k++) nb(j, p, k) *= scale;
        } else {
            scale = 1.0;  // python `scale = 1`
        }
        cen[0] = center[0];
        cen[1] = center[1];
        cen[2] = center[2];
        scale_out = scale;
    }
    return py::make_tuple(py::make_tuple(cen[0], cen[1], cen[2]), scale_out);
}

}  // namespace

PYBIND11_MODULE(brt_core, m) {
    m.doc() = "C++ port of the BRT tokenizer face hot path (see README.md)";
    m.def("process_face", &process_face, py::arg("face_addr"),
          py::arg("surf_addr"), py::arg("uknots"), py::arg("vknots"),
          py::arg("curves"), py::arg("uvs"));
    m.def("sample_face", &sample_face, py::arg("face_addr"), py::arg("uv01"),
          py::arg("umin"), py::arg("umax"), py::arg("vmin"), py::arg("vmax"));
    m.def("rotate_normalize", &rotate_normalize, py::arg("nodes"),
          py::arg("tri_normals"));
}
