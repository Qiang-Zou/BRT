import argparse
import pathlib
import pickle
import traceback

from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper

import numpy as np
import torch
from occwl.compound import Compound
from tqdm import tqdm
from multiprocessing.pool import Pool
from itertools import repeat
import signal
from occwl.entity_mapper import EntityMapper

def build_data(solid,shape_att,edge_fn,*args,**kwargs):
    mapper = EntityMapper(solid)
    # faces_data=[]
    labels=[]

    adj_faces_indices=[]
    faces_wire=[]

    wire_edges=[]

    edge_dict={}

    # sorted_face=solid.faces()
    face_mapper=lambda f:mapper.face_index(f)

    for face in solid.faces():

        # points = uvgrid(
        #     face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        # )
        # normals = uvgrid(
        #     face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        # )
        # visibility_status = uvgrid(
        #     face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        # )
        # mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # # Concatenate channel-wise to form face feature tensor
        # face_feat = np.concatenate((points, normals, mask), axis=-1)
        # face_feature.append(face_feat)

        labels.append(int(shape_att[face]['name']))

        for wire in face.wires():
            edges=[]
            for edge in wire.ordered_edges():
                if not edge.has_curve():
                    continue
                index=mapper.oriented_edge_index(edge)

                # if not edge.has_curve():
                #     raise RuntimeError("No curve found for edge")
                connected_faces = list(solid.faces_from_edge(edge))
                if len(connected_faces) < 2:
                    pass
                elif len(connected_faces) == 2:
                    left_face, right_face = find_left_and_right_faces(edge,connected_faces)
                    if left_face is None or right_face is None:
                        # continue
                        raise RuntimeError("Expected a manifold, an edge must be incident on one/two faces")
                else:
                    raise RuntimeError("Expected a manifold, an edge must be incident on one/two faces")

                edge_dict[index]={'last':None if len(edges)==0 else edges[-1],
                                  'face':face_mapper(face)}
                edges.append(edge)
            edge_dict[mapper.oriented_edge_index(edges[0])]['last']=edges[-1]

    edges_feature=[]
    for face in solid.faces():
        # face_index=mapper.face_index(face)
        face_index=face_mapper(face)

        wires=[]
        faces_wire.append(wires)

        adj_faces=[]
        adj_faces_indices.append(adj_faces)

        for wire in face.wires():
            wires.append(len(wire_edges))
            wire_data=[]
            wire_edges.append(wire_data)
            cnt=0

            for edge in wire.ordered_edges():
                if not edge.has_curve():
                    continue
                index=mapper.oriented_edge_index(edge)

                # Compute U-grids
                # points = ugrid(edge, method="point", num_u=curv_num_u_samples)
                # tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
                # Concatenate channel-wise to form edge feature tensor

                # try:
                #     assert len(points.shape)==2
                #     assert len(tangents.shape)==2
                # except AssertionError:
                #     print(points.shape)
                #     print(tangents.shape)
                #     raise

                # edge_feat = np.concatenate((points, tangents), axis=-1)
                edge_feat=edge_fn(edge)
                wire_data.append(cnt)
                cnt+=1
                edges_feature.append(edge_feat)


                e=edge

                try:
                    while True:
                        reversed_edge=e.reversed_edge()
                        index=mapper.oriented_edge_index(reversed_edge)
                        face_belongs=edge_dict[index]['face']

                        if face_belongs==face_index:
                            break
                        if face_belongs not in adj_faces:
                            adj_faces.append(face_belongs)

                        e=edge_dict[index]['last']
                except KeyError:
                    pass

    # face_feat=np.array(face_feature,dtype=np.float32)
    # edge_feat=torch.from_numpy(np.stack(edges_feature,dtype=np.float32))
    # face_feat,edge_feat=center_and_scale(face_feat,edge_feat)
    if len(edges_feature)==0:
        raise RuntimeError("No edges found in solid")

    return {
            'edge':edges_feature,
            # 'face':face_feat,
            'edge_index':wire_edges,
            'wire_index':faces_wire,
            'adj_face_index':adj_faces_indices,
            'label':torch.tensor(labels,dtype=torch.long),
            }

def build_data_no_label(solid,edge_fn,*args,**kwargs):
    mapper = EntityMapper(solid)

    adj_faces_indices=[]
    faces_wire=[]

    wire_edges=[]

    edge_dict={}

    # sorted_face=solid.faces()
    face_mapper=lambda f:mapper.face_index(f)

    for face in solid.faces():

        for wire in face.wires():
            edges=[]
            for edge in wire.ordered_edges():
                if not edge.has_curve():
                    continue
                index=mapper.oriented_edge_index(edge)

                # if not edge.has_curve():
                #     raise RuntimeError("No curve found for edge")
                connected_faces = list(solid.faces_from_edge(edge))
                if len(connected_faces) < 2:
                    pass
                elif len(connected_faces) == 2:
                    left_face, right_face = find_left_and_right_faces(edge,connected_faces)
                    if left_face is None or right_face is None:
                        # continue
                        raise RuntimeError("Expected a manifold, an edge must be incident on one/two faces")
                else:
                    raise RuntimeError("Expected a manifold, an edge must be incident on one/two faces")

                edge_dict[index]={'last':None if len(edges)==0 else edges[-1],
                                  'face':face_mapper(face)}
                edges.append(edge)
            edge_dict[mapper.oriented_edge_index(edges[0])]['last']=edges[-1]

    edges_feature=[]
    for face in solid.faces():
        # face_index=mapper.face_index(face)
        face_index=face_mapper(face)

        wires=[]
        faces_wire.append(wires)

        adj_faces=[]
        adj_faces_indices.append(adj_faces)

        for wire in face.wires():
            wires.append(len(wire_edges))
            wire_data=[]
            wire_edges.append(wire_data)
            cnt=0

            for edge in wire.ordered_edges():
                if not edge.has_curve():
                    continue
                index=mapper.oriented_edge_index(edge)

                edge_feat=edge_fn(edge)
                wire_data.append(cnt)
                cnt+=1
                edges_feature.append(edge_feat)


                e=edge

                try:
                    while True:
                        reversed_edge=e.reversed_edge()
                        index=mapper.oriented_edge_index(reversed_edge)
                        face_belongs=edge_dict[index]['face']

                        if face_belongs==face_index:
                            break
                        if face_belongs not in adj_faces:
                            adj_faces.append(face_belongs)

                        e=edge_dict[index]['last']
                except KeyError:
                    pass

    if len(edges_feature)==0:
        raise RuntimeError("No edges found in solid")

    return {
            'edge':edges_feature,
            # 'face':face_feat,
            'edge_index':wire_edges,
            'wire_index':faces_wire,
            'adj_face_index':adj_faces_indices,
            }

def find_left_and_right_faces(self, faces):
    """
    Given a list of 1 or 2 faces which are adjacent to this edge,
    we want to return the left and right face when looking from
    outside the solid.

                    Edge direction
                        ^
                        |
                Left      |   Right
                face      |   face
                        |

    In the case of a cylinder the left and right face will be
    the same.

    Args:
        faces (list(occwl.face.Face): The faces

    Returns:
        occwl.face.Face, occwl.face.Face: The left and then right face
        or
        None, None if the left and right faces cannot be found
    """
    assert len(faces) > 0
    face1 = faces[0]
    if len(faces) == 1:
        face2 = faces[0]
    else:
        face2 = faces[1]

    if face1.is_left_of(self):
        # In some cases (like a cylinder) the left and right faces
        # of the edge are the same face
        if face1 != face2:
            if face2.is_left_of(self):
                return None, None
        left_face = face1
        right_face = face2
    else:
        if not face2.is_left_of(self):
            return None, None
        left_face = face2
        right_face = face1

    return left_face, right_face

def process_one_file(arguments):
    fn, args = arguments
    fn_stem = fn.stem
    output_path_ = pathlib.Path(args.output)

    output_path=str(output_path_.joinpath(fn_stem + ".bin"))
    # if os.path.exists(output_path):
    #     print(output_path," file exists")
    #     return

    compound, shape_att = Compound.load_step_with_attributes(fn)
    try:
        if args.no_label:
            face_data = build_data_no_label(
                next(compound.solids()),args.edge_fn
            )
        elif args.genlabel:
            solid=next(compound.solids())
            labels=[shape_att[face]['name'] for face in solid.faces()]
            output_path=str(output_path_.joinpath(fn_stem + ".seg"))
            with open(output_path,'w') as f:
                for label in labels:
                    f.write(str(label)+'\n')
            return
        else:
            solid=next(compound.solids())
            # solid = load_step(fn)[0]  # Assume there's one solid per file
            face_data = build_data(
                solid,shape_att, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples
            )
        # dgl.data.utils.save_graphs(str(output_path.joinpath(fn_stem + ".bin")), [graph])
    except StopIteration:
        print("No solid found in file:", fn)
        return
    except RuntimeError:
        print("Error loading file:", fn)
        return
    except Exception as e:
        print('Unhandled Exception:')
        print(traceback.format_exc())
        raise


    with open(output_path,'wb') as f:
        pickle.dump(face_data,f,protocol=pickle.HIGHEST_PROTOCOL)
    return face_data



def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process(args):
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    step_files = list(input_path.glob("*.st*p"))
    # for fn in tqdm(step_files):
    #     process_one_file((fn, args))
    pool = Pool(processes=args.num_processes, initializer=initializer)
    try:
        results = list(tqdm(pool.imap(process_one_file, zip(step_files, repeat(args))), total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    print(f"Processed {len(results)} files.")

def center_and_scale_uvgrid(inp , return_center_scale=False):
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp

def bounding_box_uvgrid(inp):
    pts = inp[..., :3].reshape((-1, 3))
    mask = inp[..., 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)

def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return np.array(box)
def center_and_scale(face,edge):
    face, center, scale = center_and_scale_uvgrid(
        face, return_center_scale=True
    )
    edge[...,:3] -= center
    edge[...,:3] *= scale

    return face,edge

def main(input_args=None):
    parser = argparse.ArgumentParser(
        "Convert solid models to face-adjacency graphs with UV-grid features"
    )
    parser.add_argument("input", type=str, help="Input folder of STEP files")
    parser.add_argument("output", type=str, help="Output folder of DGL graph BIN files")
    parser.add_argument(
        "--curv_u_samples", type=int, default=10, help="Number of samples on each curve"
    )
    parser.add_argument(
        "--surf_u_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the u-direction",
    )
    parser.add_argument(
        "--surf_v_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the v-direction",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--genlabel",
        default=False,
        action='store_true',
        help="Only generate face label file",
    )
    parser.add_argument(
        "--no_label",
        default=False,
        action='store_true',
        help="Don't generate labels",
    )
    args = parser.parse_args(input_args)
    process(args)


if __name__ == "__main__":
    main()