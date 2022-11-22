"""
copied from https://github.com/google/neuroglancer/blob/master/python/examples/example_action.py
need to run in interactive mode: python -i ng.py

serve data:
local:
python cors_webserver.py -d "/Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/ng/" -p 9010

use env_39 on local

cis:
python cors_webserver.py -d "/cis/home/tathey/projects/mouselight/sriram/neuroglancer_data/somez/" -p 9010

python cors_webserver.py -d "/cis/project/sriram/ng_data/" -p 9010
zarr://http://127.0.0.1:9010/exp227/fg_ome.zarr

a soma - 5346, 14801, 330

use env_310 on dwalin

python -i experiments/sriram/ng.py

"""

from __future__ import print_function


import pickle
import argparse
import webbrowser
import numpy as np

import neuroglancer
import neuroglancer.cli
from cloudvolume import CloudVolume, Skeleton
from cloudvolume.exceptions import SkeletonDecodeError
from scipy.spatial import KDTree
import napari
from napari._qt.qthreading import thread_worker

data = np.random.random((10, 10,10))*255
viewer = napari.Viewer()
layer = viewer.add_image(data)

def show_napari(subvol):
    print("updating napari")
    layer.data = subvol

class ViterBrainViewer(neuroglancer.Viewer):
    def __init__(self, im_url, frag_url, trace_url, trace_path, im_path, vb_path, napari_viewer):
        super().__init__()

        ap = argparse.ArgumentParser()
        neuroglancer.cli.add_server_arguments(ap)
        args = ap.parse_args()
        neuroglancer.cli.handle_server_arguments(args)

        cv_dict = {"traces": CloudVolume(trace_path, compress=False), "im": CloudVolume(im_path, compress=False)}

        # add layers
        with self.txn() as s:
            s.layers["image"] = neuroglancer.ImageLayer(
                source=im_url,
            )
            s.layers["fragments"] = neuroglancer.SegmentationLayer(
                source=frag_url,
            )
            s.layers["traces"] = neuroglancer.SegmentationLayer(
                source=trace_url,
            )

        # add keyboard actions
        self.actions.add("s_select", self.s_select)
        self.actions.add("t_trace", self.t_trace)
        self.actions.add("n_newtrace", self.n_newtrace)
        self.actions.add("c_clearlast", self.c_clearlast)
        self.actions.add("p_print", self.p_print)
        self.actions.add("l_line", self.l_line)
        self.actions.add("h_hook", self.h_hook)
        self.actions.add("v_view", self.v_view)
        with self.config_state.txn() as s:
            s.input_event_bindings.viewer["shift+keys"] = "s_select"
            s.input_event_bindings.viewer["shift+keyt"] = "t_trace"
            s.input_event_bindings.viewer["shift+keyn"] = "n_newtrace"
            s.input_event_bindings.viewer["shift+keyc"] = "c_clearlast"
            s.input_event_bindings.viewer["shift+keyp"] = "p_print"
            s.input_event_bindings.viewer["shift+keyl"] = "l_line"
            s.input_event_bindings.viewer["shift+keyh"] = "h_hook"
            s.input_event_bindings.viewer["shift+keyv"] = "v_view"
            s.status_messages["hello"] = "Welcome to the ViterBrain tracer"

        # open vb object
        with open(vb_path, "rb") as handle:
            self.vb = pickle.load(handle)
        self.vb.fragment_path = "/Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/3-1-soma_labels.zarr"

        # set useful object attributtes
        self.start_pt = None
        self.end_pt = None
        self.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z"], units="nm", scales=cv_dict["traces"].resolution
        )
        self.im_shape = [i for i in cv_dict["traces"].shape if i != 1]
        self.cur_skel_coords = []
        self.cv_dict = cv_dict

        self.num_skels, _, _ = self.build_kdtree()
        self.cur_skel = self.num_skels
        self.cur_skel_head = 0

        self.napari_viewer = napari_viewer

    def build_kdtree(self):
        vol = self.cv_dict["traces"]
        skel_num = 0
        pts_total = []
        ids_total = []
        while True:
            try:
                pts = vol.skeleton.get(skel_num).vertices
                pts = np.divide(pts, vol.resolution)
                pts_total.append(pts)
                ids_total += [(skel_num, i) for i in range(pts.shape[0])]
            except SkeletonDecodeError:
                break

            skel_num += 1
        if skel_num == 0:
            pts_total = None
            kdtree = None
        else:
            pts_total = np.concatenate(pts_total, axis=0)
            kdtree = KDTree(pts_total)
        return skel_num, kdtree, ids_total

    def add_point(self, name, color, coord):
        """Add point to neuroglancer via Annotation Layer.

        Args:
            name (str): name of new annotation layer.
            color (str): hex code of point color.
            coord (list): coordinate of point in voxel units.
        """
        with self.txn() as vs:  # add point
            vs.layers.append(
                name=name,
                layer=neuroglancer.LocalAnnotationLayer(
                    dimensions=vs.dimensions,
                    annotation_properties=[
                        neuroglancer.AnnotationPropertySpec(
                            id="color",
                            type="rgb",
                            default="red",
                        ),
                        neuroglancer.AnnotationPropertySpec(
                            id="size",
                            type="float32",
                            default=10,
                        ),
                        neuroglancer.AnnotationPropertySpec(
                            id="p_int8",
                            type="int8",
                            default=10,
                        ),
                        neuroglancer.AnnotationPropertySpec(
                            id="p_uint8",
                            type="uint8",
                            default=10,
                        ),
                    ],
                    annotations=[
                        neuroglancer.PointAnnotation(
                            id="1",
                            point=coord,
                            props=[color, 5, 6, 7],
                        ),
                    ],
                    shader="""
                            void main() {
                            setColor(prop_color());
                            setPointMarkerSize(prop_size());
                            }
                            """,
                ),
            )

    def s_select(self, s):
        """Action to add point to neuroglancer

        Args:
            s (??): action that selects point for tracing.
        """
        with self.txn() as vs:  # add point
            layer_names = [l.name for l in vs.layers]
            print("Selected fragment: %s" % (s.selected_values["fragments"],))
            if "start" in layer_names:
                if "end" in layer_names:
                    del vs.layers["end"]
                name = "end"
                color = "#f00"
                self.end_pt = [int(p) for p in s.mouse_voxel_coordinates]
            else:
                name = "start"
                color = "#0f0"
                self.start_pt = [int(p) for p in s.mouse_voxel_coordinates]
        self.add_point(name, color, s.mouse_voxel_coordinates)

    def h_hook(self, s):
        _, kdtree, ids_total = self.build_kdtree()
        _, closest_idx = kdtree.query(s.mouse_voxel_coordinates)
        self.start_pt = [int(p) for p in s.mouse_voxel_coordinates]
        self.cur_skel = ids_total[closest_idx][0]
        self.cur_skel_head = ids_total[closest_idx][1]
        print(f"Hooking into skeleton #{self.cur_skel} at id {self.cur_skel_head}: {self.start_pt}")
        self.add_point("start", "#0f0", s.mouse_voxel_coordinates)


    def t_trace(self, s):
        with self.txn() as vs:  # trace
            layer_names = [l.name for l in vs.layers]

        if "start" in layer_names and "end" in layer_names:
            print(f"Tracing path from {self.start_pt} to {self.end_pt}")
            path = self.vb.shortest_path(coord1=self.start_pt, coord2=self.end_pt)
            self.render_line(path, layer_names)
            with self.txn() as vs:  # trace
                del vs.layers["start"]
                self.start_pt = self.end_pt
                self.end_pt = None
                vs.layers["end"].name = "start"
        else:
            print("No start/end layers yet")

    def l_line(self, s):
        with self.txn() as vs:  # trace
            layer_names = [l.name for l in vs.layers]

        if "start" in layer_names:
            if "end" in layer_names:
                print(f"Drawing line from {self.start_pt} to {self.end_pt}")
                with self.txn() as vs:  # trace
                    del vs.layers["start"]
                    vs.layers["end"].name = "start"
            else:
                self.end_pt = [int(p) for p in s.mouse_voxel_coordinates]
                print(f"Drawing line from {self.start_pt} to mouse location {self.end_pt}")
                with self.txn() as vs:  # trace
                    del vs.layers["start"]
                self.add_point("start", "#0f0", s.mouse_voxel_coordinates)

            path = [self.start_pt, self.end_pt]
            self.render_line(path, layer_names)
            self.start_pt = self.end_pt
            self.end_pt = None
        else:
            print("No start/end layers yet")

    def render_line(self, path, layer_names):
        with self.txn() as vs:  # trace
            self.cur_skel_coords.append(path)
            trace_layer_name = f"vb_traces_{self.cur_skel}"
            if trace_layer_name in layer_names:
                del vs.layers[trace_layer_name]

            coords_list = [
                inner for outer in self.cur_skel_coords for inner in outer
            ]
            skel_source = self.SkeletonSource(self.dimensions)
            skel_source.add_skeleton(coords_list=coords_list)
            vs.layers.append(
                name=trace_layer_name,
                layer=neuroglancer.SegmentationLayer(
                    source=[
                        neuroglancer.LocalVolume(
                            data=np.zeros(self.im_shape, dtype=np.uint32),
                            dimensions=self.dimensions,
                        ),
                        skel_source,
                    ],
                    segments=[0],
                ),
            )


    def c_clearlast(self, s):
        if len(self.cur_skel_coords) == 0:
            print("Nothing to clear")
        else:
            self.cur_skel_coords = self.cur_skel_coords[:-1]

            with self.txn() as vs:
                layer_names = [l.name for l in vs.layers]
                trace_layer_name = f"vb_traces_{self.cur_skel}"
                if trace_layer_name in layer_names:
                    del vs.layers[trace_layer_name]
                if "start" in layer_names:
                    del vs.layers["start"]
                if "end" in layer_names:
                    del vs.layers["end"]

            if len(self.cur_skel_coords) > 0:
                coord = self.cur_skel_coords[-1][-1]
                self.start_pt = coord
                self.add_point("start", "#f00", coord)

                coords_list = [
                    inner for outer in self.cur_skel_coords for inner in outer
                ]
                skel_source = self.SkeletonSource(self.dimensions)
                skel_source.add_skeleton(coords_list=coords_list)

                with self.txn() as vs:
                    vs.layers.append(
                        name=trace_layer_name,
                        layer=neuroglancer.SegmentationLayer(
                            source=[
                                neuroglancer.LocalVolume(
                                    data=np.zeros(self.im_shape, dtype=np.uint32),
                                    dimensions=self.dimensions,
                                ),
                                skel_source,
                            ],
                            segments=[0],
                        ),
                    )

    def n_newtrace(self, s):
        print(f"Saving trace #{self.cur_skel+1}")
        self.p_print(s)
        with self.txn() as vs:  # trace
            layer_names = [l.name for l in vs.layers]
            if "start" in layer_names:
                del vs.layers["start"]
            if "end" in layer_names:
                del vs.layers["end"]

        self.start_pt = None
        self.end_pt = None
        self.num_skels += 1
        self.cur_skel = self.num_skels
        self.cur_skel_coords = []
        self.cur_skel_head = 0
        print(f"Creating new trace #{self.cur_skel}")

    def p_print(self, s):
        if len(self.cur_skel_coords) == 0:
            print("No coordinates to save")
        else:
            print(f"Printing trace info for vb_trace_{self.cur_skel}:")
            coords_list = [inner for outer in self.cur_skel_coords for inner in outer]
            vol = self.cv_dict["traces"]
            vertices = np.array(coords_list)
            vertices = np.multiply(vertices, vol.resolution)

            try:
                skel = vol.skeleton.get(self.cur_skel)
                cur_num_verts = skel.vertices.shape[0]
                edges = [[self.cur_skel_head, cur_num_verts]] + [[i - 1, i] for i in range(cur_num_verts+1, vertices.shape[0]+cur_num_verts)]
                edges = np.array(edges)
                edges = np.concatenate((skel.edges, edges), axis=0)
                vertices = np.concatenate((skel.vertices, vertices), axis=0)
            except SkeletonDecodeError:
                if self.cur_skel_head != 0:
                    raise ValueError(f"Writing new skeleton (#{self.cur_skel}), but cur_skel_head is >0 ({self.cur_skel_head})")
                edges = [[i - 1, i] for i in range(1, len(coords_list))]


            skel = Skeleton(segid = self.cur_skel, vertices=vertices, edges=edges, space='voxel')
            skel.extra_attributes = [ attr for attr in skel.extra_attributes if attr['data_type'] == 'float32' ]
            vol.skeleton.upload(skel)
            print(f"verts: {vertices}, edges: {edges}")

    # @thread_worker(connect={'returned': show_napari})
    def v_view(self, s):
        print(f"Updating napari")
        pt = [int(p) for p in s.mouse_voxel_coordinates]
        vol_im = self.cv_dict["im"]
        radius = 10
        subvol = np.array(np.squeeze(vol_im[pt[0]-radius:pt[0]+radius,pt[1]-radius:pt[1]+radius,pt[2]-radius:pt[2]+radius]))
        print(f"range {np.amin(subvol)} to {np.amax(subvol)}")
        show_napari(subvol)
        # return subvol


    class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
        """Source used to serve skeleton objects generated by ViterBrain tracing.

        Args:
            neuroglancer (_type_): _description_
        """

        def __init__(self, dimensions):
            super().__init__(dimensions)
            self.skels = []

        def add_skeleton(self, coords_list):
            edges = [[i - 1, i] for i in range(1, len(coords_list))]
            self.skels.append(
                neuroglancer.skeleton.Skeleton(
                    vertex_positions=coords_list, edges=edges
                )
            )

        def get_skeleton(self, i):
            return self.skels[i]



vb_path_local = "/Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/3-1-soma_viterbrain.pickle"
vb_path_cis = "/cis/home/tathey/projects/mouselight/sriram/somez_viterbrain.pickle"
trace_path_local = "precomputed://file:///Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/ng/traces"
trace_path_cis = "precomputed://file:///cis/home/tathey/projects/mouselight/sriram/neuroglancer_data/somez/traces"
port = "9010"
im_layer = "im"
frag_layer = "frags"
trace_layer="traces"
trace_path = "precomputed://file:///Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/ng/traces"
trace_path_cis = "precomputed://file:///cis/home/tathey/projects/mouselight/sriram/neuroglancer_data/somez/traces"
im_path = "precomputed://file:///Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/ng/im"
im_path_cis = "precomputed://file:///cis/home/tathey/projects/mouselight/sriram/neuroglancer_data/somez/im"

vbviewer = ViterBrainViewer(
    im_url=f"precomputed://http://127.0.0.1:{port}/{im_layer}",
    frag_url=f"precomputed://http://127.0.0.1:{port}/{frag_layer}",
    trace_url=f"precomputed://http://127.0.0.1:{port}/{trace_layer}",
    trace_path=trace_path_cis,
    im_path=im_path_cis,
    vb_path=vb_path_cis,
    napari_viewer=viewer,
)
print(vbviewer)
webbrowser.open_new(vbviewer.get_viewer_url())


napari.run()
