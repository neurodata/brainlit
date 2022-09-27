'''
copied from https://github.com/google/neuroglancer/blob/master/python/examples/example_action.py
need to run in interactive mode: python -i ng.py
'''

from __future__ import print_function


import pickle
import argparse
import webbrowser
import numpy as np

import neuroglancer
import neuroglancer.cli
from cloudvolume import CloudVolume

# python cors_webserver.py -d "/Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/ng" -p 9010

class ViterBrainViewer(neuroglancer.Viewer):
    def __init__(self, im_url, frag_url, vb_path):
        super().__init__()

        ap = argparse.ArgumentParser()
        neuroglancer.cli.add_server_arguments(ap)
        args = ap.parse_args()
        neuroglancer.cli.handle_server_arguments(args)

        cv_dict = {}

        # add layers
        with self.txn() as s:
            s.layers['image'] = neuroglancer.ImageLayer(
                source=im_url,
            )
            cv_dict['image'] = CloudVolume(im_url)
            s.layers['fragments'] = neuroglancer.SegmentationLayer(
                source=frag_url,
            )
            cv_dict['fragments'] = CloudVolume(frag_url)

        # add keyboard actions
        self.actions.add('s_select', self.s_select)
        self.actions.add('t_trace', self.t_trace)
        self.actions.add('n_newtrace', self.n_newtrace)
        self.actions.add('c_clearlast', self.c_clearlast)
        self.actions.add('p_print', self.p_print)
        with self.config_state.txn() as s:
            s.input_event_bindings.viewer['shift+keys'] = 's_select'
            s.input_event_bindings.viewer['shift+keyt'] = 't_trace'
            s.input_event_bindings.viewer['shift+keyn'] = 'n_newtrace'
            s.input_event_bindings.viewer['shift+keyc'] = 'c_clearlast'
            s.input_event_bindings.viewer['shift+keyp'] = 'p_print'
            s.status_messages['hello'] = 'Welcome to the ViterBrain tracer'

        # open vb object
        with open(vb_path, "rb") as handle:
            self.vb = pickle.load(handle)
        self.vb.fragment_path = "/Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/3-1-soma_labels.zarr"

        # set useful object attributtes
        self.start_pt = None
        self.end_pt = None
        self.dimensions = neuroglancer.CoordinateSpace(
            names=['x','y','z'],
            units='nm',
            scales=cv_dict['image'].resolution
        )
        self.im_shape = [i for i in cv_dict['image'].shape if i != 1]
        self.cur_skel = 0
        self.cur_skel_coords = []

    def add_point(self, name, color, coord):
        """Add point to neuroglancer via Annotation Layer.

        Args:
            name (str): name of new annotation layer.
            color (str): hex code of point color.
            coord (list): coordinate of point in voxel units.
        """
        with self.txn() as vs: #add point
            vs.layers.append(name=name,
                layer=neuroglancer.LocalAnnotationLayer(
                    dimensions=vs.dimensions,
                    annotation_properties=[
                        neuroglancer.AnnotationPropertySpec(
                            id='color',
                            type='rgb',
                            default='red',
                        ),
                        neuroglancer.AnnotationPropertySpec(
                            id='size',
                            type='float32',
                            default=10,
                        ),
                        neuroglancer.AnnotationPropertySpec(
                            id='p_int8',
                            type='int8',
                            default=10,
                        ),
                        neuroglancer.AnnotationPropertySpec(
                            id='p_uint8',
                            type='uint8',
                            default=10,
                        ),
                    ],
                    annotations=[
                        neuroglancer.PointAnnotation(
                            id='1',
                            point=coord,
                            props=[color, 5, 6, 7],
                        ),
                    ],
                    shader='''
                            void main() {
                            setColor(prop_color());
                            setPointMarkerSize(prop_size());
                            }
                            ''',
                ),
            )


    def s_select(self, s):
        """Action to add point to neuroglancer

        Args:
            s (??): action that selects point for tracing.
        """
        with self.txn() as vs: #add point
            layer_names = [l.name for l in vs.layers]
            print('Selected fragment: %s' % (s.selected_values, ))
            if "start" in layer_names:
                if "end" in layer_names:
                    del vs.layers["end"]
                name = "end"
                color = "#f00"
                self.end_pt = [int(p) for p in s.mouse_voxel_coordinates]
            else:
                name = "start"
                color = '#0f0'
                self.start_pt = [int(p) for p in s.mouse_voxel_coordinates]
        self.add_point(name, color, s.mouse_voxel_coordinates)
            

    def t_trace(self, s):
        with self.txn() as vs: #trace
            layer_names = [l.name for l in vs.layers]
            if "start" in layer_names and "end" in layer_names:
                print(f"Tracing path from {self.start_pt} to {self.end_pt}")
                path = self.vb.shortest_path(coord1 = self.start_pt, coord2 = self.end_pt)
                self.cur_skel_coords.append(path)
                trace_layer_name = f"vb_traces_{self.cur_skel}"
                if trace_layer_name in layer_names:
                    del vs.layers[trace_layer_name]

                coords_list = [inner for outer in self.cur_skel_coords for inner in outer]
                skel_source = self.SkeletonSource(self.dimensions)
                skel_source.add_skeleton(coords_list=coords_list)
                vs.layers.append(
                    name=trace_layer_name,
                    layer = neuroglancer.SegmentationLayer(
                        source = [
                            neuroglancer.LocalVolume(
                                data = np.zeros(self.im_shape, dtype=np.uint32),
                                dimensions = self.dimensions
                            ),
                            skel_source
                        ],
                        segments = [0]
                    )
                )
            
               

                del vs.layers['start']
                vs.layers['end'].name = 'start'
                self.start_pt = self.end_pt
            else:
                print("No start/end layers yet")

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

                coords_list = [inner for outer in self.cur_skel_coords for inner in outer]
                skel_source = self.SkeletonSource(self.dimensions)
                skel_source.add_skeleton(coords_list=coords_list)

                with self.txn() as vs: 
                    vs.layers.append(
                        name=trace_layer_name,
                        layer = neuroglancer.SegmentationLayer(
                            source = [
                                neuroglancer.LocalVolume(
                                    data = np.zeros(self.im_shape, dtype=np.uint32),
                                    dimensions = self.dimensions
                                ),
                                skel_source
                            ],
                            segments = [0]
                        )
                    )



    def n_newtrace(self, s):
        print(f"Creating new trace #{self.cur_skel+1}")
        with self.txn() as vs: #trace
            layer_names = [l.name for l in vs.layers]
            if "start" in layer_names:
                del vs.layers["start"]
            if "end" in layer_names:
                del vs.layers["end"]

        self.cur_skel += 1
        self.cur_skel_coords = []

    def p_print(self, s):
        print(f"Printing trace info for vb_trace_{self.cur_skel}:")
        coords_list = [inner for outer in self.cur_skel_coords for inner in outer]
        print(coords_list)
        

    class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
        """Source used to serve skeleton objects generated by ViterBrain tracing.

        Args:
            neuroglancer (_type_): _description_
        """
        def __init__(self, dimensions):
            super().__init__(dimensions)
            self.skels = []

        def add_skeleton(self, coords_list):
            edges = [[i-1,i] for i in range(1, len(coords_list))]
            self.skels.append(neuroglancer.skeleton.Skeleton(vertex_positions=coords_list,
                edges = edges))

        def get_skeleton(self, i):
            return self.skels[i]




vbviewer = ViterBrainViewer(im_url='precomputed://http://127.0.0.1:9010/im', frag_url='precomputed://http://127.0.0.1:9010/frags', vb_path="/Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/3-1-soma_viterbrain.pickle")
print(vbviewer)
webbrowser.open_new(vbviewer.get_viewer_url())
