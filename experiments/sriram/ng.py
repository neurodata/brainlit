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

        print(s.layers['image'])
        print(cv_dict['image'].info)

        self.actions.add('s_select', self.s_select)
        self.actions.add('t_trace', self.t_trace)
        with self.config_state.txn() as s:
            s.input_event_bindings.viewer['shift+keys'] = 's_select'
            s.input_event_bindings.viewer['shift+keyt'] = 't_trace'
            s.status_messages['hello'] = 'Welcome to the ViterBrain tracer'

        # open vb object
        with open(vb_path, "rb") as handle:
            self.vb = pickle.load(handle)
        self.vb.fragment_path = "/Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/sriram/sample/3-1-soma_labels.zarr"

        self.start_pt = None
        self.end_pt = None
        self.dimensions = neuroglancer.CoordinateSpace(
            names=['x','y','z'],
            units='nm',
            scales=cv_dict['image'].resolution
        )

    def s_select(self, s):
        with self.txn() as vs: #add point
            layer_names = [l.name for l in vs.layers]
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
                            point=s.mouse_voxel_coordinates,
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
        print('Selected fragment: %s' % (s.selected_values, ))

    def t_trace(self, s):
        with self.txn() as vs: #trace
            layer_names = [l.name for l in vs.layers]
            if "start" in layer_names and "end" in layer_names:
                print(f"Computing path from {self.start_pt} to {self.end_pt}")
                path = self.vb.shortest_path(coord1 = self.start_pt, coord2 = self.end_pt)
                skel_source = self.SkeletonSource(self.dimensions)
                skel_source.add_skeleton(coords_list=path)
                print(path)
                del vs.layers['start']
                del vs.layers['end']
                vs.layers.append(
                    name="vb_traces",
                    layer = neuroglancer.SegmentationLayer(
                        source = [
                            neuroglancer.LocalVolume(
                                data = np.zeros((100,100,100), dtype=np.uint32),
                                dimensions = self.dimensions
                            ),
                            skel_source
                        ],
                        segments = [0]
                    )
                )
            else:
                print("No start/end layers yet")


    class SkeletonSource(neuroglancer.skeleton.SkeletonSource):
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
