'''
copied from https://github.com/google/neuroglancer/blob/master/python/examples/example_action.py
need to run in interactive mode: python -i ng.py
'''

from __future__ import print_function



import argparse
import webbrowser

import neuroglancer
import neuroglancer.cli


ap = argparse.ArgumentParser()
neuroglancer.cli.add_server_arguments(ap)
args = ap.parse_args()
neuroglancer.cli.handle_server_arguments(args)

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    s.layers['ara'] = neuroglancer.SegmentationLayer(
        source='precomputed://https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_10um/annotation_10um_2017',
    )

def s_select(s):
    with viewer.txn() as vs: #add point
        layer_names = [l.name for l in vs.layers]
        if "start" in layer_names:
            if "end" in layer_names:
                del vs.layers["end"]
            name = "end"
            color = "#f00"
        else:
            name = "start"
            color = '#0f0'

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
    print('  Selected fragment: %s' % (s.selected_values, ))

viewer.actions.add('s_select', s_select)
with viewer.config_state.txn() as s:
    s.input_event_bindings.viewer['shift+keys'] = 's_select'
    s.status_messages['hello'] = 'Welcome to the ViterBrain tracer'

print(viewer)
webbrowser.open_new(viewer.get_viewer_url())
