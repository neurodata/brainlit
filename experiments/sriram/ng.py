'''
copied from https://github.com/google/neuroglancer/blob/master/python/examples/example_action.py
need to run in interactive mode: python -i ng.py
'''

from __future__ import print_function



import argparse
import webbrowser

import neuroglancer
import neuroglancer.cli

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers['ara'] = neuroglancer.SegmentationLayer(
            source='precomputed://https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_10um/annotation_10um_2017',
        )

    def my_action(s):
        print('Got my-action')
        print('  Mouse position: %s' % (s.mouse_voxel_coordinates, ))
        print('  Layer selected values: %s' % (s.selected_values, ))

    viewer.actions.add('my-action', my_action)
    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer['shift+mousedown0'] = 'my-action'
        s.input_event_bindings.viewer['shift+mousedown2'] = 'my-action'
        s.input_event_bindings.viewer['shift+mousedown2'] = 'my-action'
        s.input_event_bindings.viewer['shift+keyt'] = 'my-action'
        s.status_messages['hello'] = 'Welcome to this example'

    print(viewer)
    webbrowser.open_new(viewer.get_viewer_url())
