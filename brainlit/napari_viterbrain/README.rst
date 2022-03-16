ViterBrain Napari Plugin Installation
-------------------------------------

* First, install ``brainlit`` [`Documentation <https://brainlit.netlify.app/readme#installation>`_] (you may need to install from source with `pip install -e .`, since our pypi version may not reflect the latest changes in the repo).

* Second, install `napari <https://napari.org/>`_.

* The `Plugins` tab of napari should automatically find the brainlit plugin (`Documentation <https://napari.org/plugins/find_and_install_plugin.html#find-and-install-plugins>`_).

How to Use the ViterBrain Napari Plugin
---------------------------------------

* Build a ViterBrain object according to an image and some voxel-wise predictions.

* Open the ViterBrain object in napari.

* Launch the ViterBrain plugin widget.

* Select the labels layer and hover over fragments with your cursor to identify fragment ID numbers in the bottom left of the napari window. Identify the desired start and end fragment and enter the ID's in the widget box.

* Click trace and the plugin should generate a new path layer that shows the trace between the two fragments.