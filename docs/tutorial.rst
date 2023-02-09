********
Tutorial
********

.. _utils_tutorials:

Utils
=====
Tutorials showcasing how to use the utils folder.

.. toctree::
   :maxdepth: 1

   notebooks/utils/uploading_brains
   notebooks/utils/downloading_brains
   notebooks/utils/downloading_benchmarking
   notebooks/utils/uploading_benchmarking

.. _pipeline_tutorials:

Pipelines
=========

.. _BrainLine_tutorials:

Light-Sheet Microscopy Image Analysis Pipeline
----------------------------------------------
Perform axon detection and soma detection on brain images, and combine with CloudReg image registration for visualization and analysis.

.. toctree::
   :maxdepth: 1

   notebooks/pipelines/lsm_analysis/axon_analysis
   notebooks/pipelines/lsm_analysis/soma_analysis

Semi-automatic Annotation Pipeline
----------------------------------
Demonstrate pulling data and pushing traced annotations.

.. toctree::
   :maxdepth: 1

   notebooks/pipelines/seg_pipeline_demo

Segmentation
------------
Notebooks showing how to manually and automatically segment data.

.. toctree::
   :maxdepth: 1

   notebooks/pipelines/manual_segmentation
   notebooks/pipelines/tubes_feature_extraction_demo

.. _algorithm_tutorials:

Algorithms
==========

Adaptive Thresholding
---------------------
Demonstrate region growing methods using GMM and simple ITK.

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/adaptive_thresh_tutorial.ipynb
   
Connecting Fragments
---------------------
Demonstrate fragment path connections using Viterbi algorithm on a simple grid example.

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/viterbi_tutorial.ipynb
   notebooks/algorithms/viterbi2_tutorial.ipynb

Trace Analysis
--------------
Demonstrate estimation of curvature and torsion on simple curve, and fitting splines to a neuron.

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/spline_fxns_tutorial.ipynb
   notebooks/algorithms/view_swc_spline.ipynb
   notebooks/algorithms/biccn_demo.ipynb

Soma Detection
--------------
Demonstrate simple soma detection algorithm on known somas in Janelia dataset, brain1.

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/detect_somas.ipynb


.. _preprocessing_tutorials:

Preprocessing
=============

.. toctree::
   :maxdepth: 1

   notebooks/preprocessing/connectedcomponents
   notebooks/preprocessing/gaborfilter

.. _viz_tutorials:

Vizualization
=============
These tutorials demonstrate tools to load and visualize data from s3 buckets or .swc files.

.. toctree::
   :maxdepth: 1

   notebooks/visualization/loading
   notebooks/visualization/neighborhood_visualization_demo