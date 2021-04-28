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


.. _algorithm_tutorials:

Algorithms
==========

Adaptive Thresholding
---------------------
Demonstrate region growing methods using GMM and simple ITK

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/adaptive_thresh_tutorial.ipynb

Regression Clasifiers
---------------------
Demonstrate collecting features and using them for axon classification.

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/log_regression_classifiers_tutorial


.. _preprocessing_tutorials:

Preprocessing
=============
These tutorials demonstrate different preprocessing methods: connected components, PCA whitening, paddinng, and gabor filters.

.. toctree::
   :maxdepth: 1

   notebooks/preprocessing/connectedcomponents
   notebooks/preprocessing/pcawhitening
   notebooks/preprocessing/windowpad
   notebooks/preprocessing/gaborfilter


.. _features_tutorials:

Features
========
This tutorial presents feature extraction methods: neighborhood-based and linear filter-based methods.

.. toctree::
   :maxdepth: 1
      
   notebooks/features/features

.. _viz_tutorials:

Vizualization
=============
These tutorials demonstrate tools to load and visualize data from s3 buckets or .swc files.

.. toctree::
   :maxdepth: 1

   notebooks/visualization/loading
   notebooks/visualization/visualization
   notebooks/visualization/neighborhood_visualization_demo
