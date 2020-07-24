********
Tutorial
********

.. _utils_tutorials:

Utils
=====
Tutorials showcasing how to use the utils folder.

.. toctree::
   :maxdepth: 1

   notebooks/utils/utils

.. _pipeline_tutorials:

Pipelines
=========

Semi-automatic Annotation Pipeline
----------------------------------
Demonstrate pulling data and pushing traced annotations.

.. toctree::
   :maxdepth: 1

   notebooks/pipelines/seg_pipeline_demo

Manual Segmentation
-------------------
Notebook showing how to manually segment data.

.. toctree::
   :maxdepth: 1

   notebooks/pipelines/manual_segmentation


.. _algorithm_tutorials:

Algortihms
==========

Adaptive Thresholding
---------------------
Demonstrate region growing methods using GMM and simple ITK

.. toctree::
   :maxdepth: 1

   notebooks/algorithms/tutorial_notebook_adaptive_thresh.ipynb

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

   notebooks/connectedcomponents
   notebooks/pcawhitening
   notebooks/windowpad
   notebooks/gaborfilter


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

Uploading Brains
================
This notebook describes the process to upload brain data.

.. toctree::
   :maxdepth: 1

   notebooks/utils/uploading_brains.ipynb
