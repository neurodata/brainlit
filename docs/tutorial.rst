********
Tutorial
********

.. _pipeline_tutorials:

Pipelines
=========

Semi-automatic Annotation Pipeline
----------------------------------
Demonstrate pulling data and pushing traced annotations.

.. toctree::
   :maxdepth: 1

   notebooks/seg_pipeline_demo

Manual Segmentation
-------------------
Notebook showing how to manually segment data.

.. toctree::
   :maxdepth: 1

   notebooks/manual_segmentation


.. _algorithm_tutorials:

Algortihms
==========

Adaptive Thresholding
---------------------
Demonstrate region growing methods using GMM and simple ITK

.. toctree::
   :maxdepth: 1

   notebooks/tutorial_notebook_adaptive_thresh.ipynb

Regression Clasifiers
---------------------
Demonstrate collecting features and using them for axon classification.

.. toctree::
   :maxdepth: 1

   notebooks/log_regression_classifiers_tutorial


.. _preprocessing_tutorials:

Preprocessing
=============
These tutorials demonstrate different preprocessing methods: connected components, PCA whitening, paddinng, and gabor filters.

.. toctree::
   :maxdepth: 1

   tutorials/connectedcomponents
   tutorials/pcawhitening
   tutorials/windowpad
   tutorials/gaborfilter


.. _features_tutorials:

Features
========
This tutorial presents feature extraction methods: neighborhood-based and linear filter-based methods.

.. toctree::
   :maxdepth: 1
      
   tutorials/features/features

.. _viz_tutorials:

Vizualization
=============
These tutorials demonstrate tools to load and visualize data from s3 buckets or .swc files.

.. toctree::
   :maxdepth: 1

   notebooks/loading
   notebooks/visualization

Uploading Brains
================
This notebook describes the process to upload brain data.

.. toctree::
   :maxdepth: 1

   tutorials/uploading_brains.ipynb
