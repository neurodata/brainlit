TACC Tutorial
-------------

In Jupyter session on Lonestar 6:

1. In home directory (`cdh`) make python 3.9 virtual environment: `python3 -m venv venv_39`.
2. Activate virtual environment: `source venv_39/bin/activate`.
3. Update pip `pip install --upgrade pip`
4. In scratch directory (`cds`), download ilastik (`wget https://files.ilastik.org/ilastik-1.4.0-Linux.tar.bz2`).
5. Decompress the ilastik file (`tar –xvf ilastik-1.4.0-Linux.tar.bz2`).
6. Clone brainlit repository (`git clone https://github.com/neurodata/brainlit.git`).
7. Install brainlit from source in editable mode: `cd brainlit && pip install -e .`.
8. Install packages I use for this tutorial: `pip install matplotlib-scalebar jupyter`.
9. Go back to the home directory (`cdh`) and copy the notebook I made: `cp /home1/09423/tathey1/brainline-tacc-tutorial.ipynb .`.
10. Create jupyter kernel for this virtual environment: `ipython kernel install --name "venv_39" --user`.
11. Open the jupyter notebook and select `venv_39` as the kernel.
12. As you run the notebook, you will need to change a couple variables including `brainlit_path` and `ilastik_path` according to your scratch directory path.


Atlas
-----

If you plan on using BrainLine analysis a lot (in particular, the napari coronal section views), I recommend you download the atlas from [here](https://neurodata.io/data/allen_atlas/).
