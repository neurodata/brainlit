# How to reproduce this experiment

Before reproducing this experiment, make sure that:

- You have installed the `brainlit` package [[Documentation](https://brainlit.netlify.app/readme#installation)]
- You have installed PyTorch [[Documentation](https://pytorch.org/get-started/locally/)]

This experiment does *not* require a GPU (CUDA is not needed for PyTorch).

N.B. if you are using [`conda`](https://docs.conda.io/en/latest/), make sure that both `brainlit` and PyTorch are installed within the same environment.

Now, follow these steps to reproduce the results of the experiment:

1. Download segments, which are stored in the publicly available S3 bucket [open-neurodata](https://registry.opendata.aws/open-neurodata/)

```shell
python scripts/download_segments.py
```

This script will prepare the experiment folder scaffolding

```shell
axon_geometry
├── data
│   ├── brain1
│   │   ├── segments_swc
│   │   └── trace_data
│   └── brain2        
│       ├── segments_swc
│       └── trace_data
├── figures
│
... etc.
```

and download data from S3 (no credentials are required)

2. Compute and save trace analysis data

```shell
python scripts/generate_trace_data.py
```

3. Run any of the notebooks in the `notebooks` folder, which will save the results in the `figures` folder
