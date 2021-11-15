from brainlit.algorithms.generate_fragments import state_generation

sg = state_generation("/data/tathey1/mouselight/1mm.zarr", "/home/tathey1/ilastik-1.3.3post3-Linux/run_ilastik.sh", "/data/tathey1/mouselight/octopus_exp.ilp", parallel=12)
sg.predict("/data/tathey1/mouselight/data_bin/")