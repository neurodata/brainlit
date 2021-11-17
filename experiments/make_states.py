from brainlit.algorithms.generate_fragments import state_generation
import time

t1 = time.perf_counter()
sg = state_generation(
    "/data/tathey1/mouselight/250.zarr",
    "/home/tathey1/ilastik-1.3.3post3-Linux/run_ilastik.sh",
    "/data/tathey1/mouselight/octopus_exp.ilp",
    chunk_size=[300, 300, 300],
    parallel=12,)
    #prob_path="/data/tathey1/mouselight/1_probs.zarr")
print(f"create object in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.predict("/data/tathey1/mouselight/data_bin/")
print(f"computed ilastik predictions in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.compute_frags()
print(f"computed fragments in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.compute_image_tiered()
print(f"computed fragments in {time.perf_counter()-t1} seconds")

t1 = time.perf_counter()
sg.compute_states()
print(f"computed fragments in {time.perf_counter()-t1} seconds")
