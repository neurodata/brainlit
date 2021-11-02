import multiprocessing as mp
import time

def ret(x,y):
    print(f"start process {x}")
    time.sleep(1)
    print(f"end process {x}")
    return x, y

if __name__ == "__main__":
    t1 = time.perf_counter()
    with mp.Pool(processes=4) as pool:
        args = [(i,i) for i in range(1,7)]
        results = pool.starmap(ret, args)
        print(results)
    print(time.perf_counter()-t1)