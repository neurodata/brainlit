import multiprocessing as mp
import time

def mult(x,y):
    print(f"start process {x}")
    time.sleep(1)
    prod = x*y
    print(f"end process {x}")
    return prod

if __name__ == "__main__":
    t1 = time.perf_counter()
    with mp.Pool(processes=4) as pool:
        args = [(i,i) for i in range(1,7)]
        results = pool.starmap(mult, args)
        print(results)
    print(time.perf_counter()-t1)