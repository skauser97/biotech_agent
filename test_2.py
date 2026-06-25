from multiprocessing import Pool
import os

def square(n):
    return n * n

if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(square, numbers)

    print(results)
    # [1, 4, 9, 16, 25, 36, 49, 64]
    # always in input order, even if process 3 finished first