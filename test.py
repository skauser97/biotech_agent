from multiprocessing import Pool

def heavy_maths(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        results = pool.map(heavy_maths, [5000000, 5000000, 5000000, 5000000])
    print(results)