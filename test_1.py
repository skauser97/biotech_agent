from multiprocessing import Process

def heavy(n):
    result = sum(i*i for i in range(n))
    print(f"Done: {result}")

if __name__ == "__main__":
    p1 = Process(target=heavy, args=(1000000,))
    p2 = Process(target=heavy, args=(2000000,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()