#from src.data.make_dataset import main


def f():
    lst = [lambda : i**2 for i in range(100)]

    return lst[0]()
if __name__ == "__main__":

    f()
    pass