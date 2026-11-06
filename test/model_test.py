import numpy as np


def test_matmul():
    n = 2
    weight = np.arange(0, 2048 * 2048).reshape(2048, 2048)
    weight = weight % 32 + 1
    input = np.arange(0, n * 2048).reshape(2, 2048)
    input = input % 32 + 1
    output = input @ (weight.T)
    print(output)


if __name__ == "__main__":
    test_matmul()
