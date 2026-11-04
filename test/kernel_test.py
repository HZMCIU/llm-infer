import numpy as np
import sys


def test_rms_norm():
    input = np.arange(0, 2048)
    input = (input % 32) + 1

    weight = np.arange(0, 2048)
    weight = (weight % 32) + 1

    rms = np.sqrt(((input**2).sum()) / 2048 + 1e-15)

    input = input / rms

    output = input * weight

    return output


if __name__ == "__main__":
    np.set_printoptions(suppress=True, threshold=1000)
    print(test_rms_norm()[:100])
