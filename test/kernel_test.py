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


def test_init_rope_freq():
    head_dim = 64
    half_dim = head_dim / 2
    inv_freq = np.arange(0, 64, 2)

    rope_theta = 500000.0
    inv_freq = 1 / np.pow(rope_theta, inv_freq / head_dim)

    max_hidden_dimension = 8192
    low_freq_factor = 1
    high_freq_factor = 4
    factor = 32.0
    low_freq_wave = max_hidden_dimension / low_freq_factor
    high_freq_wave = max_hidden_dimension / high_freq_factor

    wave_len = 2 * np.pi / inv_freq
    smooth = (max_hidden_dimension / wave_len - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    inv_freq_llama = np.select(
        condlist=[wave_len > low_freq_wave, wave_len >= high_freq_wave],
        choicelist=[
            inv_freq / factor,
            (1 - smooth) * (inv_freq / factor) + inv_freq * smooth,
        ],
        default=inv_freq,
    )

    max_seq_len = 2048

    pos = np.arange(0, max_seq_len, dtype=np.float32)

    angles = np.outer(pos, inv_freq_llama)

    cos_table = np.repeat(np.cos(angles), repeats=2, axis=-1)
    sin_table = np.repeat(np.sin(angles), repeats=2, axis=-1)

    cos_table_1d = cos_table.ravel()
    sin_table_1d = sin_table.ravel()

    return cos_table_1d, sin_table_1d


if __name__ == "__main__":
    np.set_printoptions(suppress=True, threshold=1000, precision=10)
