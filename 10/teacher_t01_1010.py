import numpy as np
from pathlib import Path

DIR = Path(__file__).resolve().parent

#
OUTPUT_PATH = DIR / "teacher_t01_points_1010.npy"

def main():
    c0 = np.array([1.0, 1.0])
    c1 = np.array([7.0, 7.0])
    data = np.array(
        [(1, 1), (1, 2), (2, 1), (7, 7), (7, 8), (8, 7)],
        dtype=np.float64,
    )
    print(data.min(axis=0))
    print(data.max(axis=0))
    np.save(OUTPUT_PATH, data)
    data_load = np.load(OUTPUT_PATH)
    print(data_load)
    print(data_load.shape)
    print(data_load.dtype)
    assert np.array_equal(data_load, data),"加载前后数据不同"
if __name__ == "__main__":
    main()
