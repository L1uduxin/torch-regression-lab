import numpy as np
from pathlib import Path
from teacher_t02_1010 import assign_nearest

DIR = Path(__file__).resolve().parent
DATA_PATH = DIR / "teacher_t01_points_1010.npy"


def update_centers(points, labels, n_clusters):
    # 返回新中心，形状 (n_clusters, 2)
    centers = np.zeros((n_clusters, 2))
    for i, label in enumerate(np.unique(labels)):
        num = points[labels == label]
        centers[i] = num.mean(axis=0)
    return centers


def squared_objective(points, centers, labels):
    # 返回标量 J
    diff = points - centers[labels]
    return np.sum(diff ** 2)


def main():
    data = np.load(DATA_PATH)
    c = np.array([[1.0, 1.0], [7.0, 7.0]])
    d2, labels = assign_nearest(data, c)

    c_new = update_centers(data, labels, len(c))

    J_old = squared_objective(data, c, labels)
    J_new = squared_objective(data, c_new, labels)

    target = np.array([[4/3, 4/3]])
    assert np.allclose(c_new[0], target , atol=1e-12, rtol=0)
    assert J_old + 1e-12 >= J_new


if __name__ == '__main__':
    main()