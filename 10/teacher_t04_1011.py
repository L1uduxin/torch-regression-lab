from teacher_t02_1010 import assign_nearest
from teacher_t03_1011 import update_centers, squared_objective
import numpy as np
from pathlib import Path

DIR = Path(__file__).parent

DATA = DIR / "teacher_t01_points_1010.npy"


def run_kmeans(points, initial_centers, max_iter):
    # 返回 centers, labels, history, stop_reason
    centers = initial_centers.copy()
    n = len(initial_centers)

    labels = None
    history = []
    stop_reason = "达到最大轮数"

    for i in range(max_iter):
        d2, labels = assign_nearest(points, centers)

        centers_copy = centers.copy()
        centers = update_centers(points, labels, n)

        # 先计算本轮 J，并记录历史，再判断是否停止
        j = squared_objective(points, centers, labels)
        history.append((i + 1, j))

        if np.equal(centers_copy, centers).all():
            stop_reason = "中心不变"
            break

    return centers, labels, history, stop_reason


def main():
    points = np.load(DATA)
    initial_centers = np.array([[1.0, 1.0], [7.0, 7.0]])
    max_iter = 5

    centers, labels, history, stop_reason = run_kmeans(
        points, initial_centers, max_iter
    )

    print("centers =", centers)
    print("labels =", labels)
    print("history =", history)
    print("stop_reason =", stop_reason)


if __name__ == '__main__':
    main()