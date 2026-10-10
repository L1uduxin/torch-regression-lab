import numpy as np


def assign_nearest(points, centers):
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    diff = points[:, None, :] - centers[None, :, :]   # (N, K, 2)
    d2 = np.sum(diff * diff, axis=-1)                 # (N, K)
    labels = d2.argmin(axis=1)                        # (N,)

    return d2, labels


def main():
    data = np.load("teacher_t01_points_1010.npy").astype(np.float64)
    centers = np.asarray([[1.0, 1.0], [7.0, 7.0]], dtype=np.float64)

    d2, labels = assign_nearest(data, centers)

    print("d2 =\n", d2)
    print("d2.shape =", d2.shape)
    print("labels =", labels)
    print("labels.shape =", labels.shape)

    # 手算核对：第二个样本（索引1）到两中心的平方距离应为 [1, 61]，label 为 0
    print("\n第二个样本 d2[1] =", d2[1])
    print("第二个样本 label =", labels[1])
    assert np.allclose(d2[1], [1.0, 61.0]), "第二个样本距离不匹配"
    assert labels[1] == 0, "第二个样本中心索引不匹配"
    print("第二个样本核对通过。")


if __name__ == "__main__":
    main()