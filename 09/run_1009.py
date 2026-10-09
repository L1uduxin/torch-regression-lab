import math

ROWS_COLUMNS = ['id', 'group', 'prediction', 'target']


def _check_rows(rows):
    if len(rows) == 0:
        raise ValueError("No rows found")

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Row ids not unique")

    for row in rows:
        pred = row["prediction"]
        targ = row["target"]
        if (not isinstance(pred, (int, float))
                or isinstance(pred, bool)
                or not math.isfinite(pred)):
            raise ValueError(f"Prediction must be a finite int/float: {pred!r}")
        if (not isinstance(targ, (int, float))
                or isinstance(targ, bool)
                or not math.isfinite(targ)):
            raise ValueError(f"Target must be a finite int/float: {targ!r}")


def summarize_groups(rows):
    _check_rows(rows)

    buckets = {}
    for row in rows:
        sq = (row["prediction"] - row["target"]) ** 2
        buckets.setdefault(row["group"], []).append(sq)
    per_group = {}
    mse = []
    total_sq = 0.0
    for g, sq_list in buckets.items():
        n = len(sq_list)
        total_sq += sum(sq_list)
        per_group[g] = {"n": n, "mse": sum(sq_list) / n}
        mse.append(sum(sq_list))

    overall_mse = total_sq / len(rows)



    return {"per_group": per_group, "overall_mse": overall_mse}


def main():
    # 正常例：组大小不等
    rows_ok = [
        {"id": "a1", "group": "A", "prediction": 0.0, "target": 0.0},  # err²=0
        {"id": "a2", "group": "A", "prediction": 2.0, "target": 0.0},  # err²=4
        {"id": "a3", "group": "A", "prediction": 1.0, "target": 1.0},  # err²=0
        {"id": "b1", "group": "B", "prediction": 3.0, "target": 1.0},  # err²=4
        {"id": "b2", "group": "B", "prediction": 1.0, "target": 1.0},  # err²=0
    ]
    print(summarize_groups(rows_ok))

    # 失败例：重复 id
    rows_bad_dup = [
        {"id": "x1", "group": "A", "prediction": 0.0, "target": 0.0},
        {"id": "x1", "group": "A", "prediction": 1.0, "target": 1.0},
    ]
    try:
        summarize_groups(rows_bad_dup)
    except ValueError as e:
        print("Rejected:", e)

    # 失败例：字符串 "nan"
    rows_bad_nan = [
        {"id": "x1", "group": "A", "prediction": "nan", "target": 0},
    ]
    try:
        summarize_groups(rows_bad_nan)
    except ValueError as e:
        print("Rejected:", e)


if __name__ == '__main__':
    main()