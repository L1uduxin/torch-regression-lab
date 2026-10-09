from run_04 import prepare_tensors
import math
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR
TRAIN_PATH = DATA_DIR / "train_missing_1004.csv"
VALID_PATH = DATA_DIR / "validation_missing_1004.csv"
MODEL_PATH = DATA_DIR / "output_1007_1.pt"


def _assert_finite(name, t):
    if not torch.isfinite(t).all().item():
        raise ValueError(f"{name} 含有非有限值")


def evaluate(model, x, y):
    # 1) 校验 x、y 是否有限，报错信息区分 x / y
    _assert_finite("x", x)
    _assert_finite("y", y)

    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    # 2) 前向
    with torch.no_grad():
        pred = model(x)
        _assert_finite("pred", pred)

    # 3) 校验 pred 与 y 形状
    if pred.shape != y.shape:
        raise ValueError(
            f"预测形状 {tuple(pred.shape)} 与标签形状 {tuple(y.shape)} 不一致"
        )

    # 4) 计算 MSE
    with torch.no_grad():
        mse = F.mse_loss(pred, y)
        _assert_finite("mse", mse)

    return mse.item()


def main():
    bundle = torch.load(MODEL_PATH, map_location="cpu")

    config = bundle["model_config"]
    selected_clos = bundle["feature_names"]
    fill_value = bundle["fill_value"]
    data = prepare_tensors(TRAIN_PATH, VALID_PATH, selected_clos, fill_value)

    dtype_name = config["dtype"].split(".")[-1]
    dtype = getattr(torch, dtype_name)
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = nn.Linear(config["input_dim"], config["output_dim"]).to(
        device=device, dtype=dtype
    )
    model.load_state_dict(bundle["model.state_dict"])

    x_train = data["x_train"].to(device=device, dtype=dtype)
    y_train = data["y_train"].to(device=device, dtype=dtype)
    x_valid = data["x_valid"].to(device=device, dtype=dtype)
    y_valid = data["y_valid"].to(device=device, dtype=dtype)

    # ========== 1. 正常评价 ==========

    before_weight = model.weight.detach().clone()
    before_bias = model.bias.detach().clone()

    mse_train = evaluate(model, x_train, y_train)
    mse_val = evaluate(model, x_valid, y_valid)

    after_weight = model.weight.detach().clone()
    after_bias = model.bias.detach().clone()

    assert torch.equal(before_weight, after_weight), "权重被改变"
    assert torch.equal(before_bias, after_bias), "偏置被改变"

    assert math.isclose(
        mse_train,
        0.772013969883223,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ), f"train MSE 不符合预期: got {mse_train!r}, expected 0.772013969883223"

    assert math.isclose(
        mse_val,
        0.4170254373923897,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ), f"valid MSE 不符合预期: got {mse_val!r}, expected 0.4170254373923897"

    print(f"train MSE = {mse_train:.6f}")
    print(f"valid MSE = {mse_val:.6f}")
    print("权重、偏置前后不变: OK")

    # ========== 2. x 单独含非有限值 ==========
    print("\n--- x 单独含非有限值 ---")

    x_nan = x_train.clone()
    x_nan[0, 0] = float("nan")
    try:
        evaluate(model, x_nan, y_train)
    except ValueError as e:
        assert "x" in str(e) and "非有限值" in str(e), (
            f"x 含 NaN 时错误信息不符合预期: {e}"
        )
    else:
        raise AssertionError("x 含 NaN 时应被拒绝")

    x_inf = x_train.clone()
    x_inf[0, 0] = float("inf")
    try:
        evaluate(model, x_inf, y_train)
    except ValueError as e:
        assert "x" in str(e) and "非有限值" in str(e), (
            f"x 含 Inf 时错误信息不符合预期: {e}"
        )
    else:
        raise AssertionError("x 含 Inf 时应被拒绝")

    # ========== 3. y 单独含非有限值 ==========
    print("\n--- y 单独含非有限值 ---")

    y_nan = y_train.clone()
    y_nan[0, 0] = float("nan")
    try:
        evaluate(model, x_train, y_nan)
    except ValueError as e:
        assert "y" in str(e) and "非有限值" in str(e), (
            f"y 含 NaN 时错误信息不符合预期: {e}"
        )
    else:
        raise AssertionError("y 含 NaN 时应被拒绝")

    y_inf = y_train.clone()
    y_inf[0, 0] = float("inf")
    try:
        evaluate(model, x_train, y_inf)
    except ValueError as e:
        assert "y" in str(e) and "非有限值" in str(e), (
            f"y 含 Inf 时错误信息不符合预期: {e}"
        )
    else:
        raise AssertionError("y 含 Inf 时应被拒绝")

    # ========== 4. 预测与标签形状不一致 ==========
    print("\n--- 预测与标签形状不一致 ---")

    if y_train.shape[0] > 1:
        y_bad_shape = y_train[:-1]
    else:
        y_bad_shape = y_train[:0]

    try:
        evaluate(model, x_train, y_bad_shape)
    except ValueError as e:
        assert (
            "预测形状" in str(e)
            and "标签形状" in str(e)
            and "不一致" in str(e)
        ), f"形状不一致时错误信息不符合预期: {e}"
    else:
        raise AssertionError("预测与标签形状不一致时应被拒绝")


if __name__ == "__main__":
    main()