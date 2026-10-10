import torch.nn.functional as F
from run_04 import prepare_tensors
from pathlib import Path
import torch
import torch.nn as nn
from experiment_flow_1008 import evaluate


DATA_DIR = Path(__file__).resolve().parent
DICT_DIR = DATA_DIR.parent

TRAIN_PATH = DATA_DIR / "train_missing_1004.csv"
VALID_PATH = DATA_DIR / "validation_missing_1004.csv"
BATCH_PATH = DICT_DIR / "09" / "batch_1009.csv"

SELECTED_COLS = ['x1', 'x3']
SEED = 1009
DEVICE = torch.device("cpu")
DTYPE = torch.float64


def build_mlp(input_dim=2, hidden_dim=4):

    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    return model.to(device=DEVICE, dtype=DTYPE)


def _check_model(model):
    linear1 = model[0]
    linear2 = model[2]
    params = [
        ("linear1.weight", linear1.weight),
        ("linear1.bias", linear1.bias),
        ("linear2.weight", linear2.weight),
        ("linear2.bias", linear2.bias),
    ]
    for name, param in params:
        if param.grad is None:
            raise RuntimeError(f"{name} has no gradient")
        if not torch.isfinite(param).all():
            raise RuntimeError(f"{name} has infinite values")
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(f"{name} grad has infinite values")
    if linear1.weight.grad.shape != (4, 2):
        raise RuntimeError("linear1.weight grad shape mismatch")
    if linear1.bias.grad.shape != (4,):
        raise RuntimeError("linear1.bias grad shape mismatch")
    if linear2.weight.grad.shape != (1, 4):
        raise RuntimeError("linear2.weight grad shape mismatch")
    if linear2.bias.grad.shape != (1,):
        raise RuntimeError("linear2.bias grad shape mismatch")


def one_step(model, optimizer, x_train, y_train):
    model.eval()
    with torch.no_grad():
        loss_init = F.mse_loss(model(x_train), y_train)
    model.train()
    for _ in range(1):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x_train), y_train)
        loss.backward()
        _check_model(model)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        loss_after = F.mse_loss(model(x_train), y_train)
    return loss_init.item(), loss_after.item()


def show_tensor_info(name, tensor):
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return float(value)


def main():
    selected_cols = SELECTED_COLS

    data = prepare_tensors(TRAIN_PATH, VALID_PATH, selected_cols)
    x_train = data['x_train'].to(device=DEVICE, dtype=DTYPE)
    y_train = data['y_train'].to(device=DEVICE, dtype=DTYPE)
    x_valid = data['x_valid'].to(device=DEVICE, dtype=DTYPE)
    y_valid = data['y_valid'].to(device=DEVICE, dtype=DTYPE)

    print("=== 张量信息 ===")
    show_tensor_info("x_train", x_train)
    show_tensor_info("y_train", y_train)
    show_tensor_info("x_valid", x_valid)
    show_tensor_info("y_valid", y_valid)

    model = build_mlp(input_dim=2, hidden_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0)

    print("=== 模型参数信息 ===")
    show_tensor_info("model[0].weight", model[0].weight)
    show_tensor_info("model[0].bias", model[0].bias)
    show_tensor_info("model[2].weight", model[2].weight)
    show_tensor_info("model[2].bias", model[2].bias)

    w0_before = model[0].weight.clone()
    w2_before = model[2].weight.clone()
    b0_before = model[0].bias.clone()
    b2_before = model[2].bias.clone()

    loss_init, loss_after = one_step(model, optimizer, x_train, y_train)

    model.eval()
    w0_after = model[0].weight.clone()
    w2_after = model[2].weight.clone()
    b0_after = model[0].bias.clone()
    b2_after = model[2].bias.clone()

    print("=== one_step 更新前后参数 ===")
    print("w0_before:", w0_before)
    print("w2_before:", w2_before)
    print("b0_before:", b0_before)
    print("b2_before:", b2_before)
    print("w0_after:", w0_after)
    print("w2_after:", w2_after)
    print("b0_after:", b0_after)
    print("b2_after:", b2_after)

    # evaluate 前后快照
    w0_before_t = model[0].weight.clone()
    w2_before_t = model[2].weight.clone()
    b0_before_t = model[0].bias.clone()
    b2_before_t = model[2].bias.clone()

    mse_t = evaluate(model, x_train, y_train)
    mse_v = evaluate(model, x_valid, y_valid)

    w0_after_t = model[0].weight.clone()
    w2_after_t = model[2].weight.clone()
    b0_after_t = model[0].bias.clone()
    b2_after_t = model[2].bias.clone()

    same_w0 = torch.equal(w0_before_t, w0_after_t)
    same_w2 = torch.equal(w2_before_t, w2_after_t)
    same_b0 = torch.equal(b0_before_t, b0_after_t)
    same_b2 = torch.equal(b2_before_t, b2_after_t)

    print("=== evaluate 前后参数快照比较 ===")
    print(f"model[0].weight 保持不变: {same_w0}")
    print(f"model[2].weight 保持不变: {same_w2}")
    print(f"model[0].bias   保持不变: {same_b0}")
    print(f"model[2].bias   保持不变: {same_b2}")
    print(f"四组是否全部保持不变: {all([same_w0, same_w2, same_b0, same_b2])}")

    print("=== 指标 ===")
    print(f"loss_init  (更新前训练MSE): {loss_init}")
    print(f"loss_after (更新后训练MSE): {loss_after}")
    print(f"mse_t      (evaluate 更新后训练MSE): {_to_float(mse_t)}")
    print(f"mse_v      (evaluate 更新后验证MSE): {_to_float(mse_v)}")


if __name__ == "__main__":
    main()