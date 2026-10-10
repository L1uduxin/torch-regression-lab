import torch.nn.functional as F
from run_04 import prepare_tensors, _checkpoint, save_checkpoint
from pathlib import Path
import torch
import torch.nn as nn
from experiment_flow_1008 import evaluate


DATA_DIR = Path(__file__).resolve().parent
DICT_DIR = DATA_DIR.parent

TRAIN_PATH = DATA_DIR / "train_missing_1004.csv"
VALID_PATH = DATA_DIR / "validation_missing_1004.csv"
BATCH_PATH = DICT_DIR / "09" / "batch_1009.csv"
OUTPUT_PATH = DATA_DIR / "mlp_1010_run01" / "mlp_step50.pt"
FIXED_PATH = DATA_DIR / "mlp_1010_run01" / "mlp_step50_fixed.pt"
FIXED_PATH2 = DATA_DIR / "mlp_1010_run01" / "mlp_step50_fixed_v2.pt"

SELECTED_COLS = ['x1', 'x3']
SEED = 1009
DEVICE = torch.device("cpu")
DTYPE = torch.float64
config = {
        "input_dim": len(SELECTED_COLS),
        'output_dim': 1,
        "dtype" : "float64",
        "device": "cpu",
        "loss": "mean",
        "optimizer": "SGD",
        "lr": 0.01,
        "momentum": 0.0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "batch_mode": "full_batch",
        "init": "zeros",
        "target_updates": 50,
    }

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


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return float(value)


def train(model, optimizer, x_train, y_train, x_valid, y_valid, total_steps=50):
    # 初始评估（第0步）
    model.eval()
    with torch.no_grad():
        mse_t0 = evaluate(model, x_train, y_train)
        mse_v0 = evaluate(model, x_valid, y_valid)
    print(f"训练次数 : 0, mse_train = {_to_float(mse_t0)}, mse_valid = {_to_float(mse_v0)}")

    # 评估点：第1、25、50步（可自行增删）
    eval_steps = {1, 25, 50}

    model.train()
    for step in range(1, total_steps + 1):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x_train), y_train)
        loss.backward()
        _check_model(model)
        optimizer.step()

        if step in eval_steps:
            model.eval()
            with torch.no_grad():
                mse_t = evaluate(model, x_train, y_train)
                mse_v = evaluate(model, x_valid, y_valid)
            print(f"训练次数 : {step}, mse_train = {_to_float(mse_t)}, mse_valid = {_to_float(mse_v)}")
            model.train()

    return model, step


def fix():
    # 加载旧包（含 Series，用 weights_only=False）
    ckpt = torch.load(OUTPUT_PATH, map_location=DEVICE, weights_only=False)

    selected_cols = ckpt["feature_names"]
    old_fill = ckpt["fill_value"]

    old_state = ckpt["model.state_dict"]

    # 用保存的 selected_cols / 训练均值准备数据
    data = prepare_tensors(TRAIN_PATH, VALID_PATH, selected_cols, train_means=old_fill)
    x_train = data["x_train"].to(DEVICE, DTYPE)
    y_train = data["y_train"].to(DEVICE, DTYPE)
    x_valid = data["x_valid"].to(DEVICE, DTYPE)
    y_valid = data["y_valid"].to(DEVICE, DTYPE)

    # 构建相同 MLP 并载入权重
    model = build_mlp(input_dim=len(selected_cols), hidden_dim=4)
    model.load_state_dict(old_state)
    model.eval()

    # 核对恢复后的 MSE
    with torch.no_grad():
        mse_t = evaluate(model, x_train, y_train)
        mse_v = evaluate(model, x_valid, y_valid)
    print(f"恢复后 mse_train = {mse_t}")
    print(f"恢复后 mse_valid = {mse_v}")
    assert abs(mse_t - 2.275206495174139) < 1e-12, "训练 MSE 不匹配"
    assert abs(mse_v - 2.241739602722843) < 1e-12, "验证 MSE 不匹配"

    # 修正配置
    ckpt["model_config"] = {
        "model_type": "MLP",
        "input_dim": len(selected_cols),
        "hidden_dim": 4,
        "output_dim": 1,
        "activation": "ReLU",
        "seed": SEED,
        "dtype": "float64",
        "device": "cpu",
        "loss": "mean",
        "optimizer": "SGD",
        "lr": 0.01,
        "momentum": 0.0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "batch_mode": "full_batch",
        "init": "pytorch_default_random(kaiming_uniform+a_uniform)",
        "target_updates": 50,
    }

    # 填充值转普通字典
    ckpt["fill_value"] = old_fill.to_dict()

    # 计算验证预测
    with torch.no_grad():
        valid_pred = model(x_valid).detach().clone()
    ckpt["valid_pred"] = valid_pred

    # 保存修复包
    FIXED_PATH2.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, FIXED_PATH2)
    print(f"已保存: {FIXED_PATH2}")

    # 再载入新包（普通字典，用 weights_only=True）
    cfg = torch.load(FIXED_PATH2, map_location=DEVICE, weights_only=True)
    torch.manual_seed(cfg['model_config']["seed"])
    model = nn.Sequential(
        nn.Linear(cfg["model_config"]['input_dim'], cfg["model_config"]['hidden_dim']),
        nn.ReLU(),
        nn.Linear(cfg["model_config"]['hidden_dim'], cfg["model_config"]['output_dim']),
    ).to(DEVICE, DTYPE)
    model.load_state_dict(cfg["model.state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(x_valid)
    assert torch.allclose(pred, cfg["valid_pred"], rtol=0, atol=1e-12)




def main():
    selected_cols = SELECTED_COLS

    data = prepare_tensors(TRAIN_PATH, VALID_PATH, selected_cols)
    x_train = data['x_train'].to(device=DEVICE, dtype=DTYPE)
    y_train = data['y_train'].to(device=DEVICE, dtype=DTYPE)
    x_valid = data['x_valid'].to(device=DEVICE, dtype=DTYPE)
    y_valid = data['y_valid'].to(device=DEVICE, dtype=DTYPE)
    fill_value = data['train_means']


    model = build_mlp(input_dim=2, hidden_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0)

    model, step = train(model, optimizer, x_train, y_train, x_valid, y_valid, total_steps=50)

    checkpoint = _checkpoint(model, optimizer, step, selected_cols, fill_value, config)

    save_checkpoint(checkpoint, OUTPUT_PATH)



if __name__ == "__main__":

    #main()

    fix()