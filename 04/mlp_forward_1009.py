from run_04 import prepare_tensors
from pathlib import Path
import torch
import torch.nn as nn


DATA_DIR = Path(__file__).resolve().parent
DICT_DIR = DATA_DIR.parent

#输入
TRAIN_PATH = DATA_DIR / "train_missing_1004.csv"
BATCH_PATH = DICT_DIR / "09" / "batch_1009.csv"

#常量设置
SEED = 1009
DEVICE = torch.device("cpu")
DTYPE = torch.float64


def build_mlp(input_dim=2, hidden_dim=4):
    # 返回构造好的模型
    torch.manual_seed(SEED)

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),  # Linear1: (N, 2) -> (N, 4)
        nn.ReLU(),  # 激活:   (N, 4) -> (N, 4)
        nn.Linear(hidden_dim, 1),  # Linear2: (N, 4) -> (N, 1)
    )

    return model.to(device=DEVICE, dtype=DTYPE)

def main():
    model = build_mlp(input_dim=2, hidden_dim=4)
    selected_clos = ['x1', 'x3']
    data = prepare_tensors(TRAIN_PATH, BATCH_PATH, selected_clos)
    x = data["x_valid"].to(device=DEVICE, dtype=DTYPE)
    y = data["y_valid"].to(device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        h1 = model[0](x).to(device=DEVICE, dtype=DTYPE)
        h2 = model[1](h1).to(device=DEVICE, dtype=DTYPE)
        y_hat = model(x).to(device=DEVICE, dtype=DTYPE)



    loss = nn.functional.mse_loss(y_hat, y)
    print("MSE loss =", loss.item())
    for name, t in [("x", x), ("y", y), ("h1", h1), ("h2", h2), ("y_hat", y_hat)]:
        print(name, tuple(t.shape), t.dtype, t.device)

    for name, p in model.named_parameters():
        print(name, tuple(p.shape), p.dtype, p.device)

if __name__ == "__main__":
    main()