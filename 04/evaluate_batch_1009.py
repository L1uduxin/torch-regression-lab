import torch
from pathlib import Path
from run_04 import prepare_tensors
from experiment_flow_1008 import evaluate
import torch.nn as nn
import math
import sys

# 把 run_1009.py 所在目录加入模块搜索路径
sys.path.insert(0, r"E:\develop\PythonProject_10\09")

# 导入 run_1009.py
import run_1009 as run


DIR = Path(__file__).resolve().parent

#输入
MODEL_PATH = DIR / 'output_1007_1.pt'
TRAIN_PATH = DIR / 'train_missing_1004.csv'
BATCH_PATH = DIR.parent / '09' / 'batch_1009.csv'


def load_model(model_path):
    bundle = torch.load(model_path, map_location="cpu")

    config = bundle["model_config"]
    selected_cols = bundle["feature_names"]
    fill_value = bundle["fill_value"]

    dtype_name = config["dtype"].split(".")[-1]
    dtype = getattr(torch, dtype_name)
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = nn.Linear(config["input_dim"], config["output_dim"]).to(
        device=device, dtype=dtype
    )
    model.load_state_dict(bundle["model.state_dict"])
    model.eval()

    return model,selected_cols,fill_value




def evaluate_batch(model_path, batch_path):

    model, selected_cols, fill_value = load_model(model_path)

    data = prepare_tensors(TRAIN_PATH, batch_path, selected_cols, fill_value)

    x_torch = data["x_valid"].to(device="cpu", dtype=torch.float64)
    y_torch = data["y_valid"].to(device="cpu", dtype=torch.float64)

    mse_torch = evaluate(model, x_torch, y_torch)

    model.eval()
    with torch.no_grad():
        pred = model(x_torch)


    id_valid = data["id_valid"]
    group_valid = data["group_valid"]

    y_valid = data["y_valid"].reshape(-1)
    pred = pred.reshape(-1)

    # tensor -> numpy
    if hasattr(y_valid, "detach"):
        y_valid = y_valid.detach().cpu().numpy()
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()

    def to_py(v):
        # tensor/numpy 标量 -> Python 原生类型
        return v.item() if hasattr(v, "item") else v

    rows = []
    for i in range(len(id_valid)):
        rows.append({
            "id": to_py(id_valid[i]),
            "group": to_py(group_valid[i]),
            "prediction": float(pred[i]),
            "target": float(y_valid[i]),
        })

    result = run.summarize_groups(rows)
    count = 0
    loss_total = 0
    for item in result["per_group"]:
        num = result['per_group'][item]['n']
        count += num
        per_mse = result['per_group'][item]['mse']
        per_loss_total = num * per_mse
        loss_total += per_loss_total
        print(f"组{item}数量 : {num}, mse = {per_mse}")
    assert count == len(id_valid), "分组人数与输入人数不一致"
    mse_avg = loss_total / count
    mse_compute = result['overall_mse']


    print(f"样本数 : {count}")
    print(f"总体mse : { mse_compute}")
    print(f"加权mse : { mse_avg}")
    print(f"本批次直接总体MSE : {mse_torch}")
    assert math.isclose(mse_avg, mse_compute, rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(mse_torch, mse_compute, rel_tol=1e-10, abs_tol=1e-12)
    print("检查通过")







def mian():
    evaluate_batch(MODEL_PATH, BATCH_PATH)

if __name__ == "__main__":
    mian()