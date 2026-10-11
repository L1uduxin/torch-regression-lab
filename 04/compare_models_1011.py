import torch
import torch.nn as nn
from pathlib import Path
from run_04 import prepare_tensors
from experiment_flow_1008 import evaluate

DEVICE = torch.device("cpu")
DTYPE = torch.float64

DIC_DIR = Path(__file__).resolve().parent

#input
MODEL1_DIR = DIC_DIR / "run_1004.pt"
MODEL2_DIR = DIC_DIR / "mlp_1010_run01" / "mlp_step50_fixed_v2.pt"
TRAIN_PATH = DIC_DIR / "train_missing_1004.csv"
VALID_PATH = DIC_DIR / "validation_missing_1004.csv"

def _check_model(bundle_L, bundle_M):
    assert bundle_M["feature_names"] == bundle_L["feature_names"]
    assert bundle_M["fill_value"] == bundle_L["fill_value"]

def reload_model(bundle):
    cfg = bundle["model_config"]
    if "model_type" in cfg :
        torch.manual_seed(cfg["seed"])
        model = nn.Sequential(
            nn.Linear(cfg['input_dim'], cfg['hidden_dim']),
            nn.ReLU(),
            nn.Linear(cfg['hidden_dim'], cfg['output_dim']),
        ).to(DEVICE, DTYPE)
        model.load_state_dict(bundle["model.state_dict"])
    else :
        model = nn.Linear(cfg["input_dim"], cfg["output_dim"]).to(DEVICE, DTYPE)
        model.load_state_dict(bundle["model.state_dict"])
    model.eval()
    return model.to(device=DEVICE, dtype=DTYPE)


def main():

    bundle_L = torch.load(MODEL1_DIR, weights_only=True)
    bundle_M= torch.load(MODEL2_DIR)
    selected_cols = bundle_L["feature_names"]
    _check_model(bundle_L, bundle_M)
    model_L = reload_model(bundle_L)
    model_M = reload_model(bundle_M)
    fill_value = bundle_L["fill_value"]
    data = prepare_tensors(TRAIN_PATH, VALID_PATH,selected_cols,fill_value )

    x_train = data["x_train"].to(DEVICE, DTYPE)
    y_train = data["y_train"].to(DEVICE, DTYPE)
    x_valid = data["x_valid"].to(DEVICE, DTYPE)
    y_valid = data["y_valid"].to(DEVICE, DTYPE)

    mse_train_L = evaluate(model_L, x_train, y_train)
    mse_train_M = evaluate(model_M, x_train, y_train)
    mse_valid_L = evaluate(model_L, x_valid, y_valid)
    mse_valid_M = evaluate(model_M, x_valid, y_valid)
    print("mse_train_L", mse_train_L)
    print("mse_train_M", mse_train_M)
    print("mse_valid_L", mse_valid_L)
    print("mse_valid_M", mse_valid_M)

if __name__ == "__main__":
    main()


