import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# 输入：工作区 10/4 数据目录
DATA_DIR = SCRIPT_DIR
TRAIN_PATH = DATA_DIR / "train_missing_1004.csv"
VALID_PATH = DATA_DIR / "validation_missing_1004.csv"

# 输出：以 10/4 脚本目录为基准
CHECKPOINT_PATH = SCRIPT_DIR / "run_1004.pt"
RESULT_PATH = SCRIPT_DIR / "result_04_01.csv"

ALL_COLS = ['x1','x2','x3','x4']

def validate_selected_cols(selected_cols):
    if not isinstance(selected_cols, list):
        raise TypeError('selected_cols must be a list')
    if not selected_cols:
        raise ValueError('selected_cols cannot be empty')
    if len(selected_cols) != len(set(selected_cols)):
        raise ValueError('selected_cols must have unique values')
    invalid_cols = [col for col in selected_cols if col not in ALL_COLS]
    if invalid_cols:
        raise ValueError(f'selected_cols contains invalid values{invalid_cols}')
    return selected_cols

def read_csv(path, selected_cols):
    df = pd.read_csv(path)
    selected_cols = validate_selected_cols(selected_cols)
    need_cols = selected_cols + ['y', "row_id", "group"]
    missing_cols = [c for c in need_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f'missing columns: {missing_cols}')

    if df['row_id'].isnull().values.any():
        raise ValueError('row_ids cannot be null')
    if not df['row_id'].is_unique:
        raise ValueError('row_ids must have unique values')
    if df['group'].isnull().values.any():
        raise ValueError('groups cannot be null')
    if df['y'].isnull().values.any():
        raise ValueError('y cannot be null')

    data = df[['row_id']+ ['group'] + selected_cols + ['y']].copy()
    data[selected_cols] = data[selected_cols].astype(np.float64)
    data['y'] = data['y'].astype(np.float64)

    return data


def prepare_tensors(train_path, valid_path, selected_cols,train_means = None):
    selected_cols = validate_selected_cols(selected_cols)

    train_data = read_csv(train_path, selected_cols)
    valid_data = read_csv(valid_path, selected_cols)

    overlap = set(train_data["row_id"]) & set(valid_data["row_id"])
    if overlap:
        raise ValueError(f"训练集和验证集 row_id 存在重叠: {sorted(overlap)}")

    if train_means is None:
        train_means = train_data[selected_cols].mean(numeric_only=True)
        if train_means.isna().any():
            bad_cols = train_means[train_means.isna()].index.tolist()
            raise ValueError(f"训练集选定列无法计算均值，可能整列为空: {bad_cols}")


    train_filled = train_data.copy()
    valid_filled = valid_data.copy()

    train_filled[selected_cols] = train_filled[selected_cols].fillna(train_means)
    valid_filled[selected_cols] = valid_filled[selected_cols].fillna(train_means)

    x_train = torch.tensor(
        train_filled[selected_cols].to_numpy(dtype=np.float64),
        dtype=torch.float64,
    )
    y_train = torch.tensor(
        train_filled["y"].to_numpy(dtype=np.float64),
        dtype=torch.float64,
    ).view(-1, 1)

    x_valid = torch.tensor(
        valid_filled[selected_cols].to_numpy(dtype=np.float64),
        dtype=torch.float64,
    )
    y_valid = torch.tensor(
        valid_filled["y"].to_numpy(dtype=np.float64),
        dtype=torch.float64,
    ).view(-1, 1)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "id_train": train_filled["row_id"].copy(),
        "group_train": train_filled["group"].copy(),
        "group_valid": valid_filled["group"].copy(),

        "x_valid": x_valid,
        "y_valid": y_valid,
        "id_valid": valid_filled["row_id"].copy(),

        "train_means": train_means,
        "train_filled": train_filled,
        "valid_filled": valid_filled,

        "train_original": train_data,
        "valid_original": valid_data,
    }


def _model_config(selected_cols):
    selected_cols = validate_selected_cols(selected_cols)
    return {
        "input_dim": len(selected_cols),
        'output_dim': 1,
        "dtype" : "float64",
        "device": "cpu",
        "loss": "mean",
        "optimizer": "SGD",
        "lr": 0.05,
        "momentum": 0.9,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "batch_mode": "full_batch",
        "init": "zeros",
        "target_updates": 300,
    }

def create_training_obj(model_config):
    dtype = getattr(torch, model_config['dtype'])
    device = torch.device(model_config.get('device', 'cpu'))

    model = nn.Linear(
        model_config['input_dim'],
        model_config['output_dim'],
    ).to(device=device, dtype=dtype)

    init = model_config.get('init', 'zeros')
    if init == 'zeros':
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
    else:
        raise ValueError(f'Unsupported init: {init}')

    loss_name = model_config.get('loss', 'MSELoss')
    if loss_name != "mean":
        raise ValueError(f'Unsupported loss: {loss_name}')

    criterion = nn.MSELoss(reduction=model_config.get('reduction', 'mean'))

    optimizer_name = model_config.get('optimizer', 'SGD')
    if optimizer_name != 'SGD':
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=model_config['lr'],
        momentum=model_config['momentum'],
        dampening=model_config.get('dampening', 0),
        weight_decay=model_config.get('weight_decay', 0),
        nesterov=model_config.get('nesterov', False),
    )

    return model, optimizer


def run_updates(model, optimizer,x_train, y_train, n_updates, completed_updates):

    criterion = nn.MSELoss(reduction="mean")
    for _ in range(n_updates):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    return completed_updates + n_updates

def _checkpoint(model, optimizer, completed_updates, selected_cols, fill_value, model_config=None):
    if model_config is None:
        model_config = _model_config(selected_cols)

    return {
        'model.state_dict': model.state_dict(),
        'optimizer.state_dict': optimizer.state_dict(),
        'completed_updates': completed_updates,
        'feature_names': selected_cols,
        'fill_value': fill_value,
        'model_config': model_config,
    }


def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main():
    selected_cols = ['x1', 'x3']

    data = prepare_tensors(
        str(TRAIN_PATH),
        str(VALID_PATH),
        selected_cols,
    )

    config = _model_config(selected_cols)
    model, optimizer = create_training_obj(config)

    completed_updates = run_updates(
        model,
        optimizer,
        data['x_train'],
        data['y_train'],
        config['target_updates'],
        0,
    )

    checkpoint = _checkpoint(
        model,
        optimizer,
        completed_updates,
        selected_cols,
        data['train_means'].to_dict(),
        config,
    )
    save_checkpoint(checkpoint, CHECKPOINT_PATH)

    model.eval()
    with torch.no_grad():
        pred_t = model(data['x_train'])
        mse_t = F.mse_loss(pred_t, data['y_train'])
        pred_v = model(data['x_valid'])
        mse_v = F.mse_loss(pred_v, data['y_valid'])
        print(mse_t.item(), mse_v.item())

    result = pd.DataFrame({
        'row_id': data['id_valid'].to_numpy(),
        'group': data['group_valid'].to_numpy(),
        'y_true': data['y_valid'].detach().cpu().numpy().reshape(-1),
        'y_pred': pred_v.detach().cpu().numpy().reshape(-1),
    })
    print(result.head())
    result.to_csv(RESULT_PATH, index=False)

if __name__ == "__main__":
    main()
