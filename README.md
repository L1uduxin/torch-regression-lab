# torch-regression-lab

Python／PyTorch 回归模型学习与实践仓库。目前处于基础工程实践阶段，围绕数据预处理、模型训练与保存、加载评估及分组误差分析开展练习，逐步提升实现、结果验证和工程组织能力。

## 2026-10-12：SMS读取、划分与保存

新增 `sms_classifier/` 将已有短信读取、原ID、精确正文分组和划分逻辑封装为独立模块，补齐可执行检查、六项摘要及JSON保存/读回。新包仅使用Python标准库，不拟合词表、不训练分类器。AI封装与新增实现的贡献边界及后续任务见[变更记录](CHANGELOG.md)，数据来源、许可和具体政策见[数据合同](DATA_CARD.md)。

在仓库根目录运行；下列 `PATH_TO_SMS` 和 `NEW_SPLIT_JSON` 需替换为本机原文件和新产物路径，输出父目录须已存在：

```shell
python -B -m sms_classifier prepare --data PATH_TO_SMS --output NEW_SPLIT_JSON --seed 20261012
python -B -m sms_classifier check --data PATH_TO_SMS --artifact NEW_SPLIT_JSON
python -B -m unittest discover -s tests -p "test_sms_data.py" -v
python -B -m unittest discover -s tests -p "test_*.py" -v
```

新测试只用人工数据，不下载数据、不训练模型。最后一条包含既有NumPy练习测试，需要已有NumPy环境。`prepare`完成读取、划分、输入保持/复现/隔离检查、写入及读回；`check`验证现有产物。成功退出0，数据或文件检查失败退出2；已有输出拒绝覆盖。stdout只有状态和计数，原短信与问题原行不写入CLI输出或JSON。

最小调用示例（`data_path` 为调用方提供的原始文件路径）：

```python
from sms_classifier import read_records, split_record_ids, validate_split, summarize_split

records = read_records(data_path)["records"]
split = split_record_ids(records, seed=20261012)
validate_split(records, split)
summary = summarize_split(records, split)
```

| 接口 | 返回与后续用途 |
|---|---|
| `check_line(line)` / `read_records(path)` | 单行标签正文／总行数、原ID记录、问题列表；为数据卡和训练输入提供来源 |
| `split_record_ids(records, seed)` | train_ids、val_ids、test_ids；冻结后供后续训练／评价选行 |
| `validate_split(records, split)` | 成功检查字典或异常；检查ID完整和精确正文隔离 |
| `summarize_split(records, split)` | 三个集合各六项计数；供数据报告及读回对照 |
| `artifacts.prepare_artifact(data_path, seed)` | 构造含规则、来源hash、计数和ID的字典；不写文件 |
| `artifacts.save_artifact(artifact, output_path)` | 独占新建JSON；调用方应先prepare，完整源绑定校验在load/verify中 |
| `artifacts.load_artifact(artifact_path, data_path)` | 全部验证后返回产物；供后续流程复用固定划分 |
| `artifacts.verify_artifact(artifact, data_path)` | 对已有字典验证并返回检查和摘要 |

以下日期目录保留既有实验内容。短信新包通过相对导入自包含，不依赖旧 `11` 目录的绝对路径。旧回归入口仍需要原本地数据和权重；本次没有全量迁移历史训练项目或重训。

## 当前范围

当前收录新批次评价流程、MLP前向／单步更新／短训练与恢复源码，以及模型比较、NumPy聚类迭代和人工文本质量检查练习。

| 文件 | 用途 |
| --- | --- |
| `04/evaluate_batch_1009.py` | 加载已有模型，评价新批次并检查分组汇总一致性 |
| `04/run_04.py` | 复用数据准备函数 |
| `04/experiment_flow_1008.py` | 复用模型评价函数 |
| `09/run_1009.py` | 分组计数和MSE汇总，附小型示例 |
| `09/batch_1009.csv` | 4条人工样本 |
| `04/mlp_forward_1009.py` | 2→4→1 MLP前向、MSE与张量信息检查 |
| `04/mlp_step_1010.py` | 一次SGD更新、梯度检查及评价前后参数核对 |
| `04/mlp_train_1010.py` | 50次更新流程，以及旧模型包配置修正和恢复预测检查 |
| `10/teacher_t01_1010.py` | 建立六个二维位置点，保存／载入NumPy数组并检查一致性 |
| `10/teacher_t02_1010.py` | 计算样本到固定中心的平方距离，并选择最近中心 |
| `04/compare_models_1011.py` | 恢复Linear和MLP，在同一数据划分与预处理下比较四项MSE |
| `10/teacher_t03_1011.py` | 按当前分组求中心均值，并检查平方距离目标 |
| `10/teacher_t04_1011.py` | 交替分配与更新中心，记录每轮目标和停止原因 |
| `11/basic_test.py` | 按原文分组并筛出重复文本 |
| `11/basic_test_01.py` | 分离标签和正文，保留正文内部制表符 |
| `11/basic_test_02.py` | 构建带原编号的记录，识别同一文本的标签冲突 |
| `tests/test_1011_exercises.py` | 人工输入上的文本处理与聚类回归检查 |

## 2026-10-10：MLP与NumPy练习

MLP结构为`Linear(2, 4) → ReLU → Linear(4, 1)`，使用CPU、float64、seed=1009；目标为mean MSE，SGD学习率0.01、momentum=0、weight_decay=0。本机一次50步训练得到训练MSE `2.275206495174139`、验证MSE `2.241739602722843`。恢复检查使用`rtol=0, atol=1e-12`比较同输入的保存预测与恢复后预测。这些是小型练习数据上的结果，不能据此判断真实场景表现或架构优劣。

三个MLP脚本的当前行为：

- `mlp_forward_1009.py`只做前向与评价，需要本地训练CSV及已收录的人工批次CSV。
- `mlp_step_1010.py`执行一次训练更新并打印检查结果，不保存模型，需要本地训练／验证CSV。
- `mlp_train_1010.py`当前直接执行入口为`fix()`，读取`04/mlp_1010_run01/mlp_step50.pt`，写入或覆盖同目录的`mlp_step50_fixed_v2.pt`并核对恢复预测，不新增训练。旧包包含pandas对象，使用`weights_only=False`加载，仅用于已知来源的本地旧包。

文件中保留的训练`main()`不是当前入口，旧`config`仍有过时的初始化描述；有效修复元数据由`fix()`中的`model_config`写入。此文件保留练习过程，不作为可直接切换入口的完整训练发布程序。本次只更新源码和说明，没有为发布重跑训练或重新生成模型。

NumPy练习使用六个人工位置点，两列分别为横、纵坐标，单位相同。固定中心为`(1,1)`和`(7,7)`，平方距离表形状为`(6,2)`，分配索引为`[0,0,0,1,1,1]`；第二个样本的手算参照为`[1,61]`，对应中心索引0。这是10/10固定中心分配阶段的结果；10/11新增中心更新与迭代，见下节。

NumPy示例只需要已有Python环境中的NumPy。从仓库的`10`目录依次运行：

```shell
python teacher_t01_1010.py
python teacher_t02_1010.py
```

第一个脚本会生成或覆盖脚本目录的`teacher_t01_points_1010.npy`；第二个脚本按当前工作目录读取该文件，仅打印距离、分配与检查结果。生成的`.npy`不纳入版本控制。

## 2026-10-11：比较、聚类迭代与文本检查

学习阶段继续使用小型人工样例，代码保留当前接口和适用范围。

比较入口使用无正则Linear包`04/run_1004.pt`和MLP包`04/mlp_1010_run01/mlp_step50_fixed_v2.pt`。特征顺序从模型包的`feature_names`读取；共用训练集填充值和同一份数据准备结果。入口只恢复、评价，不训练或写模型。

| 模型 | 训练MSE | 验证MSE | 初始化与已有更新预算 |
| --- | ---: | ---: | --- |
| Linear | 0.7355495778243006 | 0.4039266137632595 | 零初始化；SGD lr=0.05、momentum=0.9；300次更新 |
| MLP | 2.275206495174139 | 2.241739602722843 | seed1009随机初始化；SGD lr=0.01、momentum=0；50次更新 |

两者均为CPU、float64和纯mean MSE。初始化、优化参数及更新预算不同，上述一次结果不能用于证明架构优劣。本地模型及训练／验证CSV不随仓库发布；具备这些文件后才可运行`python -B 04/compare_models_1011.py`。

聚类沿用六个人工二维点，初始中心为`(1,1)`和`(7,7)`。在`max_iter=5`下，记录了2次均值更新，两轮平方距离和J均为`2.666666666666666`，中心为`(4/3,4/3)`和`(22/3,22/3)`，分配为`[0,0,0,1,1,1]`，停止原因为中心不变。J是平方距离总和。`len(history)`给出实际更新次数；停止轮也保留记录。

当前聚类代码仅面向二维输入、完整的0/1簇编号、两簇非空的练习范围；空簇、非连续簇编号以及一般初始化策略尚未覆盖。中心比较使用精确相等。若仅因最大轮数停止，返回的labels是最后用于求均值的分配，未额外按新中心重新分配。

`10/teacher_t03_1011.py`和`10/teacher_t04_1011.py`从各自脚本目录读取T01生成的`.npy`；两个新入口不保存数组或模型。已有数组时可分别运行它们，无需重新生成旧数据。

文本脚本只使用源码内的人工短句，没有纳入真实短信数据集。解析时仅移除行尾CR/LF，按第一个制表符切分；原编号保留，ham映射0、spam映射1。重复检查按原文精确匹配；同一文本出现不同标签时返回该组全部原编号，并不自动判断哪条标签错误。输入合同是合法的ham/spam行；标签冲突helper接收非空0/1标签列表，该历史练习本身不包含文件加载、数据划分或分类模型；当前数据准备见上方sms_classifier包。

从仓库根目录，用已有Python和NumPy运行检查：

```shell
python -B -m unittest discover -s tests -p "test_1011_exercises.py" -v
```

测试仅在内存中调用文本与聚类函数，不执行旧训练入口，不写模型或生成数据文件。

## 运行条件与当前限制

MLP与回归流程依赖Python、PyTorch、pandas、NumPy；NumPy练习仅依赖NumPy。当前保留现有目录结构，MLP与回归示例所需的本地模型和训练数据未随源码发布，尚非克隆后即可完整运行的发布包。

新批次评价入口还需要本机保留的`04/output_1007_1.pt`和`04/train_missing_1004.csv`，MLP更新与恢复另外需要`04/validation_missing_1004.csv`及相应本地模型包。这些文件未随本次源码发布。入口从模型包读取特征顺序、填充值及参数；当前模型是CPU／float64的`Linear(2, 1)`，特征顺序为`[x1, x3]`。

入口目前通过绝对路径`E:/develop/PythonProject_10/09`导入分组函数，迁移目录时需调整此处。满足上述本地条件后，在项目根目录使用已有环境运行：

```powershell
& 'E:/develop/PythonProject/.venv-torch/Scripts/python.exe' -B './04/evaluate_batch_1009.py'
```

评价入口只加载和评价，不训练或保存模型。`04/run_04.py`的旧主入口包含训练及覆盖旧产物的行为；这里仅导入它的数据准备函数，不将旧主入口作为复现本次结果的命令。

## 人工样例来源

`batch_1009.csv`是用于验证新批次评价流程的人工数据，包含A组3条、B组1条。标签在完整特征上由`y = 1.5*x1 - 0.8*x3 + 0.3 + epsilon`生成，四行固定扰动依次为`0.04, -0.03, 0.02, -0.01`，再将第2行的x1（原值-0.2）和第4行的x3（原值0.9）置空。模型使用包内填充值处理缺失，不由标签反推缺失特征。

group是样例中的分组标签，不参与模型输入；x2、x4是未使用的辅助列。这批数据不是现实采集样本，也不是模型预测生成的标签，仅用于流程与汇总检查。

## 本机实际结果

2026-10-09使用已有模型评价4条样本，未新增训练：

| 范围 | 样本数 | MSE |
| --- | ---: | ---: |
| A组 | 3 | 0.10115382722097378 |
| B组 | 1 | 0.16912619405359355 |
| 直接总体 | 4 | 0.11814691892912872 |
| 按组人数加权总体 | 4 | 0.11814691892912874 |

已检查分组人数总和等于输入人数，且MSE在`rel_tol=1e-10, abs_tol=1e-12`下相符。该检查有助于发现计数、权重和汇总错误，不能排除两条路径共有的预测或标签错配，也不证明真实泛化能力。

## 版本控制范围

`.gitignore`暂以列出的文件为收录范围；后续加入新模块时同时更新规则。模型、旧数据、生成输出、虚拟环境和IDE配置留在本地。
