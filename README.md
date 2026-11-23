# GraphML CAN复现

把本文件夹完整复制到任意位置即可使用，不依赖外层课程目录。两个 Python 文件分别负责本地重训和作者公开结果复算。

| 文件               | 用途                                                         |
| ------------------ | ------------------------------------------------------------ |
| reproduce.py       | 从 ROAD 日志提取图嵌入和信号统计，训练分类器并评估；也支持构图计时 |
| check_published.py | 复算作者公开指标的描述统计、检验与热图                       |
| requirements.txt   | 固定 Python 依赖版本                                         |
| data/road.zip      | 完整官方 ROAD 数据包，无需解压，约 531 MiB                   |
| data/published/    | 11 份作者指标 CSV 及原仓库 MIT 许可                          |

## 1 安装环境

使用 Windows、64 位 Python 3.12 和 CPU。打开本文件夹中的 PowerShell，执行一次：

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

安装依赖需要联网。数据已包含在文件夹内，安装完成后实验可离线运行。不需要 GPU、Jupyter 或原作者仓库。

## 2 完整运行两组本地实验

在同一个 PowerShell 窗口中复制执行：

```powershell
$env:PYTHONHASHSEED = '42'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$py = './.venv/Scripts/python.exe'

# 8 秒窗、8 秒步长；随机森林和 XGBoost
& $py -u reproduce.py --road data/road.zip --out results/seconds-w8-o8 --window 8 --offset 8 --jobs 8
if ($LASTEXITCODE -ne 0) { throw '8 秒实验失败，请先检查上方错误' }

# 400 帧窗、400 帧步长；随机森林
& $py -u reproduce.py --road data/road.zip --out results/frames-w400-o400 --unit frames --window 400 --offset 400 --ambient ambient_dyno_drive_basic_long --models rf --jobs 12
if ($LASTEXITCODE -ne 0) { throw '400 帧实验失败，请先检查上方错误' }
```

两组配置分别提取 1,410 和 10,679 个窗口，均评估仅图嵌入、仅信号、融合三种特征，默认运行两种五折方式：

- capture：按录制分组，同一录制不跨训练和测试。
- window：按窗口分层，同一录制的不同窗口可跨训练和测试。

资源较少时降低 --jobs；这会改变运行耗时。按已有运行记录，完整特征提取耗时约数分钟到半小时，具体取决于电脑。进程会按录制输出进度；同一输出目录内已完成的录制可以复用。若要从头重新提取，指定一个尚不存在的 --out 目录。

## 3 其他课程稿实验

以下命令在第 2 步设置的变量仍有效时运行：

```powershell
# 仅测构图，使用独立目录以保留原训练记录
& $py -u reproduce.py --road data/road.zip --out results/graph-timing-w8-o8 --window 8 --offset 8 --ttw-only

# 复算作者公开统计和时间窗热图
& $py check_published.py
```

构图计时不包含日志读取、信号提取、Node2Vec 或分类，不能当作端到端检测时延。check_published.py 的结果来自作者公开 CSV，不是本地新训练。

## 4 检查结果

本地实验输出：

- metrics.csv：总体及每类场景指标；筛选 scenario=pooled、model=rf、features=hybrid 查看下表。
- predictions.csv：逐窗口标签、测试折及折外概率。
- features.csv 和 features-config.json：提取的特征及其配置。
- run.json：数据校验值、软件版本、随机种子和耗时。
- capture-features/：按录制保存的中间特征，用于继续未完成任务。

| 配置   | 窗口数 | 正例数 | capture ROC-AUC | window ROC-AUC |
| ------ | -----: | -----: | --------------: | -------------: |
| 8 秒   |  1,410 |     47 |          0.9843 |         0.9880 |
| 400 帧 | 10,679 |  1,753 |          0.6185 |         0.9851 |

误报和漏报采用概率 0.5 阈值。浮点运算、线程与软件平台可能导致末位差异。400 帧配置仅有一段正常录制，评估分布与 8 秒配置不同。

若仅重新训练分类器，在对应的第 2 步命令末尾加 --reuse-features。需要先完成该配置的特征提取，并保持 --out、--unit、--window、--offset 和 --ambient 一致；该操作会更新该目录的预测及指标文件。

作者结果复算写入 results/published/roc-summary-and-tests.csv 和 rf-time-heatmap.png。原论文表 VII、IX 的 KS 报告口径对应 ks_notebook_p 列。

## 来源与范围

本地脚本根据原论文及公开实现适配，完成公开 ROAD 日志上的训练与评估。原论文缺少完整的训练测试清单和部分预处理依赖，因此本地结果不等同于原测试集的逐项复刻。两个脚本均为本地适配；其中 reproduce.py 与课程项目脚本相同，check_published.py 仅调整了公开 CSV 的读取目录。

- 论文：Detecting Masquerade Attacks in Controller Area Networks Using Graph Machine Learning，IEEE TIFS，2025，20：13127–13142，DOI 10.1109/TIFS.2025.3636019。
- 作者代码：https://github.com/billmj/GraphML-CONTROLLER-AREA-NETWORK-Attack-Detection
- 固定提交：f9d2c99991f1030c3ba00738d8431be8af0b4769。
- ROAD：https://zenodo.org/records/10462796
- road.zip 大小：556,718,276 字节；MD5：cab184cfc2fe12c0834bc46188c0f330。

为保持包体精简，文件夹不包含已安装环境、预计算特征、实验结果或论文文档。运行命令会生成这些实验输出；环境通过 requirements.txt 安装。
