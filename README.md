# General Model Registration

General Model Registration 是一个面向 Windows 的纯 CPU 三角网格刚性配准工具。它将一个或多个浮动 STL 按顺序、相互独立地移动到同一个固定 STL 坐标系，并输出刚性变换、配准后的 STL、连续偏差色图、日志和质量报告。

当前版本：**v1.4.1**

## 主要功能

- FPFH 特征与多候选全局粗配准。
- 可选的彻底朝向检查：枚举 24 个合法 PCA 轴姿态，并为近似轴对称模型补充绕轴候选。
- 多尺度鲁棒 point-to-plane ICP 精配准。
- 末级高精度 point-to-surface ICP，并通过覆盖率、可观测性和最大位移门控自动接受或回退。
- 只估计旋转和平移，不进行缩放或非刚性变形。
- 生成基于目标三角面法向的有符号表面偏差图。
- 一个固定 STL 与多个浮动 STL 的顺序批处理。
- 从 Windows 资源管理器拖入 STL 或输出目录。
- 配准前在内存中翻转任一模型的三角面朝向和法线，不修改原文件。
- 固定和浮动模型的自由套索选区，支持透选、Ctrl 取消、反选、有界组件和选区优先配准。
- 非破坏性面片删除与浮动小组件清理，原始 STL 保持不变。
- 质量门控失败后仍可查看明确标记的最佳候选位姿。
- Geomagic 风格连续色阶、左侧透明图例和局部偏差标注。
- 扫描或导入既往结果，在不重新配准的情况下打开三维结果。
- `align_MMDDHHmm` 自包含结果目录、逐模型日志和失败记录。
- 输出机器可读的 JSON 质量指标和警告。

## 下载 Windows 版本

前往 [Releases](https://github.com/Henry0222/general_model_registration/releases) 下载：

```text
GeneralModelRegistration-v1.4.1-win64.zip
GeneralModelRegistration-v1.4.1-win64.zip.sha256.txt
```

解压完整文件夹后运行 `GeneralModelRegistration-v1.4.1.exe`。发布包应包含 `LICENSE`、`THIRD_PARTY_NOTICES.md` 和使用说明；请不要只转发 EXE。

## 从源码运行

推荐使用 64 位 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m auto_alignment
```

也可以在 Windows 上依次运行：

```text
install_windows.bat
run_app.bat
```

## 使用方法

1. 选择固定/参考 STL，它在整批配准过程中保持固定。
2. 选择浮动模型数量，并按顺序选择或拖入各个 STL。
3. 可点击每个地址后的“3D / 选区”查看模型、划定配准优先区或编辑工作副本。
4. 仅在确有需要时勾选对应模型的“翻转面朝向/法线”。
5. 对具有多个相似粗姿态的类骰子、法兰等模型，可启用“彻底检查可能朝向”；角度步长越大，绕轴搜索越快但越粗。普通模型保持关闭。
6. 选择结果根目录和偏差参数，点击“开始顺序配准”。
7. 单个浮动模型失败不会阻止后续模型；可从日志查看失败原因。
8. 在结果列表中查看三维结果、日志或批次目录。
9. 使用“查看既往配准记录”扫描 `align_*`、导入旧版 `results.json`，或手动查看固定 STL 与已配准 STL。

两份模型必须包含足够多的共同稳定表面。低重叠、重复几何、严重缺损、单位错误或网格质量问题都可能使刚性配准不唯一或不可靠。

## 输出文件

- `fixed_target_used.stl`：实际参与该批次配准和历史查看的固定模型。
- `fixed_target_edit_state.json`：固定模型使用的选区和工作副本编辑状态（存在编辑时生成）。
- `batch_results.json` / `batch.log`：整批结果清单和人类可读日志。
- `aligned_current.stl`：已经变换到固定坐标系的浮动模型。
- `comparison_colormap.ply`：保存逐顶点颜色的偏差图。
- `transform.json`：4×4 刚性变换矩阵。
- `results.json`：配准指标、距离统计、门控决策、耗时和警告。
- `moving_model_edit_state.json`：浮动模型使用的选区和工作副本编辑状态（存在编辑时生成）。
- `registration.log`：输入摘要、法线处理、参数、变换、误差和门控说明。
- `failure.json`：失败任务的结构化错误记录。

质量门控失败但仍得到有限候选位姿时，会改用 `best_candidate_*_FAILED_PREVIEW_ONLY` 文件名输出仅供检查的 STL、PLY 和变换，并在 `results.json` 中标记 `review_only: true`；这不代表配准通过。

有符号距离以固定 STL 的最近三角面法向为唯一基准。固定模型被翻转时，偏差正负方向也会随之改变。彩虹图表示配准后浮动 STL 相对固定 STL 的表面偏差。


## 隐私与安全

请勿向 Issue、Pull Request 或仓库提交真实患者、客户或其他个人的 STL、PLY、DPLAN、截图、文件名或结果数据。提交问题时请使用 `scripts/create_demo_data.py` 生成的合成数据复现。
本地日志会记录输入文件名和路径，分享结果目录前请检查并移除可能识别个人身份的信息。

## 免责声明

本项目提供几何配准与可视化结果，不构成医学诊断、治疗建议、质量保证或测量准确度承诺。任何高风险用途都必须由具备资质的人员使用经过验证的流程独立复核。

## 许可证

本项目使用 [BSD 3-Clause License](LICENSE)。二进制发布还包含第三方依赖，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
