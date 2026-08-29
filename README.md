# General Model Registration

General Model Registration 是一个面向 Windows 的纯 CPU 三角网格刚性配准工具。它读取两份 STL，将待配准模型移动到目标模型坐标系，并输出刚性变换、配准后的 STL、带颜色的 PLY 偏差图和质量报告。

当前版本：**v1.3.0**

## 主要功能

- FPFH 特征与多候选全局粗配准。
- 多尺度鲁棒 point-to-plane ICP 精配准。
- 末级高精度 point-to-surface ICP，并通过覆盖率、可观测性和最大位移门控自动接受或回退。
- 只估计旋转和平移，不进行缩放或非刚性变形。
- 生成基于目标三角面法向的有符号表面偏差图。
- 支持包含中文字符的 Windows 文件路径。
- 输出机器可读的 JSON 质量指标和警告。

## 下载 Windows 版本

前往 [Releases](https://github.com/Henry0222/general_model_registration/releases) 下载：

```text
GeneralModelRegistration-v1.3.0-win64.zip
GeneralModelRegistration-v1.3.0-win64.zip.sha256.txt
```

解压完整文件夹后运行 `GeneralModelRegistration-v1.3.0.exe`。发布包应包含 `LICENSE`、`THIRD_PARTY_NOTICES.md` 和使用说明；请不要只转发 EXE。

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

1. 选择目标/参考 STL，它在配准过程中保持固定。
2. 选择待配准 STL，它将被移动到目标坐标系。
3. 选择输出目录和偏差色标上限。
4. 点击“一键配准并生成方向偏差图”。
5. 检查配准可信度、警告和三维偏差图。

两份模型必须包含足够多的共同稳定表面。低重叠、重复几何、严重缺损、单位错误或网格质量问题都可能使刚性配准不唯一或不可靠。

## 输出文件

- `aligned_current.stl`：已经变换到目标坐标系的待配准模型。
- `comparison_colormap.ply`：保存逐顶点颜色的偏差图。
- `transform.json`：4×4 刚性变换矩阵。
- `results.json`：配准指标、距离统计、门控决策、耗时和警告。

有符号距离以目标 STL 的最近三角面法向为基准。不同软件导出的法向方向可能相反，使用新数据来源时应通过已知区域验证颜色方向。

## 测试

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

生成不含真实个人或患者信息的合成演示数据：

```powershell
python scripts\create_demo_data.py
```

## 构建 Windows 发布包

在安装 64 位 Python 3.12 后运行：

```text
install_windows.bat
build_windows.bat
```

构建完成后，ZIP 和 SHA-256 文件位于 `dist/`。详细发布步骤见 [RELEASING.md](RELEASING.md)。

## 隐私与安全

请勿向 Issue、Pull Request 或仓库提交真实患者、客户或其他个人的 STL、PLY、DPLAN、截图、文件名或结果数据。提交问题时请使用 `scripts/create_demo_data.py` 生成的合成数据复现。

## 免责声明

本项目提供几何配准与可视化结果，不构成医学诊断、治疗建议、质量保证或测量准确度承诺。任何高风险用途都必须由具备资质的人员使用经过验证的流程独立复核。

## 许可证

本项目使用 [BSD 3-Clause License](LICENSE)。二进制发布还包含第三方依赖，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
