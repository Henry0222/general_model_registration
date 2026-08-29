# Release process

1. 在干净的 Python 3.12 环境中安装开发依赖。
2. 运行完整测试：`python -m pytest`。
3. 运行 `build_windows.bat`。
4. 在另一台 Windows 10/11 64 位电脑上解压 ZIP 并完成冒烟测试。
5. 检查 ZIP 内含 EXE、`LICENSE`、`THIRD_PARTY_NOTICES.md`、`使用说明.txt`。
6. 核对 ZIP 的 SHA-256。
7. 提交源码，创建并推送标签 `v1.3.0`。
8. 在 GitHub 创建 `v1.3.0` Release，上传 ZIP 和 `.sha256.txt`，不要把二进制提交到 Git 历史。

发布前还应检查：

- `git status --short` 没有意外文件。
- `git ls-files` 中没有 STL、PLY、DPLAN、EXE、ZIP、PYC、PYD 或真实数据。
- README 中的版本、文件名和 SHA-256 文件名与实际产物一致。
- 第三方许可证和声明与实际打包依赖一致。
