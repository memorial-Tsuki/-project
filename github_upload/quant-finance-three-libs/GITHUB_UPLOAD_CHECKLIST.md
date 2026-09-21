# GitHub 上传前检查表

## 已经整理好的内容

- 三个库各自的主程序、基础测试和 VS Code 配置；
- Tushare 沪深 300 数据下载程序和 `.env.example`；
- 三套独立依赖清单；
- Word 和 PDF 学习手册；
- 关键结果图和小型指标表；
- 官方仓库地址与源码阅读入口。

## 上传前必须确认

- [ ] `labs/alphalens/.env` 不在待上传文件中；
- [ ] GitHub 页面中搜索不到自己的 Tushare Token；
- [ ] `.venv-alphalens`、`.venv-backtrader`、`.venv-qlib` 没有上传；
- [ ] `data/tushare_csi300` 中没有 CSV 或 JSON 数据文件；
- [ ] 没有上传 `output/`、`mlruns/` 或数十 MB 的中间表；
- [ ] README 中的运行顺序和项目实际目录一致。

## 推荐上传步骤

1. 在 GitHub 新建一个空仓库，不要自动添加 README 或许可证。
2. 在 VS Code 中打开本目录的 `quant_finance.code-workspace`。
3. 打开“源代码管理”，选择“初始化仓库”。
4. 检查更改列表，确认看不到 `.env`、CSV 行情或虚拟环境。
5. 提交信息可写：`Initial three-factor research project`。
6. 选择“发布分支”并登录 GitHub。

首次上传后，建议直接在 GitHub 网页检查文件列表和 README 显示效果。
