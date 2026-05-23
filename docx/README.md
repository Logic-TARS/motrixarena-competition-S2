# MotrixArenaS2

这是一个用于整理微信公众平台文章的离线清理项目。仓库中保留了原始微信文章页面、页面依赖资源、清理后的 HTML 页面，以及从正文中提取出的纯文本内容。

当前文章主题为 **Motphys 仿真 3v3 足球赛训练平台使用指南**，内容覆盖地瓜云镜像创建、Conda 环境配置、Sim Manager 启动、Webview 操作和 Decider 决策端启动流程。

## 文件结构

```text
.
├── 微信公众平台.html          # 原始微信公众平台文章 HTML
├── 微信公众平台_files/        # 原始页面保存时带出的样式、脚本、图片等资源
├── clean.html                 # 清理后的文章 HTML，便于本地阅读
├── article.txt                # 从正文提取出的纯文本内容
├── sim_manager_registry.json  # Sim Manager 运行状态示例或缓存文件
└── tools/
    └── clean_wechat_html.py   # 微信文章正文提取与清理脚本
```

## 环境要求

- Python 3.8+
- 不需要安装第三方 Python 依赖，脚本仅使用标准库。

## 使用方法

在项目根目录运行：

```bash
python3 tools/clean_wechat_html.py
```

脚本会读取：

```text
微信公众平台.html
```

并重新生成：

```text
clean.html
article.txt
```

生成逻辑包括：

- 提取原始页面中 `id="js_content"` 的微信文章正文区域。
- 将图片的 `data-src` 转换为普通 `src`，便于本地 HTML 显示。
- 读取文章标题、作者和发布时间，写入清理后的页面头部。
- 提取正文中的文本片段，输出为 `article.txt`。

## 查看结果

清理完成后，可以直接用浏览器打开：

```text
clean.html
```

如果只需要查看正文文本，可以打开：

```text
article.txt
```

## 注意事项

- `tools/clean_wechat_html.py` 当前使用固定文件名，不接收命令行参数。
- 若要处理新的微信文章，请将新的原始页面保存为 `微信公众平台.html`，并保留对应的 `微信公众平台_files/` 资源目录。
- 运行脚本会覆盖已有的 `clean.html` 和 `article.txt`。
- `sim_manager_registry.json` 看起来是 Sim Manager 的运行状态记录，不是清理脚本的输入文件。

## 文章内容摘要

文章主要说明 MotrixArenaS2 仿真足球赛训练平台的上手流程：

1. 加入地瓜云群组并创建 `simsoccer` 镜像。
2. 进入云桌面并安装 Miniconda。
3. 创建仿真环境 `motrixsim0508` 和决策环境 `k1`。
4. 启动 Sim Manager 服务并访问 `http://127.0.0.1:8000/`。
5. 在网页端启动仿真实例、查看 Webview。
6. 通过 Decider 单机器人命令或 `start_team.sh` 批量启动队伍。
