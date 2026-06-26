# VideoKit Colab 使用指南

## 快速开始

### 方法一：使用Colab笔记本（推荐）

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 上传 `videokit_colab.ipynb` 文件
3. 选择 GPU 运行时：
   - 点击顶部菜单 `Runtime` -> `Change runtime type`
   - `Hardware accelerator` 选择 `GPU`
   - 点击 `Save`
4. 按顺序运行每个单元格
5. 等待Gradio启动，点击公网链接进入Web界面

### 方法二：快速安装脚本

在Colab单元格中运行：

```bash
!bash /path/to/colab_setup.sh
```

然后上传项目文件并启动。

## 项目文件上传方式

### 方式1：直接上传（适合小项目）
1. 点击左侧文件图标
2. 将 `videokit/` 文件夹和 `videokit.py` 拖拽上传
3. 上传到 `/content/videokit_workspace/` 目录

### 方式2：Google Drive（推荐大文件）
```python
from google.colab import drive
drive.mount('/content/drive')

# 复制文件
!cp -r /content/drive/MyDrive/videokit /content/videokit_workspace/
!cp /content/drive/MyDrive/videokit.py /content/videokit_workspace/
```

### 方式3：GitHub克隆
```bash
!git clone https://github.com/你的用户名/你的仓库.git /content/videokit_workspace/
```

## 命令行使用示例

### 启动Web UI
```bash
python videokit.py run --execution-providers cuda
```

### 无界面模式（批处理）
```bash
python videokit.py headless-run \
  --source-paths source.jpg \
  --target-path target.mp4 \
  --output-path output.mp4 \
  --processors style_transfer \
  --execution-providers cuda
```

### 预下载所有模型
```bash
python videokit.py force-download --download-scope all
```

## 性能优化建议

### 显存优化
- 使用 `--video-memory-strategy strict` 最严格的显存策略
- 降低输出分辨率：`--output-video-scale 0.5`
- 减少处理线程：`--execution-thread-count 4`

### 速度优化
- 使用CUDA执行提供者：`--execution-providers cuda`
- 增加线程数：`--execution-thread-count 16`
- 使用更大的GPU（Colab Pro有更好的GPU）

## 免费Colab限制

| 项目 | 免费版 | Pro版 |
|------|--------|-------|
| GPU | T4 | T4/P100/V100 |
| GPU显存 | 16GB | 16-32GB |
| 连续使用 | 约12小时 | 约24小时 |
| 空闲超时 | 90分钟 | 90分钟 |
| 内存 | 12GB | 32GB |

## 常见问题

### Q: 启动后找不到公网链接？
A: 查看输出日志中是否有 `Running on public URL: https://xxxx.gradio.live` 的链接

### Q: 模型下载很慢？
A: 尝试切换下载源：
```bash
python videokit.py run --download-providers huggingface
```

### Q: CUDA out of memory？
A: 
1. 使用 `--video-memory-strategy strict`
2. 降低处理分辨率
3. 减少同时使用的处理器数量

### Q: 视频处理很慢？
A: 
1. 确保使用GPU版本的onnxruntime
2. 检查执行提供者是否为CUDA
3. 适当增加线程数

### Q: Colab断开连接？
A: 
- 免费版Colab有空闲超时
- 可以使用Colab Pro获得更长时间
- 重要文件记得保存到Google Drive

## 文件保存

处理完成后，将输出文件保存到Google Drive：

```python
from google.colab import drive
drive.mount('/content/drive')

# 复制输出文件
!cp /content/videokit_workspace/output.mp4 /content/drive/MyDrive/
```

## 故障排除

### 检查GPU是否可用
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### 检查ONNX Runtime
```python
import onnxruntime as ort
print(ort.get_available_providers())
```

### 检查ffmpeg
```bash
!ffmpeg -version
```
