# 依赖管理说明

## 问题背景

在 2025-11-17 发现了依赖版本问题:
- 使用 `pip install Pillow` 会安装最新版 Pillow 10.x
- Pillow 10.x 移除了 `Image.LINEAR` 属性
- detectron2 依赖 Pillow 9.x,导致兼容性问题

## 锁定的依赖版本

已创建 `requirements_locked.txt`,包含所有依赖的精确版本(94个包)。

### 关键依赖版本:

```
Pillow==9.5.0           # ⚠️ 重要:不能升级到 10.x
PyMuPDF==1.18.19
pdfplumber==0.11.5
opencv-python==4.12.0.88
numpy==1.24.4
torch==1.10.0+cu113
torchvision==0.11.1+cu113
detectron2==0.6+cu113
```

## 使用方法

### 本机环境

```bash
# 激活环境
conda activate layoutlmv2

# 安装锁定版本的依赖
pip install -r requirements_locked.txt
```

### 云服务器环境

```bash
# 1. 上传文件到云服务器
scp requirements_locked.txt 云服务器:/path/to/layoutlmft/

# 2. 在云服务器上安装
conda activate layoutlmv2
pip install -r requirements_locked.txt
```

## 注意事项

1. **不要使用** `pip install Pillow` 或 `pip install -U Pillow`
   - 会升级到不兼容的 Pillow 10.x

2. **如果已经升级了 Pillow 10.x**:
   ```bash
   pip install Pillow==9.5.0
   ```

3. **验证 Pillow 版本**:
   ```bash
   python -c "import PIL; print(PIL.__version__)"
   # 应该输出: 9.5.0
   ```

4. **如果遇到 `Image.LINEAR` 错误**:
   - 说明 Pillow 版本过高
   - 降级到 9.5.0 即可解决

## 环境验证

运行以下命令验证环境是否正确:

```bash
# 验证 Pillow 版本
python -c "import PIL; print('Pillow:', PIL.__version__)"

# 验证 detectron2 能否正常导入
python -c "from detectron2.data.detection_utils import read_image; print('detectron2: OK')"

# 验证 LayoutLMv2 能否正常导入
python -c "from layoutlmft.models.layoutlmv2 import LayoutLMv2ForTokenClassification; print('LayoutLMv2: OK')"
```

## 更新日期

- 创建日期: 2025-11-17
- 环境: layoutlmv2 conda环境
- Python: 3.8
- CUDA: 11.3
