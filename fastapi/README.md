# FastAPI Document Structure Analysis Service

FastAPI 服务用于调用文档结构分析模型进行推理。

## 目录结构

```
fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── schemas.py           # Pydantic 请求/响应模型
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── predict.py       # /predict 路由
│   │   └── health.py        # /health 路由
│   ├── service/
│   │   ├── __init__.py
│   │   ├── model_loader.py  # 模型加载（单例）
│   │   └── infer_service.py # 推理逻辑
│   └── utils/
│       └── __init__.py
├── requirements.txt
└── README.md
```

## 配置

配置文件位于 `examples/comp_hrdoc/configs/`:
- `dev.yaml` - 本地开发环境
- `test.yaml` - 远程服务器环境

配置项（在 `inference` 段）：
```yaml
# 推理配置 (FastAPI 服务使用)
inference:
  checkpoint_path: "/path/to/checkpoint"
  data_dir_base: "/path/to/upload/base"  # 文档上传的基础目录
```

## 数据目录结构

每个文档以文档名为文件夹存放在 `data_dir_base` 下：

```
data_dir_base/                              # 配置的 inference.data_dir_base
└── {document_name}/                        # 文档名（如 tender_doc_001）
    ├── test/
    │   └── {document_name}.json            # JSON 数据文件
    └── images/
        └── {document_name}/                # 图像文件夹
            ├── 0.png
            ├── 1.png
            └── ...
```

**示例（test 环境）：**
```
/data/LLM_group/ontology/static/upload/
└── tender_doc_001/
    ├── test/
    │   └── tender_doc_001.json
    └── images/
        └── tender_doc_001/
            ├── 0.png
            └── 1.png
```

## 安装

```bash
cd fastapi
pip install -r requirements.txt
```

## 启动服务

### 使用配置文件启动（推荐）

```bash
cd /root/code/layoutlmft

# 使用 test 环境配置启动
python -m fastapi.app.main --env test --port 8000

# 或使用 uvicorn
ENV=test uvicorn fastapi.app.main:app --host 0.0.0.0 --port 8000
```

### 开发模式（自动重载）

```bash
python -m fastapi.app.main --env test --reload
```

## API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 端点

### 健康检查

```bash
# 健康检查
curl http://localhost:8000/health

# 就绪检查
curl http://localhost:8000/ready

# 查看当前配置
curl http://localhost:8000/config
```

### 推理

```bash
# POST 请求
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"document_name": "tender_doc_001"}'

# GET 请求（便捷方式）
curl "http://localhost:8000/predict/tender_doc_001"

# 返回合并原始数据的结果
curl -X POST http://localhost:8000/predict/with-original \
  -H "Content-Type: application/json" \
  -d '{"document_name": "tender_doc_001"}'
```

## 请求/响应格式

### 请求

```json
{
  "document_name": "tender_doc_001"
}
```

### 响应 (`/predict`)

```json
{
  "document_name": "tender_doc_001",
  "num_lines": 100,
  "num_pages": 5,
  "results": [
    {
      "line_id": 0,
      "class_label": "Title",
      "class_id": 0,
      "parent_id": -1,
      "relation": "connect",
      "relation_id": 0
    }
  ],
  "inference_time_ms": 150.5
}
```

### 响应 (`/predict/with-original`)

```json
{
  "document_name": "tender_doc_001",
  "num_lines": 100,
  "inference_time_ms": 150.5,
  "data": [
    {
      "line_id": 0,
      "text": "招标文件",
      "box": [100, 50, 500, 80],
      "class": "Title",
      "parent_id": -1,
      "relation": "connect"
    }
  ]
}
```

## 环境配置

### dev 环境 (本地开发)
```yaml
inference:
  checkpoint_path: null  # 需要手动设置或通过 /admin/load 加载
  data_dir_base: "/mnt/e/programFile/AIProgram/tender_ontology/static/upload"
```

### test 环境 (远程服务器)
```yaml
inference:
  checkpoint_path: "/data/LLM_group/layoutlmft/artifact/exp_20251227_161829/joint_tender/checkpoint-375"
  data_dir_base: "/data/LLM_group/ontology/static/upload"
```

## 注意事项

1. **GPU 配置**：通过 `configs/{env}.yaml` 中的 `gpu.cuda_visible_devices` 设置
2. **模型加载**：模型在启动时加载一次，之后复用（单例模式）
3. **配置优先级**：环境变量 > 配置文件
4. **并发处理**：当前实现为同步推理，如需高并发可考虑使用异步或多 worker
