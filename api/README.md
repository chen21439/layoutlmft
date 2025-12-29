# FastAPI Document Structure Analysis Service

FastAPI 服务用于调用文档结构分析模型进行推理。

## 数据目录结构

```
data_dir_base/                    # 配置的 inference.data_dir_base
└── {task_id}/                    # 任务ID
    ├── {document_name}.json      # JSON 数据文件
    └── images/
        └── {document_name}/      # 图像文件夹
            ├── 0.png
            └── 1.png
```

**示例（test 环境）：**
```
/data/LLM_group/ontology/static/upload/
└── task_001/
    ├── tender_doc.json
    └── images/
        └── tender_doc/
            ├── 0.png
            └── 1.png
```

## 配置

配置文件：`examples/comp_hrdoc/configs/{env}.yaml`

```yaml
inference:
  checkpoint_path: "/path/to/checkpoint"
  data_dir_base: "/path/to/upload/base"
```

| 环境 | data_dir_base |
|-----|--------------|
| dev | `/mnt/e/programFile/AIProgram/tender_ontology/static/upload` |
| test | `/data/LLM_group/ontology/static/upload` |

## 启动服务

```bash
cd /root/code/layoutlmft

# 使用 test 环境配置启动（默认端口 9197）
python -m api.app.main --env dev --reload

# 或
ENV=test uvicorn api.app.main:app --host 0.0.0.0 --port 9197
```

## API 端点

### 推理

```bash
curl -X POST http://localhost:9197/predict \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_001", "document_name": "tender_doc"}'
```

### 健康检查

```bash
curl http://localhost:9197/health
curl http://localhost:9197/config
```

## 请求/响应格式

### 请求

```json
{
  "task_id": "task_001",
  "document_name": "tender_doc"
}
```

### 响应

```json
{
  "document_name": "tender_doc",
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

**注意**：模型未加载时返回空数组 `{"data": [], "num_lines": 0, ...}`

## API 文档

- Swagger UI: http://localhost:9197/docs
- ReDoc: http://localhost:9197/redoc
