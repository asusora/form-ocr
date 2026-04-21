# Form OCR M1

## 1. 项目说明

本项目根据 `docs/specs` 中的实施规格，先落地 `M1` 基础识别闭环，只覆盖以下能力：

- 文件上传与任务创建
- PDF/图片渲染与预处理
- 锚点 OCR 适配层
- 模板路由决策
- 左通道字段检测
- 统一 JSON 结果导出
- 大模型 `API Key` 配置预留

当前版本**不执行**右通道 VLM、融合、Key Mapping、前端审阅和模板学习。

## 2. 目录说明

- `app/main.py`：FastAPI 启动入口
- `app/services/preprocess_service.py`：阶段 0 预处理
- `app/services/router_service.py`：模板路由
- `app/services/left_channel_service.py`：左通道字段检测
- `app/services/export_service.py`：阶段 4 结构化导出
- `config/template_registry.json`：模板注册表
- `artifacts/`：运行后生成的任务工件目录

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

如需使用 PaddleOCR，可额外安装：

```bash
pip install paddleocr
```

然后在 `.env` 中设置：

```env
FORM_OCR_OCR_ENGINE=paddleocr
```

## 4. 配置说明

从 `.env.example` 复制为 `.env`，至少检查以下配置：

```env
FORM_OCR_OCR_ENGINE=rapidocr
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

说明：

- `OPENAI_API_KEY` 与 `ANTHROPIC_API_KEY` 当前只做占位，不在 `M1` 实际调用。
- `FORM_OCR_TEMPLATES_PATH` 指向模板注册表 JSON。
- `FORM_OCR_DATA_DIR` 控制本地工件落盘目录。

## 5. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 6. 前端联调

项目已新增 `frontend/` 目录，使用 `Vue 3 + Vite` 编写，用于联调当前 `M1` 接口。

进入前端目录后执行：

```bash
cd frontend
npm install
npm run dev
```

默认开发地址为：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`

后端已预置本地开发跨域白名单：

```env
FORM_OCR_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
```

前端会通过 Vite 代理访问以下路径：

- `/api`
- `/artifacts`

## 7. 主要接口

### 6.1 健康检查

```bash
curl http://127.0.0.1:8000/api/v1/health
```

### 6.2 查看大模型配置占位状态

```bash
curl http://127.0.0.1:8000/api/v1/llm-config
```

### 6.3 上传任务

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/tasks" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@sample_form.png"
```

### 6.4 执行 M1

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/tasks/{task_id}/run-m1"
```

### 6.5 查询导出结果

```bash
curl "http://127.0.0.1:8000/api/v1/tasks/{task_id}/results/export"
```

### 6.6 查询阶段调试结果

```bash
curl "http://127.0.0.1:8000/api/v1/tasks/{task_id}/debug/stage0_output"
curl "http://127.0.0.1:8000/api/v1/tasks/{task_id}/debug/route_decision"
curl "http://127.0.0.1:8000/api/v1/tasks/{task_id}/debug/left_channel_output"
curl "http://127.0.0.1:8000/api/v1/tasks/{task_id}/debug/export_output"
```

## 8. 模板注册表示例

```json
{
  "templates": [
    {
      "template_id": "tpl_demo",
      "template_revision": 1,
      "form_title": "Demo Form",
      "page_count": 1,
      "anchors": [
        {
          "page_index": 0,
          "text": "Name",
          "bbox": {
            "x": 0.1,
            "y": 0.1,
            "w": 0.08,
            "h": 0.02
          }
        }
      ],
      "fields": [
        {
          "page_index": 0,
          "canonical_key_id": "ck_name",
          "field_type": "text",
          "bbox_relative": {
            "x": 0.25,
            "y": 0.1,
            "w": 0.25,
            "h": 0.03
          },
          "key": "姓名",
          "key_en": "Name"
        }
      ]
    }
  ]
}
```
