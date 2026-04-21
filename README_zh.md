# VibeVoice 统一 API

一个基于 FastAPI 的服务，将微软三款 VibeVoice 语音模型 —— **ASR-7B**、
**TTS-1.5B**、**Realtime-0.5B** —— 封装在同一套标准的 HTTP + WebSocket
接口之下。

- **ASR** —— 长音频语音识别，带说话人分离（diarization）和时间戳，
  以结构化片段返回（`speaker_id`、`start_time`、`end_time`、`text`）。
- **TTS** —— 多说话人长文本合成（最多支持 4 位说话人）。
- **Realtime** —— 低延迟单说话人流式 TTS（首包音频约 300 ms），同时提供
  同步接口（`POST`）和 WebSocket 流式接口（PCM16 @ 24 kHz）。

模型采用**惰性加载**，首次请求时才会装载，空闲一段时间后自动释放，
因此可以在一块 GPU 上轮流跑这三个模型。

## 目录结构

```
VibeVioice_demo/
├── api/                       FastAPI 服务（你要运行的部分）
│   ├── server.py              应用入口（uvicorn api.server:app）
│   ├── model_manager.py       惰性加载 + LRU 空闲淘汰
│   ├── config.py              由环境变量驱动的配置
│   ├── voices.py              音色预设发现
│   └── routers/
│       ├── asr.py             POST /v1/asr/transcribe
│       ├── tts.py             POST /v1/tts/synthesize（+ /voices）
│       └── realtime.py        POST /v1/tts/realtime + WS /v1/tts/realtime/ws
│
├── VibeVoice/                 Microsoft/VibeVoice 官方代码（可编辑安装源）
│   ├── vibevoice/             包含模型 / 处理器代码的 Python 包
│   │   └── modular/
│   │       └── modeling_vibevoice_inference.py  ← 来自社区 fork 的 vendored 文件
│   └── demo/voices/
│       ├── streaming_model/   Realtime-0.5B 使用的 .pt 音色预设（25 个文件）
│       └── tts_model/         TTS-1.5B 使用的 .wav 音色样本（9 个文件）
│
├── third_party/
│   └── VibeVoiceCommunity/    vibevoice-community/VibeVoice 克隆（vendored 推理文件的出处）
│
├── docs/
│   └── API.md                 接口参考（schema、示例、请求头）
├── test/                      本地测试素材（音频 + PDF）
├── logs/                      运行时输出（建议加入 .gitignore）
├── download_mode.ipynb        下载三个模型 checkpoint 的 notebook
├── start_api.bat              一键启动脚本（自动激活 `pytorchst1`）
└── README.md                  英文说明
```

## 模型

服务启动时要求三个模型文件夹已经存在于本地磁盘。默认根目录为
`E:\lenv\llmmode\localdown\`（可通过 `VIBEVOICE_MODEL_ROOT` 覆盖），
子目录结构：

| 类型     | 路径                      | HuggingFace ID                        |
|----------|---------------------------|---------------------------------------|
| ASR      | `VibeVoice-ASR/`          | `microsoft/VibeVoice-ASR`             |
| TTS      | `VibeVoice-1.5B/`         | `microsoft/VibeVoice-1.5B`            |
| Realtime | `VibeVoice-Realtime-0.5B/`| `microsoft/VibeVoice-Realtime-0.5B`   |

`download_mode.ipynb` 里演示了如何通过 `huggingface_hub` 搭配
`hf-mirror.com` 镜像下载这些权重。

## 安装（只需一次）

需要一张可用的 CUDA GPU，以及 conda 环境 `pytorchst1`。

```cmd
conda activate pytorchst1

:: 1. 核心 Python 依赖
pip install -e VibeVoice              :: 从同级目录以 editable 方式导入 `vibevoice.*`
pip install fastapi "uvicorn[standard]" python-multipart

:: 2. 将 transformers 固定到与 vendored 模型兼容的版本
pip install "transformers==4.51.3"
```

> **为什么必须是 `transformers==4.51.3`？** 更高版本改动了 `Cache`
> 布局，以及模型 `__init__` 的关键字参数（`dtype=` 变成了
> `torch_dtype=`）。上游已经在 `pyproject.toml[streamingtts]` 下将其
> 固定。

如果没装 `flash-attn`，服务会自动回退到 PyTorch SDPA（速度稍慢但结果
正确）。

## 启动

```cmd
start_api.bat
```

脚本会激活 `pytorchst1`、设置一组合理的环境变量默认值，并在 8000 端口
上运行 `uvicorn api.server:app`。可覆盖的环境变量：

| 变量                            | 默认值                               | 作用                                   |
|---------------------------------|--------------------------------------|----------------------------------------|
| `VIBEVOICE_MODEL_ROOT`          | `E:\lenv\llmmode\localdown`          | 三个模型目录的父文件夹。               |
| `VIBEVOICE_DEVICE`              | `cuda`                               | `cuda` 或 `cpu`。                      |
| `VIBEVOICE_IDLE_EVICT_SECONDS`  | `600`                                | 已加载模型被释放前的空闲秒数。         |
| `VIBEVOICE_EVICT_AFTER_REQUEST` | `0`                                  | 每次请求结束立即释放显存（不等空闲超时）。显存吃紧的机器建议设为 `1`。 |
| `ASR_LANGUAGE_MODEL_NAME`       | `Qwen/Qwen2.5-7B`                    | ASR 语言模型所用 HF tokenizer 基座。   |
| `HF_ENDPOINT`                   | `https://hf-mirror.com`              | ASR 处理器下载 Qwen tokenizer 时使用。 |

健康检查：

```bash
curl http://localhost:8000/health
```

交互式 OpenAPI 文档：<http://localhost:8000/docs>

## 接口一览

| Method | 路径                              | 作用                                         |
|-------:|-----------------------------------|----------------------------------------------|
| GET    | `/health`                         | 设备 / 已加载模型 / 显存信息。               |
| GET    | `/v1/voices`                      | Realtime 音色预设（`.pt`）。                 |
| GET    | `/v1/tts/synthesize/voices`       | TTS-1.5B 音色预设（`.wav`）。                |
| POST   | `/v1/asr/transcribe`              | 语音转文字，附带说话人 / 时间戳。            |
| POST   | `/v1/tts/synthesize`              | 多说话人长文本 TTS → WAV。                   |
| POST   | `/v1/tts/realtime`                | 一次性 Realtime TTS → WAV。                  |
| WS     | `/v1/tts/realtime/ws`             | 流式 Realtime TTS → PCM16 + 事件。           |
| POST   | `/v1/admin/evict?kind=…`          | 强制释放 ASR / TTS / Realtime 模型。         |

完整的请求 / 响应 schema 与示例见 **[docs/API.md](docs/API.md)**。

## 快速示例

**ASR（热词、说话人分离、时间戳一次搞定）**

```bash
curl -F audio=@test/@listening_with_alisher.mp3 \
     -F hotwords="Plato,Athens,Socrates" \
     http://localhost:8000/v1/asr/transcribe
```

返回结果中每个 `segments[i]` 都带有 `speaker_id`、`start_time`、
`end_time`、`text`。非语音的静音段以 `speaker_id=null`、
`text="[Silence]"` 的形式返回。

**多说话人 TTS**

```bash
curl -X POST http://localhost:8000/v1/tts/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text":"Speaker 1: Hello.\nSpeaker 2: Hi.","speakers":["en-Carter_man","en-Alice_woman"]}' \
     --output dialog.wav
```

**Realtime 流式合成**

```python
import asyncio, json, wave, websockets

async def run():
    async with websockets.connect("ws://localhost:8000/v1/tts/realtime/ws") as ws:
        await ws.send(json.dumps({"text": "Hello world.", "voice": "en-Carter_man"}))
        pcm = bytearray()
        async for m in ws:
            if isinstance(m, bytes):
                pcm.extend(m)
            elif json.loads(m).get("event") == "end":
                break
    with wave.open("stream.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(bytes(pcm))

asyncio.run(run())
```

## 关于 TTS-1.5B 的说明

微软于 2025-09-05 从上游仓库删除了 TTS 的 `.generate()` 实现。我们从
社区 fork `https://github.com/vibevoice-community/VibeVoice` 中 vendored
了单个文件 `vibevoice/modular/modeling_vibevoice_inference.py` 到本仓库
的 `VibeVoice/vibevoice/modular/`，权重本身并未改动。

ASR 和 Realtime 模型仍然使用 `VibeVoice/vibevoice/` 中的官方上游代码。

## 许可证

- `VibeVoice/` 遵循官方上游的 MIT 许可证。
- `third_party/VibeVoiceCommunity/` 遵循该 fork 自己的许可证。
- `api/`、`docs/`、`start_api.bat` 下的一切按原样（as-is）提供，供内部
  使用。
