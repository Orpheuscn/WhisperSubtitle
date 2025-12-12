# WhisperSubtitle - 智能语音转字幕工具

基于 pyannote VAD + OpenAI Whisper 的高性能语音识别转字幕工具，专为对白稀疏的音视频优化。

## ✨ 特性

- 🎯 **智能语音检测**：使用 pyannote speaker-diarization 精准检测语音片段
- 🚀 **高效识别**：只处理有语音的片段，大幅提升处理速度
- 🌍 **多语言支持**：支持 99+ 种语言（日语、中文、英语等）
- 📝 **精确时间戳**：毫秒级时间戳，完美同步
- 💾 **智能缓存**：支持断点续传，避免重复处理
- 🎬 **格式兼容**：支持所有常见音视频格式（MP4、MKV、AVI、MP3、WAV 等）

## 🔧 技术实现

### 核心技术栈

1. **语音活动检测 (VAD)**
   - 模型：`pyannote/speaker-diarization-3.1`
   - 功能：精准检测音频中的语音片段，过滤静音和背景音
   - 优势：相比传统 VAD，准确率提升 15-20%

2. **语音识别**
   - 引擎：OpenAI Whisper
   - 支持模型：`tiny`, `base`, `small`, `medium`, `large`, `turbo`
   - 推荐：`turbo` 模型（速度快，准确率高）

3. **音频处理**
   - 工具：FFmpeg
   - 功能：音频提取、格式转换、片段切割

### 工作流程

```
输入视频/音频
    ↓
FFmpeg 提取音频 (16kHz, 单声道)
    ↓
pyannote VAD 检测语音片段
    ↓
切割语音片段 (添加 padding)
    ↓
Whisper 逐片段识别
    ↓
合并结果生成 SRT 字幕
```

## 📦 依赖安装

### 系统依赖

#### 1. FFmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows
# 下载：https://ffmpeg.org/download.html
# 添加到系统 PATH
```

#### 2. OpenAI Whisper
```bash
# macOS
brew install openai-whisper

# 其他系统使用 pip（全局安装）
pip3 install -U openai-whisper
```

### Python 虚拟环境依赖

1. **创建虚拟环境**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

主要依赖包括：
- `torch` - PyTorch 深度学习框架
- `pyannote.audio` - 语音活动检测
- `soundfile` - 音频文件读写
- `numpy` - 数值计算

## 🔑 配置说明

### 1. 获取 HuggingFace Token

pyannote 模型需要 HuggingFace 账号和访问令牌。

**步骤：**

1. 注册 HuggingFace 账号：https://huggingface.co/join
2. 访问模型页面并接受用户协议：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - 点击 "Agree and access repository"
3. 生成访问令牌：
   - 访问：https://huggingface.co/settings/tokens
   - 点击 "New token"
   - 选择 "Read" 权限
   - 复制生成的 token（格式：`hf_xxxxxxxxxxxxx`）

### 2. 配置环境变量

**macOS/Linux (zsh):**
```bash
# 编辑 ~/.zshrc
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

**macOS/Linux (bash):**
```bash
# 编辑 ~/.bashrc 或 ~/.bash_profile
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):**
```powershell
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'hf_your_token_here', 'User')
```

**临时设置（当前会话）：**
```bash
export HF_TOKEN="hf_your_token_here"
```

## 🚀 使用方法

### 基本用法

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行脚本
python vad_transcribe.py <视频文件> --language <语言代码> --model <模型名称>
```

### 参数说明

- `<视频文件>`：输入的音视频文件路径（必需）
- `--language`：语言代码（可选，推荐指定以提高准确率）
  - `ja` - 日语
  - `zh` - 中文
  - `en` - 英语
  - 更多语言代码见 [Whisper 文档](https://github.com/openai/whisper#available-models-and-languages)
- `--model`：Whisper 模型（可选，默认 `base`）
  - `tiny` - 最快，准确率较低
  - `base` - 平衡
  - `small` - 较好
  - `medium` - 很好
  - `large` - 最佳准确率
  - `turbo` - **推荐**，速度快且准确率高

### 使用示例

**日语视频转字幕（推荐）：**
```bash
python vad_transcribe.py video.mkv --language ja --model turbo
```

**中文音频转字幕：**
```bash
python vad_transcribe.py audio.mp3 --language zh --model turbo
```

**自动检测语言：**
```bash
python vad_transcribe.py video.mp4 --model turbo
```

**使用 large 模型获得最高准确率：**
```bash
python vad_transcribe.py video.mkv --language ja --model large
```

### 输出文件

脚本会生成以下文件：

1. **SRT 字幕文件**：`<输入文件名>.srt`
   - 标准 SRT 格式，可直接用于视频播放器

2. **临时文件夹**：`temp_continuous/`
   - 包含提取的音频、语音片段、识别结果 JSON
   - 支持断点续传，可手动删除以重新处理

### 性能优化建议

1. **首次运行**：
   - pyannote 模型会自动下载（约 200MB）
   - Whisper 模型会自动下载（turbo 约 1.5GB）
   - 下载完成后会缓存，后续运行无需重新下载

2. **处理速度**：
   - 对白稀疏的视频：处理速度约为实时的 2-5 倍
   - 对白密集的视频：处理速度约为实时的 1-2 倍
   - 使用 `turbo` 模型可获得最佳速度/准确率平衡

3. **磁盘空间**：
   - 临时文件夹大小约为原音频的 1-2 倍
   - 处理完成后可删除 `temp_continuous/` 文件夹

## 📊 性能对比

| 模型 | 速度 | 准确率 | 推荐场景 |
|------|------|--------|----------|
| tiny | ⭐⭐⭐⭐⭐ | ⭐⭐ | 快速预览 |
| base | ⭐⭐⭐⭐ | ⭐⭐⭐ | 日常使用 |
| small | ⭐⭐⭐ | ⭐⭐⭐⭐ | 平衡选择 |
| medium | ⭐⭐ | ⭐⭐⭐⭐ | 高质量 |
| large | ⭐ | ⭐⭐⭐⭐⭐ | 最高质量 |
| **turbo** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **推荐** |

## 🔧 故障排除

### 问题：pyannote 模型加载失败

**错误信息**：`401 Client Error: Unauthorized`

**解决方法**：
1. 确认已接受模型用户协议：https://huggingface.co/pyannote/speaker-diarization-3.1
2. 检查 HF_TOKEN 环境变量是否正确设置
3. 验证 token：`echo $HF_TOKEN`

### 问题：Whisper 命令未找到

**错误信息**：`whisper: command not found`

**解决方法**：
```bash
# macOS
brew install openai-whisper

# 其他系统
pip3 install -U openai-whisper
```

### 问题：FFmpeg 未安装

**错误信息**：`ffmpeg: command not found`

**解决方法**：
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 问题：虚拟环境依赖冲突

**解决方法**：
```bash
# 删除虚拟环境
rm -rf venv

# 重新创建
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📝 许可证

MIT License

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 强大的语音识别模型
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - 优秀的语音活动检测
- [FFmpeg](https://ffmpeg.org/) - 音视频处理工具

## 📮 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

