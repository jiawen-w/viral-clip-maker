# Viral Clip Maker

自动将长视频剪辑成爆款短视频。输入一个 YouTube 链接，输出多个适配抖音/视频号/Reels 的竖屏短视频。

## 流程

```
YouTube 链接
    ↓
① yt-dlp 下载视频（已下载自动跳过）
    ↓
② Whisper 语音转文字（带时间戳，结果自动缓存）
    ↓
③ PySceneDetect 镜头边界检测
    ↓
④ 合并转录段 → 生成候选片段
    ↓
⑤ AI 打分（情绪张力 / 信息密度 / 钩子 / 传播力）
    ↓
⑥ OpenCV 人脸居中裁剪（竖屏 9:16）
    ↓
⑦ ffmpeg 剪辑 + ASS 字幕烧录
    ↓
输出短视频 short_01_score9.mp4 ...
```

## 环境要求

- Python 3.9+
- ffmpeg（需含 libass，用于字幕烧录）

```bash
# macOS — 使用含完整编解码器的版本
brew tap homebrew-ffmpeg/ffmpeg
brew install homebrew-ffmpeg/ffmpeg/ffmpeg

# Ubuntu
apt install ffmpeg
```

> 注意：`brew install ffmpeg` 安装的精简版不含 libass，会导致字幕烧录失败，请使用上面的完整版。

## 安装

```bash
git clone https://github.com/jiawen-w/viral-clip-maker.git
cd viral-clip-maker
pip install yt-dlp openai-whisper scenedetect anthropic opencv-python
```

## 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 Key（.env 已加入 .gitignore，不会上传）
```

**使用 Claude 官方 API：**
```
ANTHROPIC_API_KEY=your_key_here
```

**使用豆包 / 火山引擎 Coding Plan：**
```
ANTHROPIC_BASE_URL=https://ark.cn-beijing.volces.com/api/coding
ANTHROPIC_AUTH_TOKEN=your_token_here
ANTHROPIC_MODEL=doubao-seed-2.0-pro
```

## 使用

```bash
# 直接运行，按提示输入 YouTube 链接（再次运行可回车复用上次）
python viral_clip_maker.py

# 或直接传入链接
python viral_clip_maker.py "https://www.youtube.com/watch?v=xxxxx"
```

**完整参数：**

```bash
python viral_clip_maker.py "链接" \
  -o ./my_shorts \          # 输出目录（默认 ./output_shorts）
  -n 3 \                    # 输出几个短视频（默认 5）
  -w medium \               # Whisper 模型：tiny/base/small/medium/large
  --min-dur 20 \            # 片段最短秒数（默认 30）
  --max-dur 60 \            # 片段最长秒数（默认 90）
  --scene-threshold 25      # 场景检测灵敏度，越小越灵敏（默认 30）
```

## 缓存机制

重复运行同一视频时，以下步骤自动跳过：

| 步骤 | 缓存位置 |
|------|---------|
| 视频下载 | `output_shorts/_work/session.json` |
| Whisper 转录 | `视频名.mp4.whisper.json` |

第二次运行直接从 Step 3 场景检测开始，节省大量时间。

## Whisper 模型选择

| 模型 | 速度 | 中文 | 英文 | 显存 |
|------|------|------|------|------|
| tiny | 最快 | 差 | 一般 | ~1GB |
| base | 快 | 一般 | 还行 | ~1GB |
| small | 中 | 还行 | 良好 | ~2GB |
| **medium** | 慢 | **良好** | 很好 | ~5GB |
| large | 最慢 | 最好 | 最好 | ~10GB |

中文视频推荐 `medium` 及以上。

## AI 打分维度

每个候选片段由 AI 从以下 5 个维度综合评分（满分 10 分）：

1. **情绪张力** — 笑点、冲突、反转、惊喜、感动
2. **信息密度** — 干货密集，结论清晰
3. **开头钩子** — 前几秒是否能抓住注意力
4. **语义完整性** — 内容自成一段，有头有尾
5. **传播潜力** — 看完是否有分享欲

## 输出说明

```
output_shorts/
├── short_01_score9.mp4   # 得分最高
├── short_02_score8.mp4
├── short_03_score8.mp4
├── short_04_score7.mp4
├── short_05_score7.mp4
└── _work/                # 中间文件（原视频、字幕、缓存）
```

所有短视频均为 1080×1920 竖屏，已烧录字幕，可直接发布。
