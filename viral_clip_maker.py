#!/usr/bin/env python3
"""
爆款短视频自动剪辑工具
流程：YouTube 下载 → Whisper 转录 → PySceneDetect 场景边界 →
      Claude AI 打分 → 竖屏人脸居中裁剪 → 字幕烧录 → 输出短视频
"""

import os
import sys
import json
import subprocess
import argparse
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 自动加载同目录下的 .env 文件
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip().strip('"').strip("'")

_load_dotenv()

# ============================================================
# 依赖检查
# ============================================================
try:
    import yt_dlp
except ImportError:
    sys.exit("请安装: pip install yt-dlp")

try:
    import whisper
except ImportError:
    sys.exit("请安装: pip install openai-whisper")

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_NEW_API = True
except ImportError:
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
        SCENEDETECT_NEW_API = False
    except ImportError:
        sys.exit("请安装: pip install scenedetect")

try:
    import anthropic
except ImportError:
    sys.exit("请安装: pip install anthropic")


# ============================================================
# 默认参数
# ============================================================
DEFAULT_OUTPUT_DIR = "./output_shorts"
MAX_CLIPS          = 5
CLIP_MIN_DURATION  = 30.0
CLIP_MAX_DURATION  = 90.0
WHISPER_MODEL      = "medium"
CLAUDE_MODEL       = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


@dataclass
class Segment:
    start: float
    end: float
    text: str
    score: float = 0.0
    score_reason: str = ""


# ============================================================
# Step 1: 下载 YouTube 视频
# ============================================================
def _session_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, "_work", "session.json")


def _load_session(output_dir: str) -> dict:
    path = _session_cache_path(output_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_session(output_dir: str, data: dict) -> None:
    path = _session_cache_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def download_youtube(url: str, output_dir: str, session: dict) -> str:
    # 同一个链接且文件还在，直接复用
    if session.get("url") == url and session.get("video_path"):
        cached = session["video_path"]
        if os.path.exists(cached):
            print(f"[Step 1] 使用已下载的视频: {os.path.basename(cached)}")
            return cached

    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "merge_output_format": "mp4",
        "noplaylist": True,
    }

    print(f"[Step 1] 下载视频: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if not filename.endswith(".mp4"):
            filename = os.path.splitext(filename)[0] + ".mp4"

    if not os.path.exists(filename):
        candidates = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
        if not candidates:
            sys.exit("下载失败：找不到 mp4 文件")
        filename = os.path.join(output_dir, candidates[0])

    print(f"  → {filename}")
    return filename


# ============================================================
# Step 2: Whisper 转录（带时间戳，自动缓存）
# ============================================================
def transcribe(video_path: str, model_size: str = WHISPER_MODEL) -> List[dict]:
    cache_path = video_path + ".whisper.json"

    if os.path.exists(cache_path):
        print(f"[Step 2] 读取转录缓存: {os.path.basename(cache_path)}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  → 共 {len(data['segments'])} 段，语言: {data.get('language', '未知')}（已缓存）")
        return data["segments"]

    print(f"[Step 2] Whisper 转录（模型: {model_size}）...")
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path, verbose=False)
    segs = result["segments"]
    lang = result.get("language", "未知")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"language": lang, "segments": segs}, f, ensure_ascii=False, indent=2)
    print(f"  → 共 {len(segs)} 段，检测语言: {lang}（已缓存到 {os.path.basename(cache_path)}）")
    return segs


# ============================================================
# Step 3: PySceneDetect 场景边界
# ============================================================
def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Tuple[float, float]]:
    print("[Step 3] 场景边界检测...")
    if SCENEDETECT_NEW_API:
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        sm.detect_scenes(video)
        scene_list = sm.get_scene_list()
    else:
        vm = VideoManager([video_path])
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        vm.set_downscale_factor()
        vm.start()
        sm.detect_scenes(frame_source=vm)
        scene_list = sm.get_scene_list()
        vm.release()

    scenes = [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
    print(f"  → 共检测到 {len(scenes)} 个场景")
    return scenes


# ============================================================
# Step 4: 对齐转录段 + 场景边界，生成候选片段
# ============================================================
def build_candidate_segments(
    whisper_segs: List[dict],
    scenes: List[Tuple[float, float]],
    min_dur: float,
    max_dur: float,
) -> List[Segment]:
    print("[Step 4] 构建候选片段...")
    scene_ends = {s[1] for s in scenes}

    candidates: List[Segment] = []
    buf_texts: List[str] = []
    buf_start: Optional[float] = None

    for ws in whisper_segs:
        t_start = ws["start"]
        t_end = ws["end"]
        text = ws["text"].strip()

        if buf_start is None:
            buf_start = t_start

        buf_texts.append(text)
        buf_dur = t_end - buf_start

        near_scene_end = any(abs(t_end - se) < 1.5 for se in scene_ends)
        should_cut = (near_scene_end and buf_dur >= min_dur) or buf_dur >= max_dur

        if should_cut:
            candidates.append(Segment(
                start=buf_start,
                end=t_end,
                text=" ".join(buf_texts),
            ))
            buf_texts = []
            buf_start = None

    if buf_texts and buf_start is not None:
        buf_dur = whisper_segs[-1]["end"] - buf_start
        if buf_dur >= min_dur:
            candidates.append(Segment(
                start=buf_start,
                end=whisper_segs[-1]["end"],
                text=" ".join(buf_texts),
            ))

    print(f"  → 共 {len(candidates)} 个候选片段")
    return candidates


# ============================================================
# Step 5: Claude 打分
# ============================================================
SCORE_PROMPT = """\
你是专业的短视频内容策划，擅长判断哪些片段最容易在抖音/视频号/Reels 上爆款。

对下列视频片段的转录文字打分（满分 10 分），评分维度：
1. 情绪张力（笑点/冲突/反转/惊喜/感动）
2. 信息密度（干货密集，结论清晰）
3. 开头钩子（前几秒能否抓住注意力）
4. 语义完整性（内容自成一段，有头有尾）
5. 传播潜力（看完是否想分享）

只返回 JSON 数组，格式：
[{{"index": 0, "score": 8.5, "reason": "一句话原因"}}, ...]

片段列表：
{segments_json}
"""


def _call_api(prompt: str, model: str, api_key: Optional[str]) -> str:
    """统一 API 调用：自定义 base_url 时直接用 httpx 发 Anthropic 格式请求，避免 SDK URL 拼接问题。"""
    import httpx

    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
    token = api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")

    if base_url:
        # 直接拼接，避免 Anthropic SDK 内部 URL 路径被截断
        url = f"{base_url}/v1/messages"
        headers = {
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = httpx.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    else:
        # 原生 Anthropic SDK（官方 API Key）
        client = anthropic.Anthropic(api_key=token)
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


def score_with_claude(
    segments: List[Segment],
    api_key: Optional[str] = None,
    model: str = CLAUDE_MODEL,
) -> List[Segment]:
    payload = [
        {
            "index": i,
            "duration_s": round(seg.end - seg.start, 1),
            "text": seg.text[:600],
        }
        for i, seg in enumerate(segments)
    ]

    print(f"[Step 5] AI 打分（{len(segments)} 个片段）...")
    prompt = SCORE_PROMPT.format(
        segments_json=json.dumps(payload, ensure_ascii=False, indent=2)
    )
    raw = _call_api(prompt, model, api_key)
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Claude 返回格式异常:\n{raw[:300]}")

    scores = json.loads(match.group())
    score_map = {item["index"]: item for item in scores}

    for i, seg in enumerate(segments):
        if i in score_map:
            seg.score = float(score_map[i].get("score", 0))
            seg.score_reason = score_map[i].get("reason", "")

    ranked = sorted(segments, key=lambda x: x.score, reverse=True)

    print("  → Top 片段:")
    for seg in ranked[:MAX_CLIPS]:
        dur = seg.end - seg.start
        print(f"    [{seg.score:.1f}分] {seg.start:.1f}s-{seg.end:.1f}s ({dur:.0f}s) | {seg.score_reason}")

    return ranked




# ============================================================
# Step 7: 生成 ASS 字幕（样式内嵌，避免 ffmpeg force_style 引号转义问题）
# ============================================================
ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,100,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""


def _ass_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}:{m:02d}:{sec:05.2f}"


def write_ass(whisper_segs: List[dict], seg: Segment, ass_path: str) -> None:
    lines = [ASS_HEADER]
    for ws in whisper_segs:
        if ws["end"] <= seg.start or ws["start"] >= seg.end:
            continue
        s = max(ws["start"], seg.start) - seg.start
        e = min(ws["end"], seg.end) - seg.start
        text = ws["text"].strip().replace("\n", "\\N")
        if not text or e <= s:
            continue
        lines.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Default,,0,0,0,,{text}")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Step 8: ffmpeg 剪辑 + 字幕烧录
# ============================================================
def _find_chinese_font() -> Optional[str]:
    for path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(path):
            return path
    return None


def _has_subtitles_filter() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
        return " subtitles " in r.stdout
    except Exception:
        return False


def build_drawtext_chain(whisper_segs: List[dict], seg: Segment, work_dir: str, clip_idx: int) -> str:
    """ffmpeg 没有 libass 时，用 drawtext 逐行绘制字幕。文本写到 txt 文件避免转义。"""
    font = _find_chinese_font()
    if not font:
        return ""

    filters = []
    txt_idx = 0
    for ws in whisper_segs:
        if ws["end"] <= seg.start or ws["start"] >= seg.end:
            continue
        s = max(ws["start"], seg.start) - seg.start
        e = min(ws["end"], seg.end) - seg.start
        text = ws["text"].strip()
        if not text or e <= s:
            continue
        # 每行字幕写一个 txt 文件，drawtext 用 textfile 读取（避免特殊字符转义）
        txt_path = f"/tmp/viral_sub_{clip_idx:02d}_{txt_idx:03d}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        txt_idx += 1

        filters.append(
            f"drawtext=fontfile='{font}'"
            f":textfile={txt_path}"
            f":fontsize=56"
            f":fontcolor=white"
            f":borderw=4"
            f":bordercolor=black"
            f":x=(w-text_w)/2"
            f":y=h-220"
            f":enable='between(t\\,{s:.3f}\\,{e:.3f})'"
        )
    return ",".join(filters)


def export_clip(
    video_path: str,
    seg: Segment,
    out_path: str,
    whisper_segs: List[dict],
    clip_idx: int,
    work_dir: str,
) -> None:
    duration = seg.end - seg.start

    # 优先使用 subtitles 滤镜（需 libass），否则降级用 drawtext
    if _has_subtitles_filter():
        import shutil
        ass_path = os.path.join(work_dir, f"clip_{clip_idx:02d}.ass")
        write_ass(whisper_segs, seg, ass_path)
        tmp_ass = f"/tmp/viral_clip_{clip_idx:02d}.ass"
        shutil.copy2(ass_path, tmp_ass)
        vf = f"subtitles={tmp_ass}"
    else:
        sub_filter = build_drawtext_chain(whisper_segs, seg, work_dir, clip_idx)
        vf = sub_filter if sub_filter else None

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seg.start),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd.append(out_path)

    print(f"  [剪辑 {clip_idx+1}] {seg.start:.1f}s → {seg.end:.1f}s ({duration:.0f}s) 分数: {seg.score:.1f}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [ffmpeg 错误]\n{result.stderr[-800:]}")
    else:
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"    → {out_path} ({size_mb:.1f} MB)")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="YouTube 爆款短视频自动剪辑工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("url", nargs="?", help="YouTube 视频链接（不填则运行时输入）")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("-n", "--num-clips", type=int, default=MAX_CLIPS, help="最多输出几个短视频")
    parser.add_argument(
        "-w", "--whisper-model",
        default=WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型（越大越准越慢）",
    )
    parser.add_argument("--min-dur", type=float, default=CLIP_MIN_DURATION, help="片段最短秒数")
    parser.add_argument("--max-dur", type=float, default=CLIP_MAX_DURATION, help="片段最长秒数")
    parser.add_argument("--api-key", help="Anthropic API Key（或设置环境变量 ANTHROPIC_API_KEY）")
    parser.add_argument("--scene-threshold", type=float, default=30.0, help="场景检测灵敏度（越小越灵敏）")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    work_dir = os.path.join(args.output, "_work")
    os.makedirs(work_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("  爆款短视频自动剪辑工具")
    print("=" * 50)

    session = _load_session(args.output)

    if not args.url:
        last = session.get("url", "")
        prompt = f"\n请输入 YouTube 视频链接（直接回车复用上次: {last}）: " if last else "\n请输入 YouTube 视频链接: "
        entered = input(prompt).strip()
        args.url = entered or last
    if not args.url:
        sys.exit("未输入链接，退出。")

    # Step 1
    video_path = download_youtube(args.url, work_dir, session)
    _save_session(args.output, {"url": args.url, "video_path": video_path})

    # Step 2
    whisper_segs = transcribe(video_path, args.whisper_model)

    # Step 3
    scenes = detect_scenes(video_path, args.scene_threshold)

    # Step 4
    candidates = build_candidate_segments(
        whisper_segs, scenes, args.min_dur, args.max_dur
    )
    if not candidates:
        sys.exit(
            "\n[错误] 未生成有效候选片段。\n"
            f"提示：当前 --min-dur={args.min_dur}s，可尝试调小该值"
        )

    # Step 5
    ranked = score_with_claude(candidates, args.api_key)

    # Step 6-8
    top = ranked[: args.num_clips]
    print(f"\n[Step 6-8] 剪辑输出 {len(top)} 个短视频...")
    for i, seg in enumerate(top):
        out_path = os.path.join(
            args.output,
            f"short_{i+1:02d}_score{seg.score:.0f}.mp4",
        )
        export_clip(video_path, seg, out_path, whisper_segs, i, work_dir)

    print("\n" + "=" * 50)
    print(f"  完成！共输出 {len(top)} 个短视频")
    print(f"  目录: {os.path.abspath(args.output)}")
    print("=" * 50)


if __name__ == "__main__":
    main()

