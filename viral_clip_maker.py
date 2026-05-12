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
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

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

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    print("[警告] opencv-python 未安装，将跳过人脸居中，使用画面中心裁剪")
    _CV2_AVAILABLE = False

# ============================================================
# 默认参数
# ============================================================
DEFAULT_OUTPUT_DIR = "./output_shorts"
MAX_CLIPS          = 5
CLIP_MIN_DURATION  = 30.0
CLIP_MAX_DURATION  = 90.0
TARGET_WIDTH       = 1080
TARGET_HEIGHT      = 1920
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


def score_with_claude(
    segments: List[Segment],
    api_key: Optional[str] = None,
    model: str = CLAUDE_MODEL,
) -> List[Segment]:
    client = anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    payload = [
        {
            "index": i,
            "duration_s": round(seg.end - seg.start, 1),
            "text": seg.text[:600],
        }
        for i, seg in enumerate(segments)
    ]

    print(f"[Step 5] Claude 打分（{len(segments)} 个片段）...")
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": SCORE_PROMPT.format(
                segments_json=json.dumps(payload, ensure_ascii=False, indent=2)
            ),
        }],
    )

    raw = resp.content[0].text
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
# Step 6: 获取视频分辨率
# ============================================================
def get_video_size(video_path: str) -> Tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(out.stdout)["streams"][0]
    return info["width"], info["height"]


# ============================================================
# Step 6: 人脸居中竖屏裁剪参数
# ============================================================
def build_crop_filter(video_path: str, seg: Segment) -> str:
    src_w, src_h = get_video_size(video_path)

    # 竖屏目标比例 9:16，从原视频裁出合适宽度
    crop_w = int(src_h * 9 / 16)
    crop_w = min(crop_w, src_w)
    crop_h = src_h

    # 人脸居中
    face_cx = src_w // 2  # 默认画面中心
    if _CV2_AVAILABLE:
        face_cx = _detect_face_cx(video_path, (seg.start + seg.end) / 2, src_w)

    crop_x = max(0, min(face_cx - crop_w // 2, src_w - crop_w))

    return (
        f"crop={crop_w}:{crop_h}:{crop_x}:0,"
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
    )


def _detect_face_cx(video_path: str, timestamp: float, frame_width: int) -> int:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return frame_width // 2

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return frame_width // 2

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return x + w // 2


# ============================================================
# Step 7: 生成 SRT 字幕
# ============================================================
def _srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")


def write_srt(whisper_segs: List[dict], seg: Segment, srt_path: str) -> None:
    lines = []
    idx = 1
    for ws in whisper_segs:
        if ws["end"] <= seg.start or ws["start"] >= seg.end:
            continue
        s = max(ws["start"], seg.start) - seg.start
        e = min(ws["end"], seg.end) - seg.start
        text = ws["text"].strip()
        if not text or e <= s:
            continue
        lines += [str(idx), f"{_srt_time(s)} --> {_srt_time(e)}", text, ""]
        idx += 1
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Step 8: ffmpeg 剪辑 + 字幕烧录
# ============================================================
def export_clip(
    video_path: str,
    seg: Segment,
    out_path: str,
    whisper_segs: List[dict],
    clip_idx: int,
    work_dir: str,
) -> None:
    srt_path = os.path.join(work_dir, f"clip_{clip_idx:02d}.srt")
    write_srt(whisper_segs, seg, srt_path)

    crop_filter = build_crop_filter(video_path, seg)
    duration = seg.end - seg.start

    # ffmpeg 的 subtitles 滤镜路径在 Windows 上需要转义冒号，macOS/Linux 直接用
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    subtitle_style = (
        "FontName=Arial,"
        "FontSize=64,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BackColour=&H80000000,"
        "Outline=2,Shadow=1,"
        "Alignment=2,MarginV=100"
    )
    subtitle_filter = f"subtitles='{srt_escaped}':force_style='{subtitle_style}'"

    vf = f"{crop_filter},{subtitle_filter}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seg.start),
        "-i", video_path,
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]

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
