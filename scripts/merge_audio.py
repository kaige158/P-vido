"""
音频拼接：将分段 TTS 合成的 WAV 合并为完整音频文件。

用法：
    python scripts/merge_audio.py --input output/audio/ --output output/final/
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# 搜索 ffmpeg/ffprobe 路径（便携优先）
import shutil as _shutil

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FFMPEG_LOCAL = str(_PROJECT_ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe")
_FFPROBE_LOCAL = str(_PROJECT_ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe")

FFMPEG = _FFMPEG_LOCAL if os.path.exists(_FFMPEG_LOCAL) else _shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = _FFPROBE_LOCAL if os.path.exists(_FFPROBE_LOCAL) else _shutil.which("ffprobe") or "ffprobe"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_segment_files(audio_dir: Path, state_path: Path) -> list[Path]:
    """从状态文件获取待合并的音频段列表（按顺序）。"""
    if not state_path.exists():
        # 回退：按文件名排序
        wavs = sorted(audio_dir.glob("segment_*.wav"))
        return wavs

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    files = []
    for key in sorted(state.get("completed", [])):
        info = state.get("progress", {}).get(key, {})
        audio_path = info.get("audio", "")
        if audio_path and Path(audio_path).exists():
            files.append(Path(audio_path))

    return files


def add_silence_between_segments(segment_list_path: Path, silence_ms: int):
    """生成带静音间隔的 concat 文件列表（用于 ffmpeg concat demuxer）。"""
    # 格式:
    # file 'path/to/segment_00000.wav'
    # duration 0.3
    # file 'path/to/segment_00001.wav'
    pass


def merge_with_ffmpeg(
    segment_files: list[Path],
    output_path: Path,
    silence_ms: int = 300,
    export_format: str = "wav",
    mp3_bitrate_k: int = 192,
) -> Path:
    """使用 ffmpeg concat 拼接音频段。"""
    import shutil

    out_file = output_path / f"merged.{export_format}"

    if len(segment_files) == 0:
        logger.error("没有可合并的音频段")
        sys.exit(1)

    # 先输出到临时 WAV（避免多格式时互相覆盖）
    temp_wav = output_path / f"_merge_tmp_{export_format}.wav"

    if len(segment_files) == 1:
        shutil.copy2(segment_files[0], temp_wav)
    else:
        # 分批合并
        batch_size = 100
        temp_files = []

        for batch_start in range(0, len(segment_files), batch_size):
            batch = segment_files[batch_start : batch_start + batch_size]
            concat_list_path = output_path / f"_concat_{batch_start:05d}.txt"
            with open(concat_list_path, "w", encoding="utf-8") as f:
                for seg_file in batch:
                    f.write(f"file '{seg_file.absolute().as_posix()}'\n")
                    if silence_ms > 0:
                        f.write(f"duration {silence_ms / 1000:.1f}\n")

            batch_out = output_path / f"_batch_{batch_start:05d}.wav"
            cmd = [FFMPEG, "-f", "concat", "-safe", "0", "-i", str(concat_list_path),
                   "-ac", "1", "-ar", "44100", "-y", str(batch_out)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("ffmpeg concat 失败: %s", result.stderr)
                sys.exit(1)
            temp_files.append(batch_out)
            concat_list_path.unlink(missing_ok=True)

        if len(temp_files) == 1:
            shutil.move(str(temp_files[0]), str(temp_wav))
        else:
            batch_concat = output_path / "_all_batches.txt"
            with open(batch_concat, "w", encoding="utf-8") as f:
                for tf in temp_files:
                    f.write(f"file '{tf.absolute().as_posix()}'\n")
            cmd = [FFMPEG, "-f", "concat", "-safe", "0", "-i", str(batch_concat),
                   "-ac", "1", "-y", str(temp_wav)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("ffmpeg 批次合并失败: %s", result.stderr)
                sys.exit(1)
            batch_concat.unlink(missing_ok=True)
            for tf in temp_files:
                Path(tf).unlink(missing_ok=True)

    # 格式转换
    if export_format == "mp3":
        cmd = [FFMPEG, "-i", str(temp_wav), "-b:a", f"{mp3_bitrate_k}k", "-y", str(out_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("MP3 转码失败: %s", result.stderr)
        temp_wav.unlink(missing_ok=True)
    else:
        shutil.move(str(temp_wav), str(out_file))

    # 获取时长
    probe_cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(out_file)]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

    logger.info("合并完成: %s (%.1f 秒, %d 段)", out_file.name, duration, len(segment_files))
    return out_file


def main():
    parser = argparse.ArgumentParser(description="音频拼接")
    parser.add_argument("--input", default="output/audio/", help="分段音频目录")
    parser.add_argument("--output", default="output/final/", help="合并输出目录")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--format", choices=["wav", "mp3"], help="输出格式（默认用 config）")
    args = parser.parse_args()

    config = load_config(args.config)
    audio_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = audio_dir / "state.json"
    segment_files = get_segment_files(audio_dir, state_path)

    if not segment_files:
        logger.error("未找到分段音频文件")
        sys.exit(1)

    logger.info("找到 %d 个分段", len(segment_files))

    silence_ms = config.get("audio", {}).get("silence_duration_ms", 300)
    export_cfg = config.get("export", {})
    export_formats = [args.format] if args.format else export_cfg.get("formats", ["wav"])
    mp3_bitrate_k = export_cfg.get("mp3_bitrate_k", 192)

    for fmt in export_formats:
        merge_with_ffmpeg(segment_files, output_dir, silence_ms, fmt, mp3_bitrate_k)

    # 生成章节/字幕元数据（如果配置了）
    if export_cfg.get("generate_subtitle", False) and state_path.exists():
        generate_subtitle(state_path, output_dir)


def generate_subtitle(state_path: Path, output_dir: Path):
    """根据 segment 元数据生成简单 SRT 字幕和时间轴 JSON。"""
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    timeline = []
    current_time = 0.0

    for key in sorted(state.get("progress", {}).keys()):
        info = state["progress"][key]
        # 实际应用中需要 probe 每个音频文件的真实时长
        # 这里仅做骨架
        est_duration = len(info["text"]) / 3.5  # 粗略估计：每秒约3.5字
        timeline.append(
            {
                "segment": key,
                "start": round(current_time, 2),
                "end": round(current_time + est_duration, 2),
                "text": info["text"],
            }
        )
        current_time += est_duration

    timeline_path = output_dir / "timeline.json"
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    logger.info("时间轴已导出: %s", timeline_path)


if __name__ == "__main__":
    main()
