"""
参考音频预处理：切片、降噪、导出训练列表。

用法：
    python scripts/prepare_audio.py --input data/raw/ --output data/processed/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def trim_silence(audio_path: Path, threshold_db: float = -40, min_silence_ms: int = 500):
    """切除首尾静音段。依赖 pydub + ffmpeg。"""
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_leading_silence
    except ImportError:
        logger.warning("pydub 未安装，跳过静音切除。pip install pydub")
        return audio_path

    sound = AudioSegment.from_file(audio_path)
    start_ms = detect_leading_silence(sound, silence_threshold=threshold_db)
    if start_ms > 0:
        logger.info("  切除开头静音 %d ms", start_ms)
        sound = sound[start_ms:]

    end_silence = detect_leading_silence(sound.reverse(), silence_threshold=threshold_db)
    if end_silence > 0:
        sound = sound[: len(sound) - end_silence]

    sound.export(audio_path, format=audio_path.suffix.lstrip("."))
    return audio_path


def reduce_noise(audio_path: Path, noise_level: float = 0.02):
    """简易降噪。依赖 noisereduce。"""
    try:
        import noisereduce as nr
        import librosa
    except ImportError:
        logger.warning("noisereduce / librosa 未安装，跳过降噪。")
        return audio_path

    y_data, sr = librosa.load(str(audio_path), sr=None)
    reduced = nr.reduce_noise(y=y_data, sr=sr, prop_decrease=noise_level)
    import soundfile as sf

    sf.write(audio_path, reduced, sr)
    return audio_path


def normalize_loudness(audio_path: Path, target_lufs: float = -16.0):
    """响度归一。依赖 ffmpeg-normalize 或直接调用 ffmpeg。"""
    import subprocess

    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-y",
        str(audio_path.with_suffix(".norm.wav")),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("ffmpeg loudnorm 失败: %s", result.stderr)
        return audio_path
    # 替换原文件
    norm_path = audio_path.with_suffix(".norm.wav")
    audio_path.unlink()
    norm_path.rename(audio_path)
    return audio_path


def process_audio(input_dir: Path, output_dir: Path, config: dict) -> list[dict]:
    """遍历 raw 目录，处理每条音频并返回标注列表。"""
    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    entries = []

    for audio_file in input_dir.rglob("*"):
        if audio_file.suffix.lower() not in audio_exts:
            continue

        # 复制到 processed 目录
        rel = audio_file.relative_to(input_dir)
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copy2(audio_file, dest)
        logger.info("处理: %s", rel)

        # 处理流程（可选，默认跳过静音切除和降噪）
        target_lufs = config.get("audio", {}).get("target_lufs", -16)
        normalize_loudness(dest, target_lufs)

        # 查找对应文本标注文件
        txt_file = audio_file.with_suffix(".txt")
        transcript = ""
        if txt_file.exists():
            transcript = txt_file.read_text(encoding="utf-8").strip()

        entries.append(
            {
                "audio": str(dest),
                "transcript": transcript,
                "speaker": config.get("tts", {})
                .get("engines", {})
                .get("gpt_sovits", {})
                .get("speaker_name", "default"),
                "category": audio_file.parent.name,
            }
        )

    return entries


def main():
    parser = argparse.ArgumentParser(description="参考音频预处理")
    parser.add_argument("--input", default="data/raw/", help="原始音频目录")
    parser.add_argument("--output", default="data/processed/", help="处理后输出目录")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--skip-denoise", action="store_true", help="跳过降噪")
    parser.add_argument("--skip-normalize", action="store_true", help="跳过响度归一")
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error("输入目录不存在: %s", input_dir)
        sys.exit(1)

    entries = process_audio(input_dir, output_dir, config)

    # 导出标注列表（用于 TTS 训练/微调）
    list_path = output_dir / "annotations.json"
    with open(list_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    logger.info("标注列表已导出: %s (%d 条)", list_path, len(entries))


if __name__ == "__main__":
    main()
