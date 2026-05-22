"""
批量 TTS 合成 + 断点续传：将纯文本分段后逐段调用 GPT-SoVITS 引擎，记录状态。

用法：
    python scripts/tts_batch.py --input output/text_with_fillers/ --output output/audio/
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# GPT-SoVITS 路径
_PROJECT_DIR = Path(__file__).resolve().parent.parent
_GPT_SOVITS_ROOT = _PROJECT_DIR / "third_party" / "GPT-SoVITS"

# 全局 TTS 实例（惰性初始化）
_tts_instance = None
_tts_config = None


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_into_segments(text: str, max_chars: int = 300, min_chars: int = 50) -> list[str]:
    import re

    sentences = re.split(r"(?<=[。！？])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    segments = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current += sent
        else:
            if len(current) >= min_chars:
                segments.append(current)
            current = sent
    if current.strip():
        segments.append(current)

    return segments


# ---- TTS 引擎适配层 ----

def _setup_gpt_sovits_path():
    """设置 GPT-SoVITS 的 Python 路径。"""
    gpt_sovits_module = str(_GPT_SOVITS_ROOT / "GPT_SoVITS")
    if str(_GPT_SOVITS_ROOT) not in sys.path:
        sys.path.insert(0, str(_GPT_SOVITS_ROOT))
    if gpt_sovits_module not in sys.path:
        sys.path.insert(0, gpt_sovits_module)
    os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
    os.environ["all_proxy"] = ""


def init_tts_engine(config: dict):
    """初始化 TTS 引擎（在 GPU 上加载模型，只执行一次）。"""
    global _tts_instance, _tts_config

    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    hardware = config.get("hardware", {})
    device = hardware.get("device", "cuda")
    if device not in ("cuda", "cpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_half = device == "cuda" and torch.cuda.is_available()

    tts_cfg = config.get("tts", {}).get("engines", {}).get("gpt_sovits", {})

    model_cfg = {
        "custom": {
            "device": device,
            "is_half": is_half,
            "version": "v2",
            "t2s_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        }
    }

    # 参考音频路径：处理相对路径（相对于项目根目录）
    ref_audio = tts_cfg.get("ref_audio_path", "")
    if ref_audio and not os.path.isabs(ref_audio):
        ref_audio = str((_PROJECT_DIR / ref_audio).resolve())

    _tts_config = {
        "ref_audio_path": ref_audio,
        "prompt_text": tts_cfg.get("prompt_text", ""),
        "prompt_lang": tts_cfg.get("prompt_lang", "zh"),
        "speed": tts_cfg.get("speed", 1.0),
    }

    logger.info("加载 GPT-SoVITS 模型 (device=%s, half=%s)...", device, is_half)
    tts_cfg_obj = TTS_Config(model_cfg)
    _tts_instance = TTS(tts_cfg_obj)
    logger.info("GPT-SoVITS 模型加载完成")


def tts_gpt_sovits(text: str, config: dict, segment_index: int, output_dir: Path) -> Path:
    """使用 GPT-SoVITS 直接 Python API 合成单段音频。"""
    global _tts_instance, _tts_config

    out_path = output_dir / f"segment_{segment_index:05d}.wav"

    if _tts_instance is None:
        raise RuntimeError("TTS 引擎未初始化，请先调用 init_tts_engine()")

    try:
        inputs = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": _tts_config["ref_audio_path"],
            "aux_ref_audio_paths": [],
            "prompt_text": _tts_config["prompt_text"],
            "prompt_lang": _tts_config["prompt_lang"],
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut1",
            "batch_size": 1,
            "speed_factor": _tts_config["speed"],
            "split_bucket": True,
            "return_fragment": False,
            "fragment_interval": 0.3,
            "seed": -1,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": 32,
            "super_sampling": False,
        }

        result = None
        for sr_out, audio_out in _tts_instance.run(inputs):
            result = (sr_out, audio_out)

        if result is None:
            raise RuntimeError("TTS 推理未返回音频")

        sr, audio = result
        sf.write(str(out_path), audio, sr)
        logger.debug("段 %05d: %.1fs 音频 @ %dHz", segment_index, len(audio) / sr, sr)
        return out_path

    except Exception as e:
        logger.warning("GPT-SoVITS 推理失败 (%s)，生成静音占位", e)

    # 静音占位（回退）
    import wave
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00" * 44100 * 2)
    return out_path


def tts_cosyvoice2(text: str, config: dict, segment_index: int, output_dir: Path) -> Path:
    """CosyVoice2（已禁用）。"""
    raise RuntimeError("CosyVoice2 已禁用（4GB 显存不足）。请使用 gpt_sovits。")


ENGINE_MAP = {
    "gpt_sovits": tts_gpt_sovits,
    "cosyvoice2": tts_cosyvoice2,
}


# ---- 状态管理 ----


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "progress": {}}


def save_state(state_path: Path, state: dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="批量 TTS 合成")
    parser.add_argument("--input", required=True, help="纯文本文件或目录")
    parser.add_argument("--output", default="output/audio/", help="音频输出目录")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--engine", help="TTS 引擎: gpt_sovits | cosyvoice2（默认用 config）")
    parser.add_argument("--preview", action="store_true", help="预览模式：只打印分段，不合成")
    parser.add_argument("--limit", type=int, help="只合成前 N 段（测试用）")
    args = parser.parse_args()

    config = load_config(args.config)
    engine_name = args.engine or config["tts"]["default_engine"]

    if engine_name not in ENGINE_MAP:
        logger.error("未知引擎: %s。可用: %s", engine_name, list(ENGINE_MAP.keys()))
        sys.exit(1)

    # 初始化 TTS 引擎（GPU 模型加载，仅一次）
    if engine_name == "gpt_sovits":
        # 在切换 CWD 前将路径转为绝对路径
        input_path = Path(args.input).resolve()
        output_dir = Path(args.output).resolve()
        _setup_gpt_sovits_path()
        os.chdir(str(_GPT_SOVITS_ROOT))
        init_tts_engine(config)

    # 读取文本
    if input_path.is_file():
        text = input_path.read_text(encoding="utf-8")
    elif input_path.is_dir():
        texts = []
        for f in sorted(input_path.rglob("*.txt")):
            texts.append(f.read_text(encoding="utf-8"))
        text = "\n\n".join(texts)
    else:
        logger.error("输入路径不存在: %s", input_path)
        sys.exit(1)

    # 分段
    seg_cfg = config.get("segment", {})
    max_chars = seg_cfg.get("max_chars", 300)
    min_chars = seg_cfg.get("min_chars", 50)
    segments = split_into_segments(text, max_chars, min_chars)
    logger.info("共 %d 段待合成", len(segments))

    if args.preview:
        for i, seg in enumerate(segments[: args.limit or 5]):
            print(f"\n--- 段 {i:05d} ({len(seg)} 字) ---")
            print(seg)
        return

    if args.limit:
        segments = segments[: args.limit]

    # 状态
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    state = load_state(state_path)

    tts_fn = ENGINE_MAP[engine_name]
    batch_cfg = config.get("batch", {})
    retry_count = batch_cfg.get("retry_count", 2)

    # 合成
    pause_sec = seg_cfg.get("pause_between_segments_sec", 2)
    preview_count = batch_cfg.get("preview_first_segments", 0)

    for i, seg in enumerate(segments):
        seg_key = f"segment_{i:05d}"
        if seg_key in state["completed"]:
            logger.info("[%d/%d] 跳过已完成: %s", i + 1, len(segments), seg_key)
            continue

        # 预览模式：只合前 N 段
        if preview_count > 0 and i >= preview_count:
            logger.info("预览限制：已合成 %d 段，跳过剩余 %d 段。试听无误后将 preview_first_segments 置 0。", preview_count, len(segments) - i)
            break

        logger.info("[%d/%d] 合成中 (%d 字)...", i + 1, len(segments), len(seg))

        for attempt in range(1 + retry_count):
            try:
                audio_path = tts_fn(seg, config, i, output_dir)
                state["completed"].append(seg_key)
                state["progress"][seg_key] = {
                    "text": seg,
                    "audio": str(audio_path),
                    "chars": len(seg),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_state(state_path, state)
                break
            except Exception as e:
                logger.error("[%d/%d] 失败 (尝试 %d/%d): %s", i + 1, len(segments), attempt + 1, 1 + retry_count, e)
                if attempt == retry_count:
                    if batch_cfg.get("resume_on_error", True):
                        state["failed"].append({"key": seg_key, "text": seg, "error": str(e)})
                        save_state(state_path, state)
                    else:
                        logger.critical("遇到失败且 resume_on_error=false，退出。")
                        sys.exit(1)
                time.sleep(1)

        # 4GB 显存：段间暂停释放 GPU 缓存
        if pause_sec > 0 and i + 1 < len(segments) and (preview_count == 0 or i + 1 < preview_count):
            time.sleep(pause_sec)

    # 输出摘要
    completed = len(state["completed"])
    failed = len(state["failed"])
    logger.info("合成结束。成功: %d, 失败: %d, 总计: %d", completed, failed, len(segments))

    # 导出段落元数据（用于字幕/时间轴）
    metadata_path = output_dir / "segments_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(state["progress"], f, ensure_ascii=False, indent=2)
    logger.info("段落元数据已导出: %s", metadata_path)


if __name__ == "__main__":
    main()
