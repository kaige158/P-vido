"""
GPT-SoVITS 独立 TTS 推理脚本（绕过 Gradio WebUI，直接调用 Python API）。

用法:
    cd third_party/GPT-SoVITS
    python ../../scripts/gpt_sovits_infer.py \
        --text "你好世界" \
        --ref_audio ../../data/speakers/default/ref_19s.WAV \
        --prompt_text "雨下得毫无预兆。" \
        --output ../../output/tts_out.wav
"""

import argparse
import os
import sys
import time

# 必须在其他导入之前设置工作目录
GPT_SOVITS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPT_SOVITS_ROOT = os.path.join(GPT_SOVITS_DIR, "third_party", "GPT-SoVITS")

os.chdir(GPT_SOVITS_ROOT)
sys.path.insert(0, GPT_SOVITS_ROOT)
sys.path.insert(0, os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS"))

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""

import numpy as np
import soundfile as sf
import torch
import yaml

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config


def load_config(config_path: str) -> dict:
    config_path = os.path.join(GPT_SOVITS_DIR, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_default_model_paths(config: dict) -> dict:
    """从 config.yaml 推断模型路径和设备设置。"""
    hardware = config.get("hardware", {})
    device = hardware.get("device", "cuda")
    if device not in ("cuda", "cpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_half = device == "cuda" and torch.cuda.is_available()

    version = "v2"
    t2s_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    vits_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    cnhub_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"

    if not os.path.exists(t2s_path):
        import glob as _g
        t2s_candidates = _g.glob("GPT_SoVITS/pretrained_models/**/*.ckpt", recursive=True)
        vits_candidates = _g.glob("GPT_SoVITS/pretrained_models/**/*.pth", recursive=True)
        if t2s_candidates:
            t2s_path = t2s_candidates[0]
        if vits_candidates:
            vits_path = vits_candidates[0]

    return {
        "custom": {
            "device": device,
            "is_half": is_half,
            "version": version,
            "t2s_weights_path": t2s_path,
            "vits_weights_path": vits_path,
            "bert_base_path": bert_path,
            "cnhuhbert_base_path": cnhub_path,
        }
    }


def run_tts(
    text: str,
    ref_audio_path: str,
    prompt_text: str = "",
    prompt_lang: str = "zh",
    text_lang: str = "zh",
    speed: float = 1.0,
    model_config: dict = None,
    output_path: str = None,
):
    """执行 TTS 推理，返回 (sample_rate, audio_numpy)。"""
    if model_config is None:
        cfg = load_config("config.yaml")
        model_config = get_default_model_paths(cfg)

    tts_config = TTS_Config(model_config)
    print(f"设备: {tts_config.device}, 半精度: {tts_config.is_half}, 版本: {tts_config.version}")
    print(f"T2S: {tts_config.t2s_weights_path}")
    print(f"VITS: {tts_config.vits_weights_path}")

    tts = TTS(tts_config)

    inputs = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": [],
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut1",
        "batch_size": 1,
        "speed_factor": speed,
        "split_bucket": True,
        "return_fragment": False,
        "fragment_interval": 0.3,
        "seed": -1,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "super_sampling": False,
        "streaming_mode": False,
    }

    t0 = time.time()
    result = None
    for sr_out, audio_out in tts.run(inputs):
        result = (sr_out, audio_out)
    elapsed = time.time() - t0

    if result is None:
        raise RuntimeError("TTS 推理未返回音频")

    sr, audio = result
    print(f"合成完成: {len(audio)/sr:.1f} 秒, 采样率 {sr}, 耗时 {elapsed:.1f}s")

    if output_path:
        sf.write(output_path, audio, sr)
        print(f"已保存: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS 独立 TTS 推理")
    parser.add_argument("--text", required=True, help="合成文本")
    parser.add_argument("--ref_audio", required=True, help="参考音频路径")
    parser.add_argument("--prompt_text", default="", help="参考音频对应文本")
    parser.add_argument("--prompt_lang", default="zh", help="参考文本语种")
    parser.add_argument("--text_lang", default="zh", help="合成文本语种")
    parser.add_argument("--speed", type=float, default=1.0, help="语速 (0.9~1.1)")
    parser.add_argument("--output", default=None, help="输出 WAV 路径")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU 推理")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    model_cfg = get_default_model_paths(cfg)

    if args.cpu:
        model_cfg["device"] = "cpu"
        model_cfg["is_half"] = False

    run_tts(
        text=args.text,
        ref_audio_path=args.ref_audio,
        prompt_text=args.prompt_text,
        prompt_lang=args.prompt_lang,
        text_lang=args.text_lang,
        speed=args.speed,
        model_config=model_cfg,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
