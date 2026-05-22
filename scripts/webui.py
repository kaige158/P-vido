"""
语音克隆系统 — Gradio WebUI

用法:
    双击 启动WebUI.bat
    或: python scripts/webui.py
"""

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr
import yaml

# ── 路径初始化 ──────────────────────────────────────────────
_PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(str(_PROJECT_DIR))
sys.path.insert(0, str(_PROJECT_DIR))

# GPT-SoVITS 路径
_GPT_SOVITS_ROOT = _PROJECT_DIR / "third_party" / "GPT-SoVITS"
sys.path.insert(0, str(_GPT_SOVITS_ROOT))
sys.path.insert(0, str(_GPT_SOVITS_ROOT / "GPT_SoVITS"))
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""

# ── 配置 ────────────────────────────────────────────────────
CONFIG_PATH = _PROJECT_DIR / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


# ── TTS 引擎（惰性初始化）────────────────────────────────────
_tts_instance = None
_tts_model_cfg = None


def get_tts_instance():
    global _tts_instance, _tts_model_cfg
    if _tts_instance is not None:
        return _tts_instance, _tts_model_cfg

    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    cfg = load_config()
    hardware = cfg.get("hardware", {})
    device = hardware.get("device", "cuda")
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    is_half = device == "cuda"

    tts_cfg = cfg.get("tts", {}).get("engines", {}).get("gpt_sovits", {})
    ref_audio = tts_cfg.get("ref_audio_path", "")
    if ref_audio and not os.path.isabs(ref_audio):
        ref_audio = str((_PROJECT_DIR / ref_audio).resolve())

    _tts_model_cfg = {
        "ref_audio_path": ref_audio,
        "prompt_text": tts_cfg.get("prompt_text", "").strip(),
        "prompt_lang": tts_cfg.get("prompt_lang", "zh"),
        "speed": tts_cfg.get("speed", 1.0),
    }

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

    os.chdir(str(_GPT_SOVITS_ROOT))
    tts_config_obj = TTS_Config(model_cfg)
    _tts_instance = TTS(tts_config_obj)
    os.chdir(str(_PROJECT_DIR))

    return _tts_instance, _tts_model_cfg


# ── 文档处理 ────────────────────────────────────────────────
def process_document(file_obj, pasted_text):
    """处理上传的文档或粘贴的文本，返回清洗后的纯文本。"""
    from scripts.doc_to_text import extract_markdown, extract_txt, extract_pdf, normalize_text

    text = ""
    if file_obj is not None:
        file_path = Path(file_obj.name)
        suffix = file_path.suffix.lower()
        if suffix == ".md":
            text = file_path.read_text(encoding="utf-8")
            text = extract_markdown(text)
        elif suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
            text = extract_txt(text)
        elif suffix == ".pdf":
            text = extract_pdf(file_path)
        else:
            return f"不支持的文件类型: {suffix}"
    elif pasted_text and pasted_text.strip():
        text = pasted_text.strip()
        # 如果是纯文本，也走一次 markdown 清洗（去掉可能的格式标记）
        text = extract_markdown(text)

    if not text:
        return "请上传文件或粘贴文本"

    text = normalize_text(text)
    return text


def do_insert_fillers(text, enable_fillers, filler_rate):
    """在文本中插入语气词。"""
    if not enable_fillers or not text:
        return text

    from scripts.insert_fillers import insert_fillers as do_insert

    cfg = load_config()
    # 用 UI 的设置临时覆盖 config
    cfg["fillers"]["enabled"] = enable_fillers
    cfg["fillers"]["rate"] = filler_rate
    return do_insert(text, cfg)


# ── TTS 合成 ─────────────────────────────────────────────────
def split_text(text, max_chars=120, min_chars=30):
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


def synthesize(text, speed, progress=gr.Progress()):
    """对输入文本分段合成，返回进度信息和音频文件列表。"""
    segments = split_text(text)
    if not segments:
        yield "文本为空，无法合成", None, None
        return

    tts, model_cfg = get_tts_instance()
    ref_audio = model_cfg["ref_audio_path"]
    prompt_text = model_cfg["prompt_text"]
    prompt_lang = model_cfg["prompt_lang"]

    if not ref_audio or not os.path.exists(ref_audio):
        yield "参考音频不存在，请检查 config.yaml 中的 ref_audio_path", None, None
        return

    import numpy as np
    import soundfile as sf

    audio_dir = _PROJECT_DIR / "output" / "audio_ui"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    for old in audio_dir.glob("segment_*.wav"):
        old.unlink()

    total = len(segments)
    output_files = []
    failed = 0

    for i, seg in enumerate(segments):
        progress((i, total), desc=f"合成中 ({i+1}/{total}): {len(seg)}字")
        out_path = audio_dir / f"segment_{i:05d}.wav"

        try:
            inputs = {
                "text": seg,
                "text_lang": "zh",
                "ref_audio_path": ref_audio,
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
            }
            result = None
            for sr_out, audio_out in tts.run(inputs):
                result = (sr_out, audio_out)

            if result is not None:
                sr, audio_data = result
                sf.write(str(out_path), audio_data, sr)
                output_files.append(str(out_path))
            else:
                failed += 1
        except Exception as e:
            failed += 1
            traceback.print_exc()

        # 段间暂停释放 GPU 缓存
        if i + 1 < total:
            time.sleep(1.5)

    status = f"合成完成: {total - failed}/{total} 段成功"
    if failed > 0:
        status += f"，{failed} 段失败"

    # 保存元数据
    meta = {
        "total": total,
        "success": total - failed,
        "failed": failed,
        "speed": speed,
        "segments": [{"index": i, "text": s, "chars": len(s)} for i, s in enumerate(segments)],
    }
    with open(audio_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    yield status, str(audio_dir), json.dumps(meta, ensure_ascii=False, indent=2)


def merge_audio_files(audio_dir, export_wav, export_mp3, silence_ms, mp3_bitrate):
    """合并分段音频为最终文件。"""
    if not audio_dir or not Path(audio_dir).exists():
        return None, None, "请先合成音频"

    from scripts.merge_audio import merge_with_ffmpeg

    audio_path = Path(audio_dir)
    segment_files = sorted(audio_path.glob("segment_*.wav"))
    if not segment_files:
        return None, None, "没有找到音频分段"

    output_dir = _PROJECT_DIR / "output" / "final_ui"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.iterdir():
        old.unlink()

    results = []
    if export_wav:
        try:
            merge_with_ffmpeg(segment_files, output_dir, silence_ms, "wav", mp3_bitrate)
            results.append(str(output_dir / "merged.wav"))
        except Exception as e:
            results.append(f"WAV合并失败: {e}")

    if export_mp3:
        try:
            merge_with_ffmpeg(segment_files, output_dir, silence_ms, "mp3", mp3_bitrate)
            results.append(str(output_dir / "merged.mp3"))
        except Exception as e:
            results.append(f"MP3合并失败: {e}")

    wav_file = str(output_dir / "merged.wav") if (output_dir / "merged.wav").exists() else None
    mp3_file = str(output_dir / "merged.mp3") if (output_dir / "merged.mp3").exists() else None

    status = "合并完成"
    if results:
        status = "; ".join([r for r in results if isinstance(r, str) and not r.endswith((".wav", ".mp3"))])

    return wav_file, mp3_file, status


# ── UI 构建 ──────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 900px !important; }
.header { text-align: center; margin-bottom: 1em; }
.header h1 { font-size: 1.8em; color: #1a56db; }
.status-ok { color: #059669; font-weight: bold; }
.status-err { color: #dc2626; font-weight: bold; }
"""

SPEED_INFO = """
语速控制：**0.9** = 偏慢（适合朗读散文） · **1.0** = 正常 · **1.1** = 偏快（适合新闻播报）
"""


def create_ui():
    cfg = load_config()
    tts_cfg = cfg.get("tts", {}).get("engines", {}).get("gpt_sovits", {})

    with gr.Blocks(title="语音克隆朗读系统", css=CSS, theme=gr.themes.Soft()) as app:
        gr.HTML("""
        <div class="header">
            <h1>🎙 语音克隆朗读系统</h1>
            <p>上传文档 → 文字处理 → TTS 合成 → 音频导出 | GPU: RTX 3050 4GB</p>
        </div>
        """)

        # ── 状态 ──
        raw_text = gr.State("")
        processed_text = gr.State("")
        audio_dir_state = gr.State("")

        with gr.Tabs():
            # ============================================================
            # Tab 1: 文本输入
            # ============================================================
            with gr.TabItem("📄 文本输入"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传文档")
                        file_input = gr.File(
                            label="支持 MD / TXT / PDF",
                            file_types=[".md", ".txt", ".pdf"],
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 或直接粘贴文本")
                        pasted_input = gr.Textbox(
                            label="",
                            placeholder="在此粘贴要朗读的文本...",
                            lines=8,
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        enable_fillers_checkbox = gr.Checkbox(
                            label="插入语气词（嗯、啊、那个...)",
                            value=cfg.get("fillers", {}).get("enabled", True),
                        )
                    with gr.Column(scale=1):
                        filler_rate_slider = gr.Slider(
                            label="语气词插入率",
                            minimum=0.0, maximum=0.15, step=0.01,
                            value=cfg.get("fillers", {}).get("rate", 0.05),
                        )

                process_btn = gr.Button("🔍 处理文本", variant="primary", size="lg")

                gr.Markdown("### 处理后文本（可编辑）")
                text_editor = gr.Textbox(
                    label="",
                    lines=15,
                    placeholder="处理后文本将显示在此，可直接编辑...",
                )
                char_count = gr.Textbox(label="字数统计", interactive=False)

                def handle_process(file_obj, pasted, enable_f, rate):
                    text = process_document(file_obj, pasted)
                    if text.startswith("请上传") or text.startswith("不支持"):
                        return text, text, f"错误: {text}"
                    text_with_fillers = do_insert_fillers(text, enable_f, rate)
                    return text_with_fillers, text_with_fillers, f"共 {len(text_with_fillers)} 字"

                process_btn.click(
                    fn=handle_process,
                    inputs=[file_input, pasted_input, enable_fillers_checkbox, filler_rate_slider],
                    outputs=[text_editor, processed_text, char_count],
                )

            # ============================================================
            # Tab 2: TTS 合成
            # ============================================================
            with gr.TabItem("🎤 TTS 合成"):
                gr.Markdown("### 合成设置")

                with gr.Row():
                    speed_slider = gr.Slider(
                        label="语速",
                        minimum=0.85, maximum=1.15, step=0.05,
                        value=tts_cfg.get("speed", 1.0),
                    )
                    max_chars_slider = gr.Slider(
                        label="每段最大字数（4GB 建议 ≤120）",
                        minimum=60, maximum=200, step=10,
                        value=cfg.get("segment", {}).get("max_chars", 120),
                    )

                gr.Markdown(SPEED_INFO)

                with gr.Row():
                    synthesize_btn = gr.Button("🎵 开始合成", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹ 停止", variant="stop")

                progress_status = gr.Textbox(label="合成状态", interactive=False)
                segment_info = gr.Textbox(label="分段详情", interactive=False, lines=5)

                # 第一段预览
                gr.Markdown("### 试听（合成完成后显示）")
                with gr.Row():
                    segment_audio = gr.Audio(label="分段试听", type="filepath", scale=2)
                    merged_audio = gr.Audio(label="完整音频", type="filepath", scale=2)

                def handle_synthesize(text, speed, max_chars):
                    progress = gr.Progress()
                    if not text or text.startswith("请上传") or text.startswith("错误"):
                        yield "请先在【文本输入】中处理文本", "", None, None
                        return

                    import re
                    sentences = re.split(r"(?<=[。！？])\s*", text)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    segments = []
                    cur = ""
                    for sent in sentences:
                        if len(cur) + len(sent) <= max_chars:
                            cur += sent
                        else:
                            if len(cur) >= 30:
                                segments.append(cur)
                            cur = sent
                    if cur.strip():
                        segments.append(cur)

                    if not segments:
                        yield "文本为空，无法合成", "", None, None
                        return

                    tts, model_cfg = get_tts_instance()
                    ref_audio = model_cfg["ref_audio_path"]
                    if not ref_audio or not os.path.exists(ref_audio):
                        yield "参考音频不存在！请检查 config.yaml → ref_audio_path", "", None, None
                        return

                    import numpy as np
                    import soundfile as sf

                    audio_dir = _PROJECT_DIR / "output" / "audio_ui"
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    for old in audio_dir.glob("segment_*.wav"):
                        try:
                            old.unlink()
                        except Exception:
                            pass

                    total = len(segments)
                    output_files = []
                    failed = 0
                    seg_details = []

                    for i, seg in enumerate(segments):
                        progress(i / total, desc=f"合成中 ({i+1}/{total}): {len(seg)}字")
                        out_path = audio_dir / f"segment_{i:05d}.wav"

                        try:
                            inputs = {
                                "text": seg,
                                "text_lang": "zh",
                                "ref_audio_path": ref_audio,
                                "aux_ref_audio_paths": [],
                                "prompt_text": model_cfg["prompt_text"],
                                "prompt_lang": model_cfg["prompt_lang"],
                                "top_k": 15, "top_p": 1.0, "temperature": 1.0,
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
                            }
                            os.chdir(str(_GPT_SOVITS_ROOT))
                            result = None
                            for sr_out, audio_out in tts.run(inputs):
                                result = (sr_out, audio_out)
                            os.chdir(str(_PROJECT_DIR))
                            if result is not None:
                                sr, audio_data = result
                                sf.write(str(out_path), audio_data, sr)
                                output_files.append(str(out_path))
                                seg_details.append(f"[OK] 段{i+1}: {len(seg)}字 → {len(audio_data)/sr:.1f}s")
                            else:
                                failed += 1
                                seg_details.append(f"[FAIL] 段{i+1}: {len(seg)}字")
                        except Exception as e:
                            failed += 1
                            seg_details.append(f"[ERR] 段{i+1}: {str(e)[:60]}")

                        if i + 1 < total:
                            time.sleep(1.5)

                    status = f"合成完成: {total - failed}/{total} 段成功 | 共 {total} 段"
                    details = "\n".join(seg_details)

                    # 第一段用于试听
                    preview_audio = output_files[0] if output_files else None

                    # 自动合并
                    merged = None
                    if len(output_files) >= 1:
                        try:
                            from scripts.merge_audio import merge_with_ffmpeg
                            out_dir = _PROJECT_DIR / "output" / "final_ui"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            for old in out_dir.iterdir():
                                try:
                                    old.unlink()
                                except Exception:
                                    pass
                            wav_files = sorted(audio_dir.glob("segment_*.wav"))
                            merge_with_ffmpeg(wav_files, out_dir, 300, "wav", 128)
                            merged_wav = out_dir / "merged.wav"
                            if merged_wav.exists():
                                merged = str(merged_wav)
                        except Exception:
                            pass

                    yield status, details, preview_audio, merged

                synthesize_btn.click(
                    fn=handle_synthesize,
                    inputs=[text_editor, speed_slider, max_chars_slider],
                    outputs=[progress_status, segment_info, segment_audio, merged_audio],
                )

            # ============================================================
            # Tab 3: 设置
            # ============================================================
            with gr.TabItem("⚙ 设置"):
                gr.Markdown("### 参考音频")

                ref_audio_display = gr.Textbox(
                    label="当前参考音频路径",
                    value=tts_cfg.get("ref_audio_path", ""),
                    interactive=False,
                )
                ref_audio_upload = gr.File(label="上传新参考音频（5~10秒，单声道 WAV）", file_types=[".wav"])
                upload_ref_btn = gr.Button("更新参考音频", variant="secondary")

                def update_ref_audio(file_obj):
                    if file_obj is None:
                        return "未选择文件"
                    import shutil
                    src = Path(file_obj.name)
                    dest = _PROJECT_DIR / "data" / "speakers" / "default" / src.name
                    shutil.copy2(str(src), str(dest))
                    cfg = load_config()
                    cfg["tts"]["engines"]["gpt_sovits"]["ref_audio_path"] = str(
                        Path("data/speakers/default") / src.name
                    )
                    save_config(cfg)
                    return str(dest)

                upload_ref_btn.click(
                    fn=update_ref_audio,
                    inputs=[ref_audio_upload],
                    outputs=[ref_audio_display],
                )

                gr.Markdown("### 关于")
                gr.Markdown("""
                - **引擎**: GPT-SoVITS v2 零样本推理
                - **GPU**: NVIDIA RTX 3050 Laptop 4GB (仅推理，不微调)
                - **语言**: 中英混合
                - **许可**: 非商用
                """)

        # ── 页脚 ──
        gr.HTML("""
        <div style="text-align:center;padding:1em;color:#888;font-size:0.85em;">
            <p>GPT-SoVITS v2 · CUDA GPU · 零样本推理</p>
        </div>
        """)

    return app


# ── 入口 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  语音克隆朗读系统 WebUI")
    print("  GPU: NVIDIA RTX 3050 Laptop 4GB")
    print("=" * 60)
    print()
    print("正在预加载 GPT-SoVITS 模型到 GPU...")
    os.chdir(str(_GPT_SOVITS_ROOT))
    tts, tts_cfg = get_tts_instance()
    os.chdir(str(_PROJECT_DIR))
    print(f"模型加载完成 (device={tts.configs.device}, half={tts.configs.is_half})")
    print()

    app = create_ui()
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
    )
