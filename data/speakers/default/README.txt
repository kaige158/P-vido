# 参考音频录制指引

## 4GB 笔记本零样本参考音

请放置：ref_10s.wav（5~15 秒干净人声，44.1kHz mono）

## 录制要求

- 采样率：44.1kHz 或 48kHz，单声道（mono）
- 环境：安静房间，关闭风扇/空调/窗户
- 口距：距麦克风 15-25 cm，避免喷麦（可用防喷罩）
- 电平：峰值 -6dB 到 -3dB，避免削波

## 录制内容（10 秒示例）

自然说话即可，例如：
  "大家好，我是[你的名字]，今天我想和大家分享一下关于语音合成技术的一些想法。"

## 录制工具（Windows 推荐）

1. Audacity（免费）：https://www.audacityteam.org/
   - 设置：Edit > Preferences > Audio Settings > 44100 Hz, Mono
   - 导出：File > Export > Export as WAV, Signed 16-bit PCM
2. 或直接用系统自带的"语音录音机"

## 录制后检查

- 首尾不要有多余的长静音（可用 Audacity 裁剪）
- 确认文字完全正确，没有口误
- 用 Audacity 打开确认没有削波（红色区域）

## 配置（config.yaml）

录制完成后在 config.yaml 中填写：
  tts.engines.gpt_sovits.prompt_text — 与 ref_10s.wav 完全一致的文本
