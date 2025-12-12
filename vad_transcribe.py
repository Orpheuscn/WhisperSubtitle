#!/usr/bin/env python3
"""
连续语音识别脚本 - 针对对白稀疏的音频（改进版）
检测连续语音片段，分别识别后合并字幕
使用 pyannote segmentation-3.0 VAD + OpenAI Whisper 进行语音识别

注意：需要在虚拟环境中运行以使用 pyannote，同时系统需要安装 openai-whisper
"""
import argparse
import subprocess
import os
import json
import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# 导入虚拟环境的 torch 和 pyannote
import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection


def extract_audio(input_file: str, output_audio: str) -> None:
    """从视频或音频中提取音频，ffmpeg会自动处理所有格式"""
    print(f"正在提取音频（ffmpeg自动检测格式）...")
    cmd = [
        'ffmpeg', '-i', input_file,
        '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
        '-y', output_audio
    ]
    subprocess.run(cmd, check=True)


def detect_continuous_speech_segments(audio_file: str,
                                      silence_threshold_sec: float = 2.0,
                                      speech_pad_ms: int = 300,
                                      enable_second_pass: bool = True) -> List[Tuple[float, float]]:
    """
    使用 pyannote segmentation-3.0 VAD 检测连续语音片段

    参数:
        silence_threshold_sec: 静音阈值（秒），超过此时长视为片段分隔
        speech_pad_ms: 语音片段前后扩展的毫秒数，避免截断
        enable_second_pass: 是否启用第二轮检测（pyannote VAD 精度高，通常不需要）

    返回:
        连续语音片段列表 [(开始ms, 结束ms), ...]
    """
    print(f"正在检测连续语音片段（静音阈值: {silence_threshold_sec}秒）...")

    # 加载 pyannote segmentation-3.0 VAD 模型
    print("正在加载 pyannote segmentation-3.0 VAD 模型...")
    try:
        # 加载 segmentation 模型（token 通过环境变量 HF_TOKEN 自动读取）
        model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=os.environ.get("HF_TOKEN")
        )

        # 创建 VAD pipeline
        vad_pipeline = VoiceActivityDetection(segmentation=model)

        # 设置超参数
        HYPER_PARAMETERS = {
            # 移除短于此秒数的语音片段
            "min_duration_on": 0.0,
            # 填充短于此秒数的非语音片段
            "min_duration_off": 0.0
        }
        vad_pipeline.instantiate(HYPER_PARAMETERS)

        print("模型加载完成\n")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("提示: 如果需要 HuggingFace token，请设置环境变量 HF_TOKEN")
        print("请访问 https://huggingface.co/pyannote/segmentation-3.0 接受用户条件")
        raise

    # 读取音频文件（使用 soundfile 以兼容 pyannote）
    print(f"正在分析音频文件...")
    import soundfile as sf
    import numpy as np

    # 读取音频数据
    waveform, sample_rate = sf.read(audio_file)

    # 如果是立体声，转换为单声道
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    audio_length = len(waveform) / sample_rate
    audio_length_ms = audio_length * 1000
    print(f"音频读取完成，总长度: {audio_length:.2f}秒\n")

    # 使用 pyannote 进行语音检测
    print("=" * 60)
    print("使用 pyannote segmentation-3.0 检测语音片段")
    print("=" * 60)

    # 准备音频数据（pyannote 需要的格式）
    audio_data = {
        "waveform": torch.from_numpy(waveform[np.newaxis, :]).float(),
        "sample_rate": sample_rate
    }

    # 运行 VAD
    vad_result = vad_pipeline(audio_data)

    # 提取语音片段
    speech_segments = []
    for segment, _, label in vad_result.itertracks(yield_label=True):
        start_ms = segment.start * 1000  # 转换为毫秒
        end_ms = segment.end * 1000

        # 添加 padding
        start_ms = max(0, start_ms - speech_pad_ms)
        end_ms = min(audio_length_ms, end_ms + speech_pad_ms)

        speech_segments.append((start_ms, end_ms))

    if not speech_segments:
        print("未检测到任何语音")
        return []

    print(f"检测到 {len(speech_segments)} 个语音片段")

    # 合并相近的语音片段
    merged_segments = merge_speech_segments_from_tuples(
        speech_segments,
        silence_threshold_sec
    )

    print(f"\n合并后: {len(merged_segments)} 个连续语音片段")
    for i, (start_ms, end_ms) in enumerate(merged_segments):
        print(f"  片段{i+1}: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s "
              f"(时长: {(end_ms-start_ms)/1000:.2f}s)")

    print(f"\n" + "=" * 60)
    print(f"检测完成！总计: {len(merged_segments)} 个连续语音片段")
    print("=" * 60)

    return merged_segments


def merge_speech_segments_from_tuples(segments: List[Tuple[float, float]],
                                     silence_threshold_sec: float) -> List[Tuple[float, float]]:
    """合并tuple格式的语音片段"""
    if not segments:
        return []
    
    # 先按开始时间排序
    segments = sorted(segments, key=lambda x: x[0])
    
    silence_threshold_ms = silence_threshold_sec * 1000
    merged_segments = []
    
    current_start, current_end = segments[0]
    
    for i in range(1, len(segments)):
        next_start, next_end = segments[i]
        
        # 如果间隔小于阈值，合并
        if (next_start - current_end) < silence_threshold_ms:
            current_end = max(current_end, next_end)
        else:
            # 保存当前片段
            merged_segments.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    # 添加最后一个片段
    merged_segments.append((current_start, current_end))
    
    return merged_segments


def get_silence_regions(speech_segments: List[Tuple[float, float]], 
                       audio_length_ms: float) -> List[Tuple[float, float]]:
    """获取静音区域（第一轮未检测到语音的区域）"""
    if not speech_segments:
        return [(0, audio_length_ms)]
    
    silence_regions = []
    
    # 开头的静音
    if speech_segments[0][0] > 0:
        silence_regions.append((0, speech_segments[0][0]))
    
    # 中间的静音
    for i in range(len(speech_segments) - 1):
        silence_start = speech_segments[i][1]
        silence_end = speech_segments[i + 1][0]
        if silence_end > silence_start:
            silence_regions.append((silence_start, silence_end))
    
    # 结尾的静音
    if speech_segments[-1][1] < audio_length_ms:
        silence_regions.append((speech_segments[-1][1], audio_length_ms))
    
    return silence_regions


def cut_audio_segment(audio_file: str, start_ms: float, end_ms: float, 
                     output_file: str) -> None:
    """切割单个音频片段"""
    start_sec = start_ms / 1000
    duration_sec = (end_ms - start_ms) / 1000
    
    cmd = [
        'ffmpeg', '-i', audio_file,
        '-ss', str(start_sec),
        '-t', str(duration_sec),
        '-ar', '16000', '-ac', '1',
        '-y', output_file
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)


def check_segment_completed(segment_index: int, output_dir: str) -> bool:
    """检查片段是否已经识别完成"""
    json_file = os.path.join(output_dir, f"segment_{segment_index:04d}.json")
    return os.path.exists(json_file) and os.path.getsize(json_file) > 0


def load_existing_result(segment_index: int, output_dir: str) -> Dict:
    """加载已存在的识别结果"""
    json_file = os.path.join(output_dir, f"segment_{segment_index:04d}.json")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        segment_count = len(result.get('segments', []))
        print(f"    ✓ 使用已有结果: {segment_count} 个字幕片段")
        return result
    except Exception as e:
        print(f"    ⚠️ 读取已有结果失败: {e}")
        return {'segments': []}


def transcribe_with_whisper(audio_file: str, language: str, model: str,
                           output_dir: str, max_retries: int = 3) -> Dict:
    """使用系统的 Whisper 命令行工具转录音频，支持重试"""
    print(f"    使用 Whisper 识别: {os.path.basename(audio_file)}")

    # 显示音频时长信息
    try:
        import wave
        with wave.open(audio_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        print(f"    音频时长: {duration:.1f}秒")
    except Exception:
        pass

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"    重试 {attempt}/{max_retries}...")

        try:
            # 构建 whisper 命令
            cmd = [
                "whisper",
                audio_file,
                "--model", model,
                "--output_dir", output_dir,
                "--output_format", "json",
                "--verbose", "False"
            ]

            # 添加语言参数
            if language:
                cmd.extend(["--language", language.lower()])
                print(f"    使用指定语言: {language}")
            else:
                print(f"    使用自动语言检测")

            # 执行 whisper 命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # 读取生成的 JSON 文件
            audio_basename = os.path.basename(audio_file).replace('.wav', '')
            json_file = os.path.join(output_dir, f"{audio_basename}.json")

            with open(json_file, 'r', encoding='utf-8') as f:
                transcription_result = json.load(f)

            # 重新保存 JSON 文件，使用 ensure_ascii=False 以显示原始文本而非 Unicode 编码
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(transcription_result, f, ensure_ascii=False, indent=2)

            segment_count = len(transcription_result.get('segments', []))
            print(f"    ✓ 识别完成: {segment_count} 个字幕片段")

            return transcription_result

        except Exception as e:
            print(f"    ⚠️ faster-whisper 识别失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue

    print(f"    ✗ 识别失败，已重试{max_retries}次")
    return {'segments': []}


def milliseconds_to_srt_time(ms: float) -> str:
    """将毫秒转换为SRT时间格式 HH:MM:SS,mmm"""
    hours = int(ms // 3600000)
    ms %= 3600000
    minutes = int(ms // 60000)
    ms %= 60000
    seconds = int(ms // 1000)
    milliseconds = int(ms % 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def generate_srt(all_segments: List[Dict], output_file: str) -> None:
    """生成SRT字幕文件"""
    print("\n正在生成SRT字幕文件...")
    
    subtitle_index = 1
    with open(output_file, 'w', encoding='utf-8') as f:
        for seg_info in all_segments:
            segment_start_ms = seg_info['segment_start_ms']
            whisper_result = seg_info['whisper_result']
            
            # 处理whisper识别的每个片段
            for segment in whisper_result.get('segments', []):
                # whisper的时间戳是相对于当前片段的秒数
                # 需要加上当前片段在原始音频中的起始位置
                start_ms = segment_start_ms + (segment['start'] * 1000)
                end_ms = segment_start_ms + (segment['end'] * 1000)
                text = segment['text'].strip()
                
                if text:  # 只有非空文本才写入
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{milliseconds_to_srt_time(start_ms)} --> "
                           f"{milliseconds_to_srt_time(end_ms)}\n")
                    f.write(f"{text}\n")
                    f.write("\n")
                    subtitle_index += 1
    
    print(f"SRT字幕已生成: {output_file}")
    print(f"共 {subtitle_index - 1} 条字幕")


def save_segments_cache(temp_dir: str, segments: List[Tuple[float, float]]):
    """保存片段信息到缓存文件"""
    cache_file = os.path.join(temp_dir, 'segments_cache.json')
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"片段信息已保存到缓存")


def load_segments_cache(temp_dir: str) -> List[Tuple[float, float]]:
    """从缓存文件加载片段信息"""
    cache_file = os.path.join(temp_dir, 'segments_cache.json')
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            print(f"✓ 从缓存加载了 {len(segments)} 个片段信息")
            return [tuple(seg) for seg in segments]
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description='连续语音识别（改进版） - 针对对白稀疏的音频，支持断点续传'
    )
    parser.add_argument('input_file', help='输入视频或音频文件')
    parser.add_argument('--language', default=None,
                       help='语言名称（如：Japanese, English, Chinese）。不指定时Whisper会自动检测')
    parser.add_argument('--model', default='base',
                       help='Whisper模型（默认：base）')
    parser.add_argument('--silence-threshold', type=float, default=2.0,
                       help='静音阈值（秒），超过此时长视为片段分隔（默认：2.0）')
    parser.add_argument('--speech-pad', type=int, default=300,
                       help='语音片段前后padding（毫秒），避免截断（默认：300）')
    parser.add_argument('--force-redetect', action='store_true',
                       help='强制重新检测语音片段（忽略缓存）')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"错误：文件不存在 - {args.input_file}")
        return

    # 在当前目录创建temp_continuous文件夹
    temp_dir = os.path.join(os.getcwd(), 'temp_continuous')
    os.makedirs(temp_dir, exist_ok=True)
    print(f"临时目录: {temp_dir}")
    print("注意：处理完成后temp_continuous文件夹不会自动删除，请手动清理\n")

    # 步骤1: 提取音频
    audio_file = os.path.join(temp_dir, 'extracted_audio.wav')
    if not os.path.exists(audio_file):
        extract_audio(args.input_file, audio_file)
        print()
    else:
        print(f"✓ 使用已提取的音频文件\n")

    # 步骤2: 检测连续语音片段（优先使用缓存，断点续传时跳过重新检测）
    continuous_segments = None
    if not args.force_redetect:
        continuous_segments = load_segments_cache(temp_dir)
        if continuous_segments:
            print(f"✓ 使用缓存的语音片段信息 ({len(continuous_segments)} 个片段)\n")

    if continuous_segments is None:
        print("正在检测连续语音片段...")
        continuous_segments = detect_continuous_speech_segments(
            audio_file,
            args.silence_threshold,
            args.speech_pad
        )
        # 保存到缓存
        save_segments_cache(temp_dir, continuous_segments)
        print()

    if not continuous_segments:
        print("未检测到语音片段")
        return
    
    # 步骤3 & 4: 切割并识别每个连续片段
    print("开始处理各个连续语音片段...\n")
    all_segments = []

    # 统计信息
    total_segments = len(continuous_segments)
    completed_count = 0
    skipped_count = 0
    failed_count = 0

    for i, (start_ms, end_ms) in enumerate(continuous_segments):
        print(f"【片段 {i+1}/{total_segments}】")
        print(f"  位置: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s")
        print(f"  时长: {(end_ms - start_ms)/1000:.2f}s")

        # 检查是否已经识别完成
        if check_segment_completed(i, temp_dir):
            print(f"  ⏭️  片段已识别，跳过")
            whisper_result = load_existing_result(i, temp_dir)
            skipped_count += 1
        else:
            # 切割音频片段
            segment_file = os.path.join(temp_dir, f"segment_{i:04d}.wav")

            # 如果音频文件不存在，才切割
            if not os.path.exists(segment_file):
                cut_audio_segment(audio_file, start_ms, end_ms, segment_file)
            else:
                print(f"  使用已有音频片段")

            # 用Whisper识别
            whisper_result = transcribe_with_whisper(
                segment_file,
                args.language,
                args.model,
                temp_dir
            )

            if whisper_result.get('segments'):
                completed_count += 1
            else:
                failed_count += 1

        # 保存结果
        all_segments.append({
            'segment_index': i,
            'segment_start_ms': start_ms,
            'segment_end_ms': end_ms,
            'whisper_result': whisper_result
        })

        print()

    # 显示统计信息
    print("\n" + "="*50)
    print("处理统计:")
    print(f"  总片段数: {total_segments}")
    print(f"  新识别: {completed_count}")
    print(f"  跳过(已完成): {skipped_count}")
    print(f"  失败: {failed_count}")
    print("="*50 + "\n")
    
    # 步骤5: 生成SRT文件
    output_srt = Path(args.input_file).stem + '.srt'
    generate_srt(all_segments, output_srt)
    
    # 输出统计信息
    total_speech_duration = sum(
        seg['segment_end_ms'] - seg['segment_start_ms'] 
        for seg in all_segments
    ) / 1000
    
    # 读取原始音频总时长
    import wave
    with wave.open(audio_file, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        total_duration = frames / float(rate)
    
    print("\n" + "="*50)
    print("处理完成！")
    print("="*50)
    print(f"原始音频时长: {total_duration:.2f}秒")
    print(f"语音片段总时长: {total_speech_duration:.2f}秒")
    print(f"语音占比: {total_speech_duration/total_duration*100:.1f}%")
    print(f"连续片段数量: {len(continuous_segments)}")
    print(f"SRT文件: {output_srt}")
    print(f"临时文件: {temp_dir}")


if __name__ == '__main__':
    main()
