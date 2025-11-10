import os
import subprocess
import re
import json
import shutil

SCENE_DIR = "scene"
QUALITY = "high_quality"  # 可选: "low_quality", "medium_quality", "high_quality"
FRAME_RATE = 60
OUTPUT_MP4 = "media/all_scenes_merged.mp4"
OUTPUT_SRT = "media/all_scenes_merged.srt"


def find_py_files(path):
    """只返回包含 Scene 类的 py 文件"""
    py_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".py") and not f.startswith("__"):
                full_path = os.path.join(root, f)
                with open(full_path, "r", encoding="utf-8") as file:
                    code = file.read()
                    # 检查是否有 Scene 子类定义
                    if "class " in code and "(Scene" in code or "(ThreeDScene" in code:
                        py_files.append(full_path)
    return sorted(py_files)


def render_all_scenes(py_file, quality=QUALITY, frame_rate=FRAME_RATE):
    """用 manim 命令行渲染 py 文件中的所有场景，生成字幕"""
    cmd = [
        "manim",
        "-a",
        "-q",
        quality[0],  # ql/qm/qh
        f"--frame_rate={frame_rate}",
        # "--write_srt",
        py_file,
    ]
    print("Rendering:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def collect_mp4_srt_files(py_files):
    mp4s, srts = [], []
    quality_to_resolution = {
        "low_quality": "480",
        "medium_quality": "720",
        "high_quality": "1080",
    }
    preferred_suffix = None
    if QUALITY in quality_to_resolution:
        preferred_suffix = f"{quality_to_resolution[QUALITY]}p{FRAME_RATE}"

    for py in py_files:
        basename = os.path.splitext(os.path.basename(py))[0]
        scene_dir = os.path.join("media", "videos", basename)
        if not os.path.isdir(scene_dir):
            continue

        candidate_dirs = []
        if preferred_suffix:
            expected_dir = os.path.join(scene_dir, preferred_suffix)
            if os.path.isdir(expected_dir):
                candidate_dirs.append(expected_dir)
        if not candidate_dirs:
            for entry in sorted(os.listdir(scene_dir)):
                full_path = os.path.join(scene_dir, entry)
                if os.path.isdir(full_path):
                    candidate_dirs.append(full_path)

        for out_dir in candidate_dirs:
            videos = sorted([f for f in os.listdir(out_dir) if f.endswith(".mp4")])
            if not videos:
                continue
            for v in videos:
                mp4_path = os.path.join(out_dir, v)
                srt_path = mp4_path.replace(".mp4", ".srt")
                mp4s.append(mp4_path)
                srts.append(srt_path if os.path.exists(srt_path) else None)
            break
    return mp4s, srts


def make_ffmpeg_list(mp4s, listfile="videos.txt"):
    with open(listfile, "w", encoding="utf-8") as f:
        for mp4 in mp4s:
            f.write(f"file '{os.path.abspath(mp4)}'\n")


def merge_mp4(mp4_list_file, output=OUTPUT_MP4):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        mp4_list_file,
        "-c",
        "copy",
        output,
    ]
    print("Merging videos:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# 字幕相关辅助函数
def srt_time_to_seconds(t):
    h, m, s_ms = t.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def seconds_to_srt_time(sec):
    ms = int(round((sec - int(sec)) * 1000))
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def shift_srt(srtfile, offset):
    pat = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")
    out = []
    with open(srtfile, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m:
                t1 = seconds_to_srt_time(srt_time_to_seconds(m.group(1)) + offset)
                t2 = seconds_to_srt_time(srt_time_to_seconds(m.group(2)) + offset)
                out.append(f"{t1} --> {t2}\n")
            else:
                out.append(line)
    return "".join(out)


def get_video_duration(mp4file):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        mp4file,
    ]
    output = subprocess.check_output(cmd)
    duration = float(json.loads(output)["format"]["duration"])
    return duration


def merge_srt(mp4s, srts, output=OUTPUT_SRT):
    offset = 0.0
    out_srt = []
    idx = 1
    for mp4, srt in zip(mp4s, srts):
        if srt is not None:
            srt_content = shift_srt(srt, offset)
            # 修正srt序号
            for line in srt_content.splitlines(keepends=True):
                if line.strip().isdigit():
                    out_srt.append(f"{idx}\n")
                    idx += 1
                else:
                    out_srt.append(line)
        # 没有 srt 的片段，不输出字幕内容，仅时间偏移
        offset += get_video_duration(mp4)
    with open(output, "w", encoding="utf-8") as f:
        f.writelines(out_srt)
    print(f"✅ 字幕已合并为 {output}")


if __name__ == "__main__":
    # 删除media/videos
    shutil.rmtree("media/videos", ignore_errors=True)
    py_files = find_py_files(SCENE_DIR)
    # 1. 逐个渲染所有py文件下的所有Scene（自动生成mp4/srt）
    for py in py_files:
        render_all_scenes(py)
    # 2. 收集所有mp4和srt，顺序严格一致
    mp4s, srts = collect_mp4_srt_files(py_files)
    # 3. 生成ffmpeg合并列表
    make_ffmpeg_list(mp4s)
    # 4. 合并视频
    merge_mp4("videos.txt")
    # 5. 合并字幕
    merge_srt(mp4s, srts)
    print(f"✅ 全部完成！合成视频：{OUTPUT_MP4}，合成字幕：{OUTPUT_SRT}")
