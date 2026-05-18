import os
import sys
import json
import time
import shutil
import subprocess
import yt_dlp
import cutlet
from faster_whisper import WhisperModel

# =====================================================================
# STEP 1: FIX CUDA/CUBLAS PATH FOR UBUNTU
# =====================================================================
if 'VIRTUAL_ENV' in os.environ:
    venv_path = os.environ['VIRTUAL_ENV']
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_base = os.path.join(venv_path, 'lib', py_version, 'site-packages', 'nvidia')
    cublas_path = os.path.join(nvidia_base, 'cublas', 'lib')
    cudnn_path = os.path.join(nvidia_base, 'cudnn', 'lib')
    nvrtc_path = os.path.join(nvidia_base, 'cuda_nvrtc', 'lib')
    
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if cublas_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{cublas_path}:{cudnn_path}:{nvrtc_path}:{current_ld_path}".strip(':')
        os.execv(sys.executable, [sys.executable] + sys.argv)

# =====================================================================
# CONFIGURATION
# =====================================================================
MAX_AUDIO_FOLDER_SIZE_MB = 500  # กำหนดลิมิตพื้นที่โฟลเดอร์ audio (MB) เพื่อไม่ให้เกินโควต้า GitHub

# Extra audio before/after each Whisper segment so clips do not cut off endings (e.g. "desu").
SEGMENT_PAD_START_SEC = 0.28
SEGMENT_PAD_END_SEC = 0.62
# Second translate attempt uses a wider window if the first looks like a hallucination.
TRANSLATE_RETRY_EXTRA_PAD_SEC = 0.45

# VAD: slightly longer silence required to split + more speech padding = smoother, longer chunks.
VAD_PARAMETERS = {
    "min_silence_duration_ms": 2400,
    "speech_pad_ms": 720,
}

# Substrings that usually mean Whisper guessed wrong on a short clip (not real dialogue).
_BAD_EN_SUBSTRINGS = (
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "see you next",
    "see you in the next",
    "goodbye everyone",
    "translating...",
    "(translating",
    "http://",
    "https://",
    "www.",
    "subtitle",
    "subtitles by",
    "captions by",
    "amara.org",
)

_TRANSLATE_PROMPT = (
    "Faithful English translation of spoken Japanese dialogue from anime or TV. "
    "Ignore music, outros, and channel promos. Translate only what is spoken."
)


def _get_audio_duration_seconds(path):
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return float(out.strip())
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


def _is_bad_english_translation(text):
    if not text or not text.strip():
        return True
    lower = text.lower()
    if lower.strip() in ("(translating...)", "translating..."):
        return True
    return any(s in lower for s in _BAD_EN_SUBSTRINGS)


def _translate_clip(model, audio_path, clip_start, clip_end, initial_prompt=None):
    kwargs = {
        "task": "translate",
        "language": "ja",
        "clip_timestamps": f"{clip_start},{clip_end}",
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    segments, _ = model.transcribe(audio_path, **kwargs)
    return "".join(s.text for s in segments).strip()


def _primary_clip_bounds(source_duration, seg_start, seg_end):
    """Time range used for MP3 export (and first translate pass)."""
    if source_duration is None:
        source_duration = float("inf")
    t0 = max(0.0, seg_start - SEGMENT_PAD_START_SEC)
    t1 = min(source_duration, seg_end + SEGMENT_PAD_END_SEC)
    if t1 <= t0:
        t1 = min(source_duration, seg_end + 0.05)
        t0 = max(0.0, seg_start - 0.05)
    return t0, t1


def _translate_segment_english(model, audio_path, source_duration, seg_start, seg_end):
    """Translate: first pass matches audio padding; retry uses wider clip + prompt if output looks wrong."""
    t0, t1 = _primary_clip_bounds(source_duration, seg_start, seg_end)
    en = _translate_clip(model, audio_path, t0, t1)

    if _is_bad_english_translation(en):
        if source_duration is None:
            source_duration = float("inf")
        t0 = max(0.0, seg_start - SEGMENT_PAD_START_SEC - TRANSLATE_RETRY_EXTRA_PAD_SEC)
        t1 = min(
            source_duration,
            seg_end + SEGMENT_PAD_END_SEC + TRANSLATE_RETRY_EXTRA_PAD_SEC,
        )
        if t1 <= t0:
            t1 = min(source_duration, seg_end + 0.05)
            t0 = max(0.0, seg_start - 0.05)
        en = _translate_clip(
            model, audio_path, t0, t1, initial_prompt=_TRANSLATE_PROMPT
        )

    if _is_bad_english_translation(en):
        return ""

    return en

def get_folder_size(folder):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def manage_storage():
    """ตรวจสอบขนาดโฟลเดอร์ audio และลบบทเรียนเก่าที่สุดถ้าเกินลิมิต"""
    audio_dir = "audio"
    if not os.path.exists(audio_dir): return
    
    while get_folder_size(audio_dir) / (1024 * 1024) > MAX_AUDIO_FOLDER_SIZE_MB:
        print(f"\n⚠️ พื้นที่จัดเก็บเกิน {MAX_AUDIO_FOLDER_SIZE_MB}MB ทำการลบบทเรียนที่เก่าที่สุด...")
        index_path = "data/index.json"
        if not os.path.exists(index_path): break
        
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
            
        if not index_data: break
        
        # ลบอันเก่าสุด (อันแรกสุดใน list)
        oldest_lesson = index_data.pop(0)
        vid_id = oldest_lesson["id"]
        
        # ลบโฟลเดอร์เสียง
        audio_path = os.path.join("audio", vid_id)
        if os.path.exists(audio_path):
            shutil.rmtree(audio_path)
            
        # ลบไฟล์ JSON ของบทเรียน
        lesson_json = os.path.join("data", f"lesson_{vid_id}.json")
        if os.path.exists(lesson_json):
            os.remove(lesson_json)
            
        # อัปเดต index.json ใหม่
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=4)
            
        print(f"🗑️ ลบบทเรียน '{oldest_lesson['title']}' เรียบร้อยเพื่อคืนพื้นที่")

def process():
    url = input("วางลิงก์ YouTube ที่นี่: ").strip()
    start_time_input = input("ระบุเวลาเริ่ม (นาที:วินาที เช่น 1:30) [เว้นว่างถ้าต้องการตั้งแต่ต้น]: ").strip()
    end_time_input = input("ระบุเวลาจบ (นาที:วินาที เช่น 24:15) [เว้นว่างถ้าต้องการจนจบ]: ").strip()
    
    # ดึงข้อมูลวิดีโอเพื่อเอา ID และ Title
    print("\n[1/5] กำลังดึงข้อมูลวิดีโอ...")
    ydl_opts_info = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info['id']
        video_title = info.get('title', f"Lesson {video_id}")
    
    # สร้างโครงสร้างโฟลเดอร์
    os.makedirs("data", exist_ok=True)
    audio_out_dir = os.path.join("audio", video_id)
    os.makedirs(audio_out_dir, exist_ok=True)
    
    # 1. Download Audio
    print("\n[2/5] กำลังดึงไฟล์เสียง...")
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'audio_source.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # 1.5 Crop Audio if needed
    if start_time_input or end_time_input:
        print("\n[2.5] กำลังตัดช่วงเสียง (ข้ามเพลงเปิด/ปิด)...")
        crop_cmd = ['ffmpeg', '-y']
        if start_time_input:
            crop_cmd.extend(['-ss', start_time_input])
        if end_time_input:
            crop_cmd.extend(['-to', end_time_input])
        crop_cmd.extend(['-i', 'audio_source.m4a', '-c', 'copy', 'audio_cropped.m4a'])
        subprocess.run(crop_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace('audio_cropped.m4a', 'audio_source.m4a')

    source_duration = _get_audio_duration_seconds("audio_source.m4a")

    # 2. Load Model
    print("\n[3/5] กำลังโหลด Model เข้า RTX 4070...")
    model = WhisperModel("medium", device="cuda", compute_type="float16")

    # 3. Transcribe
    print("\n[4/5] กำลังแกะไทม์ไลน์และประมวลผลเสียง...")
    start_time = time.time()
    segments_jp, _ = model.transcribe(
        "audio_source.m4a",
        language="ja",
        vad_filter=True,
        vad_parameters=VAD_PARAMETERS,
    )
    
    cutter = cutlet.Cutlet()
    
    structured_data = []
    
    print("\n[5/5] -> กำลังหั่นเสียงและบีบอัด (32kbps)...")
    for idx, seg in enumerate(segments_jp, 1):
        jp_text = seg.text.strip()
        if not jp_text:
            continue
            
        en_text = _translate_segment_english(
            model, "audio_source.m4a", source_duration, seg.start, seg.end
        )

        romaji_text = cutter.romaji(jp_text)

        segment_audio_filename = f"seg_{idx}.mp3"
        segment_audio_path = os.path.join(audio_out_dir, segment_audio_filename)

        clip_t0, clip_t1 = _primary_clip_bounds(source_duration, seg.start, seg.end)
        clip_duration = clip_t1 - clip_t0

        # บีบอัดเสียง Mono 32kbps (ช่วงเวลาขยายก่อน/หลังเล็กน้อยเพื่อไม่ตัดท้ายประโยค)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(clip_t0),
            "-t",
            str(clip_duration),
            "-i",
            "audio_source.m4a",
            "-acodec",
            "libmp3lame",
            "-ac",
            "1",
            "-b:a",
            "32k",
            segment_audio_path,
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        data_item = {
            "id": idx,
            "jp": jp_text,
            "rm": romaji_text,
            "en": en_text,
            "audio": f"audio/{video_id}/{segment_audio_filename}"
        }
        structured_data.append(data_item)
    
    # บันทึก lesson_{id}.json
    lesson_path = os.path.join("data", f"lesson_{video_id}.json")
    with open(lesson_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=4)
        
    # อัปเดต index.json
    index_path = "data/index.json"
    index_data = []
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
            
    # ลบของเก่าออกถ้ามี ID ซ้ำ แล้วเพิ่มของใหม่ต่อท้าย
    index_data = [item for item in index_data if item["id"] != video_id]
    index_data.append({
        "id": video_id,
        "title": video_title,
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=4)

    # ตรวจสอบและจัดการพื้นที่เก็บข้อมูล
    manage_storage()

    end_time = time.time()
    print(f"\n🎉 เตรียมข้อมูลสำเร็จ! ใช้เวลา: {(end_time - start_time)/60:.2f} นาที")
    
    if os.path.exists("audio_source.m4a"):
        os.remove("audio_source.m4a")
        
    print("\n📦 ระบบพร้อมแล้ว! กำลังทำการ Push ขึ้น GitHub อัตโนมัติ...")
    try:
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', f'Auto-update lesson: {video_title}'], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("✅ Push สำเร็จ! คุณสามารถปิดคอมพิวเตอร์และไปเรียนบนมือถือได้เลยครับ")
    except Exception as e:
        print("⚠️ ไม่สามารถ Push ขึ้น Git ได้อัตโนมัติ (อาจจะยังไม่ได้ init git ไว้ หรือตั้งค่า remote ไม่สมบูรณ์)")

if __name__ == "__main__":
    process()
