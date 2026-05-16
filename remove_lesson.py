import os
import json
import shutil

def main():
    index_path = "data/index.json"
    
    if not os.path.exists(index_path):
        print("ไม่พบไฟล์ index.json (ยังไม่มีข้อมูลบทเรียน)")
        return
        
    with open(index_path, "r", encoding="utf-8") as f:
        try:
            index_data = json.load(f)
        except json.JSONDecodeError:
            print("ไฟล์ index.json เสียหาย")
            return
            
    if not index_data:
        print("ไม่มีบทเรียนให้ลบ")
        return
        
    print("\n=== รายการบทเรียนทั้งหมด ===")
    for idx, lesson in enumerate(index_data, 1):
        print(f"[{idx}] {lesson.get('title', 'Unknown')} (ID: {lesson.get('id', 'Unknown')})")
        
    print("\nพิมพ์ตัวเลขของบทเรียนที่ต้องการลบ (หรือพิมพ์ 'q' เพื่อออก)")
    choice = input("เลือก: ").strip()
    
    if choice.lower() == 'q':
        return
        
    if not choice.isdigit():
        print("กรุณาใส่ตัวเลขที่ถูกต้อง")
        return
        
    choice_idx = int(choice) - 1
    if choice_idx < 0 or choice_idx >= len(index_data):
        print("ตัวเลขไม่อยู่ในรายการ")
        return
        
    target_lesson = index_data[choice_idx]
    vid_id = target_lesson["id"]
    
    # ยืนยันการลบ
    confirm = input(f"\nแน่ใจหรือไม่ที่จะลบ '{target_lesson.get('title', 'Unknown')}'? (y/n): ").strip().lower()
    if confirm != 'y':
        print("ยกเลิกการลบ")
        return
        
    print("\nกำลังลบข้อมูล...")
    
    # 1. ลบจาก index_data
    index_data.pop(choice_idx)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=4)
        
    # 2. ลบไฟล์ lesson_{vid_id}.json
    lesson_json = os.path.join("data", f"lesson_{vid_id}.json")
    if os.path.exists(lesson_json):
        os.remove(lesson_json)
        print(f"- ลบไฟล์ {lesson_json} แล้ว")
        
    # 3. ลบโฟลเดอร์ audio/{vid_id}
    audio_dir = os.path.join("audio", vid_id)
    if os.path.exists(audio_dir):
        shutil.rmtree(audio_dir)
        print(f"- ลบโฟลเดอร์ {audio_dir} แล้ว")
        
    print(f"\n✅ ลบบทเรียนเรียบร้อยแล้ว!")

if __name__ == "__main__":
    main()
