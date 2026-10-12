from datetime import datetime

def get_current_datetime_string(lang="ko"):
    now = datetime.now()
    if lang == "ko":
        weekday = ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]
        return f"{now.year}년 {now.month}월 {now.day}일 {weekday}요일 {now.hour}시 {now.minute}분"
    # 필요시 다국어 포맷 추가
    return now.strftime("%Y-%m-%d %H:%M:%S")
