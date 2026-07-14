import re
import unicodedata


class LevelDetector:
    def __init__(self):
        self.level_map = {
            "title": 0,
            "chương": 1,
            "phụ lục": 1,
            "mục": 2,
            "phần": 1,
            "điều": 3,
            "điều_ver2": 3,
            "khoản": 4,
            "điểm": 5,
            "điểm con": 6,
            "nội dung": 7,
            "table": 7
        }
        self.level_patterns = {
            "chương": r"^Chương\s+([IVX]+|(\d+))",
            "mục": r"^mục\s+([IVX]+|(\d+))",
            "phụ lục": r"^PHỤ\s+LỤC\s+[IVX]+(?:\.[A-Z])?",
            "phần": r"^Phần\s*(\d+)[.)-]",
            "điều": r"^Điều\s*(\d+)[.)-]",
            # "điều_ver2": r"^([IVX]+)[.)-]",
            "điều_ver2": r"^(?![a-zđ])[IVXLCDM]+[.)-]",  
            "khoản": r"^(\d+)\s*[.)-]",
            "điểm": r'^([a-zđ])([.)-])\s+\.*',
            "điểm con": r"(^[a-zđ]\.\d+)[.)-]\s+\.*",
            "nội dung": r".+",  # Matches any text if no other pattern matches
            "table": r"<table>"
        }
        
    def normalize_vietnamese(self,text):
        return unicodedata.normalize('NFC', text)

    def check_title(self, text: str) -> bool:
        norm_text = text.strip()
        # check if contain word or number 
        if norm_text.isupper():
            return True
        return False
    
    def check_table(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        return re.search(r"<table>", text, re.IGNORECASE) is not None
    
    def get_level(self, text: str) -> dict:
        text = self.normalize_vietnamese(text)
        is_title = self.check_title(text)
        is_table = self.check_table(text)
        for key, pattern in self.level_patterns.items():
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    num_id = matches.group(1)
                except Exception as e:
                    num_id = ""
                if is_title and key == "nội dung":
                    return {
                        "type": "title",
                        "level_id": 0,
                        "number": ""
                    }
                if is_table and key == "nội dung":
                    return {
                        "type": "table",
                        "level_id": 7,
                        "number": ""
                    }
                return {
                    "type": key if key not in ["điểm con"] else "điểm",
                    "level_id": self.level_map.get(key),
                    "number": num_id
                }
        
        return {
            "type": "nội dung",
            "level_id": 7,
            "number": ""
        }
        
if __name__ == "__main__":
    text = "Điều 65. Sửa đổi giờ học"
    level_detector = LevelDetector()
    print(level_detector.get_level(text))