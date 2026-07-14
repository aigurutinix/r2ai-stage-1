import re
class TitlePattern:
    def __init__(self, regex, level):
        self.regex = re.compile(regex)
        self.level = level

patterns = [
    r"^PHẦN\s+[IVXLCDM]+",              # PHẦN I, PHẦN II, PHẦN III
    r"^Phần\s+[IVXLCDM]+",              # Phần I, Phần II (viết thường)
    r"^CHƯƠNG\s+[IVXLCDM\d]+",          # CHƯƠNG I, CHƯƠNG 1
    r"^Chương\s+[IVXLCDM\d]+",          # Chương I, Chương 1
    r"^MỤC\s+\d+",                      # MỤC 1, MỤC 2
    r"^Mục\s+\d+",                      # Mục 1, Mục 2
    r"^TIẾT\s+\d+",                     # TIẾT 1, TIẾT 2
    r"^Tiết\s+\d+",                     # Tiết 1, Tiết 2
    r"^ĐIỀU\s+\d+",                     # ĐIỀU 1, ĐIỀU 2
    r"^Điều\s+\d+",                     # Điều 1, Điều 2
    r"^\d+\.\s",                        # 1. 2. 3.
    r"^\d+\.\d+\.\s",                   # 1.1. 1.2. 2.1.
    r"^\d+\.\d+\.\d+\.\s",              # 1.1.1. 1.1.2.
    r"^[a-z]\)\s",                      # a) b) c)
    r"^[a-z]\.\d+\)\s",                 # a.1) a.2) b.1)
]