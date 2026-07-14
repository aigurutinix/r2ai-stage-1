from typing import List, Optional, Dict, Any
import re

from GeneralTreeConstruct.token_counter import TokenCounter

class TreeNode:
    __slots__ = (
        "text",
        "level",
        "node_type",
        "children",
        "parent",
        "index",
        "number",
        "is_edit",
        "content_raw",
        "document_code",
    )

    def __init__(
        self,
        text: str,
        level: Optional[int],
        node_type: str,
        index: Optional[int] = None,
        number: str = "",
        is_edit: bool = False,
        content_raw: Optional[str] = None,
        document_code: str = "",
        parent: Optional["TreeNode"] = None,
    ):
        self.text = text.strip()
        self.level = level
        self.node_type = node_type
        self.children: List["TreeNode"] = []
        self.parent = parent
        self.index = index
        self.number = number
        self.is_edit = is_edit
        self.content_raw = content_raw if content_raw is not None else self.text
        self.document_code = document_code

    def add_child(self, node: "TreeNode"):
        self.children.append(node)
        node.parent = self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": "" if self.index is None else str(self.index),
            "level": self.level,
            "type": self.node_type,
            "number": self.number,
            "content": self.text,
            "is_edit": self.is_edit,
            "children": [child.to_dict() for child in self.children],
            "num_tokens": 0,
            "content_raw": self.content_raw,
            "document_code": self.document_code,
        }


# ==============================

from GeneralTreeConstruct.config.models import patterns
class StructuredDocumentTreeBuilder:
    def __init__(self,):
        self.compiled_patterns = [re.compile(pattern) for pattern in patterns]
        self.token_counter = TokenCounter()

    def detect_title_level(self, text: str) -> Optional[int]:
        for idx, pattern in enumerate(self.compiled_patterns):
            if pattern.match(text):
                return idx
        return None
    

    # ==========================
    # Case 1: Build from level
    # ==========================

    def build_tree_from_level(self, items):
        """
        Build a tree structure from a flat list of items with 'level' field.
        Rules:
        - level 1 > level 2 (level 1 is parent of level 2)
        - level -1 is always the lowest level (child of nearest preceding node)
        """
        if not items:
            return []

        # Add children field to all items (deep copy to avoid mutation)
        import copy
        nodes = [copy.deepcopy(item) for item in items]
        for node in nodes:
            node["children"] = []

        root = []
        # Stack lưu các node theo level thực: [(effective_level, node)]
        # level -1 được coi là level cao nhất (vô cực) → luôn là con
        stack = []  # list of (effective_level, node_ref)

        def effective_level(lvl):
            """Convert level to comparable int. -1 means deepest (treated as infinity)."""
            return float('inf') if lvl == -1 else lvl

        for node in nodes:
            el = effective_level(node["level"])

            if not stack:
                # Stack rỗng → node thuộc root (chỉ khi level != -1)
                # hoặc level -1 đầu tiên cũng thuộc root
                root.append(node)
                stack.append((el, node))
            else:
                # Pop stack cho đến khi tìm được parent phù hợp
                # Parent phù hợp là node có effective_level < el hiện tại
                while stack and stack[-1][0] >= el:
                    stack.pop()

                if not stack:
                    # Không còn parent → thuộc root
                    root.append(node)
                else:
                    # Parent là top of stack
                    stack[-1][1]["children"].append(node)

                stack.append((el, node))

        return root
    
    def build_tree_from_detection(self, flattened_data: List[Dict[str, Any]]) -> List[TreeNode]:
        """
        Xây dựng cấu trúc cây từ danh sách dictionary.
        
        Args:
            flattened_data: List of dicts với format:
                {
                    'text': str,
                    'level': int (khởi tạo = -1),
                    'type': str,
                    'index': int (optional)
                }
        
        Returns:
            List of TreeNode (root nodes)
        """
        # Convert dicts to TreeNodes
        nodes = []
        for data in flattened_data:
            text_value = data.get("content", data.get("text", ""))
            node = TreeNode(
                text=text_value,
                level=data.get('level', -1),
                node_type=data.get('type', 'nội dung'),
                index=data.get('id', data.get('index')),
                number=data.get('number', ''),
                is_edit=data.get('is_edit', False),
                content_raw=data.get('content_raw', text_value),
                document_code=data.get('document_code', ''),
            )
            nodes.append(node)
        
        nodes =  self._build_tree_by_detection(nodes)
        return [node.to_dict() for node in nodes]
    
    
    def detect_pattern_level(self, text: str) -> Optional[int]:
        """
        Phát hiện level của text dựa trên pattern khớp.
        
        Args:
            text: Text cần kiểm tra
            
        Returns:
            Index của pattern khớp (0-based), hoặc None nếu không khớp
        """
        text = text.strip()
        for idx, pattern in enumerate(self.compiled_patterns):
            if pattern.match(text):
                return idx
        return None
    

    # ==========================
    # Case 2: Build from detection
    # ==========================

    def _build_tree_by_detection(self, nodes: List[TreeNode]) -> List[TreeNode]:
        """
        Xây dựng cấu trúc cây phân cấp từ danh sách TreeNode phẳng.
        
        Logic ĐỘNG:
        - Level được quyết định bởi THỨ TỰ XUẤT HIỆN, không phải pattern index
        - Title xuất hiện trước làm cha
        - Title xuất hiện sau (cùng pattern hoặc khác pattern) làm con
        - Khi gặp title mới:
          + Nếu cùng pattern_id với node trước = anh em (cùng level)
          + Nếu khác pattern_id = con/cháu (level sâu hơn)
        - Content (không match pattern) luôn là leaf node với level = -1
        
        Args:
            nodes: Danh sách TreeNode (level có thể chưa được gán)
            
        Returns:
            List of root TreeNodes với cấu trúc cây đã xây dựng
        """
        if not nodes:
            return []
        
        root_nodes = []
        stack = []  # Stack chứa: [(node, actual_level), ...]
        pattern_to_level = {}  # Map: pattern_id -> level đã gán
        next_level = 0  # Level tiếp theo chưa được dùng
        
        for node in nodes:
            pattern_id = self.detect_pattern_level(node.text)
            
            if pattern_id is None:
                # Content node (không match pattern nào)
                node.level = -1
                
                if stack:
                    # Thêm vào parent gần nhất
                    stack[-1][0].add_child(node)
                else:
                    # Không có parent, thêm vào root
                    root_nodes.append(node)
                    
            else:
                # Title node (match một pattern)
                
                # Quyết định level cho node này
                if pattern_id in pattern_to_level:
                    # Pattern đã gặp trước đó, dùng lại level đã gán
                    current_level = pattern_to_level[pattern_id]
                    
                    # Pop stack để tìm parent phù hợp
                    # Loại bỏ tất cả node có level >= current_level
                    while stack and stack[-1][1] >= current_level:
                        stack.pop()
                        
                else:
                    # Pattern mới lần đầu gặp
                    if not stack:
                        # Không có parent, đây là root level
                        current_level = next_level
                        next_level += 1
                    else:
                        # Có parent, level = parent_level + 1
                        parent_level = stack[-1][1]
                        current_level = parent_level + 1
                        
                        # Cập nhật next_level nếu cần
                        if current_level >= next_level:
                            next_level = current_level + 1
                    
                    # Lưu mapping pattern -> level
                    pattern_to_level[pattern_id] = current_level
                
                node.level = current_level
                
                # Thêm vào parent (nếu có)
                if stack:
                    stack[-1][0].add_child(node)
                else:
                    root_nodes.append(node)
                
                # Thêm vào stack để có thể làm parent cho node sau
                stack.append((node, current_level))
        
        return root_nodes
    
    def normalize_level(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for item in items:
            if item["level"] == -1:
                item["level"] = 7
            else:
                item["level"] = item["level"] + 1

        return items    

    def build(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:

        # Check level exists
        has_real_level = any(
            item.get("level") not in (None, -1)
            for item in items
        )

        nodes = []
    
        # if has_real_level:
            # nodes =  self.build_tree_from_level(items)
        # else:
        nodes = self.build_tree_from_detection(items)

        nodes = self.normalize_level(nodes)

        for node in nodes[:3]:
            if node["type"] == "nội dung":
                node["type"] = "title"
                node["level"] = -1

        _ = [self.token_counter.calculate_tokens_recursive(node) for node in nodes]

        return nodes
