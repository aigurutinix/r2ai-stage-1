
class TreeBuilder:
    def build_tree(self, nodes: list) -> list:
        """
        Builds a nested JSON tree structure from a list of nodes.
        Each node should include an 'id', 'content', 'level', etc.
        Returns a list of root nodes with their children attached.
        """
        if not nodes:
            return []
        
        stack = []
        root_nodes = []
        min_level = 100  # a high initial value
        document_code = nodes[0].get("document_code")
        
        for current_node in nodes:
            node_id = current_node.get("id")
            current_content = current_node.get("content")
            current_content_raw = current_node.get("content_raw")
            level = current_node.get("level")
            current_level_id = level.get("level_id")
            min_level = min(min_level, current_level_id)
            
            new_node = {
                "id": node_id,
                "content": current_content,
                "content_raw": current_content_raw,
                "type": level.get("type"),
                "number": level.get("number"),
                "level": current_level_id,
                "children": [],
                "is_edit": current_node.get("is_edit"),
                "document_code": document_code
            }
            
            # If the node is at the minimal level, treat it as a root.
            if current_level_id == min_level:
                root_nodes.append(new_node)
                stack = [new_node]
            else:
                # Pop nodes until we find the parent.
                while stack and stack[-1].get("level") >= current_level_id:
                    stack.pop()
                if stack:
                    stack[-1]["children"].append(new_node)
                else:
                    # If no valid parent is found, treat as a root.
                    root_nodes.append(new_node)
                stack.append(new_node)
        return root_nodes

    def build_closure_table(self, tree: list) -> list:
        """
        Builds a closure table from the nested JSON tree.
        The closure table is a list of dicts with 'ancestor', 'descendant', and 'depth'.
        Each node is related to itself (depth 0) and to each of its ancestors.
        """
        closure_table = []

        def traverse(node, ancestors):
            # Record each relationship: each ancestor with depth, plus self-relation.
            for depth, ancestor in enumerate(ancestors, start=1):
                closure_table.append({
                    "ancestor": ancestor,
                    "descendant": node["id"],
                    "depth": depth
                })
            closure_table.append({
                "ancestor": node["id"],
                "descendant": node["id"],
                "depth": 0
            })
            # Process children recursively, extending the ancestor chain.
            for child in node.get("children", []):
                traverse(child, ancestors + [node["id"]])
        
        # Start traversal at each root node.
        for root in tree:
            traverse(root, [])
        return closure_table
    
    def build_adjacency_list(self, tree: list) -> list:
        """
        Builds an adjacency list from the nested JSON tree.
        The adjacency list is a list of dicts with 'parent' and 'child'.
        Each node is related to its parent, if any.
        return also mapping id to node's information
        """
        adjacency_list = []

        def traverse(node, parent=None):
            list_child_id = []
            for child in node.get("children", []):
                list_child_id.append(child["id"])
            if parent:
                adjacency_list.append({
                    "parent": parent,
                    "children": list_child_id
                })
            for child in node.get("children", []):
                traverse(child, node["id"])
        # print()
        # Start traversal at each root node.
        for root in tree:
            # print(root)
            traverse(root)
            
        return adjacency_list
