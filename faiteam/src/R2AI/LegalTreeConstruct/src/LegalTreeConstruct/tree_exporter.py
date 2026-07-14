import json

class TreeExporter:
    # def __init__(self, output_dir="./"):
    #     self.output_dir = output_dir
        
    @staticmethod
    def save_nodes( tree_data: dict, save_path: str):
        nodes_data = []
        mapping_id_content = tree_data.get("mapping_id_content")
        
        def flatten_tree(nodes, parent_name=None):
            for node in nodes:
                node_id = node.get("id")
                node_content = mapping_id_content.get(node_id)
                nodes_data.append({"id": node_id, "content": node_content, "parent": parent_name})
                if node.get("children"):
                    parent_name = node_content
                flatten_tree(node.get("children"), parent_name)
        
        if tree_data and tree_data.get("tree"):
            flatten_tree(tree_data.get("tree"))

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, indent=4, ensure_ascii=False)
        # print(f"Nodes data saved to: {save_path}")

    @staticmethod
    def save_edges( tree_data: dict, save_path: str):
        edges_data = []
        
        def traverse_tree_for_edges(nodes, parent_id=None):
            for node in nodes:
                current_id = node.get("id")
                if parent_id:
                    edges_data.append({"source": parent_id, "target": current_id})
                traverse_tree_for_edges(node.get("children"), current_id)
        
        if tree_data and tree_data.get("tree"):
            traverse_tree_for_edges(tree_data.get("tree"))
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(edges_data, f, indent=4, ensure_ascii=False)