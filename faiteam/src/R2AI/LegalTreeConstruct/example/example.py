from LegalTreeConstruct.document_constructor import DocumentConstructor
import json

if __name__ == "__main__":
    with open("./example/sample_data.json", "r") as f:
        data = json.load(f)
    document_constructor = DocumentConstructor()
    # Check if the paragraph is an edit article 
    list_edit_paragraph = document_constructor.pre_process(data)
    # print(list_edit_paragraph)
    # Construct the nodes before building the tree
    nodes = document_constructor.node_constructor(data, "example")
    
    # Construct the tree
    tree = document_constructor.construct_tree(nodes)
    print(tree)
    
    # # Construct the closure table
    closure_table = document_constructor.construct_closure_table(nodes)
    print(closure_table)
    
    # Construct the adjacency list
    adjacency_list = document_constructor.construct_adjacency_list(nodes)
    print(adjacency_list)