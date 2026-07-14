# Legal Document-Tree Construct Package

This is LegalTreeConstruct package construct list of paragraphs that extract from document into 3 format.

- Tree Nested Json 

- Closure Table

- Adjacency List

You also can check which paragraph is start of edit article.


## Install Locally

After git clone, you can access the codebase and simply run the following command line:

```bash
cd LegalTreeConstruct
pip install -e .
```

## Example Usage

After installing, You can view ```examples/example.py``` to get usage of LegalTreeConstruct.

The Input data must be same format like file ```example/sample_data.json```.


You can run ```example/example.py``` file as following:

```bash
python examples/sample.py
```

## Basic Usage

You can import DocumentConstructor from LegalTreeConstruct.

```python
from LegalTreeConstruct.document_constructor import DocumentConstructor
import json

if __name__ == "__main__":
    with open("./example/sample_data.json", "r") as f:
        data = json.load(f)
    document_constructor = DocumentConstructor()

    # Check if the paragraph is an edit article 
    # if the is no edit article then list is []
    list_edit_paragraph = document_constructor.pre_process(data)
    print(list_edit_paragraph)

    # Construct the nodes before building the tree
    nodes = document_constructor.node_constructor(data, "example")
    
    # Construct the tree
    tree = document_constructor.construct_tree(nodes)
    print(tree)
    
    # Construct the closure table
    closure_table = document_constructor.construct_closure_table(nodes)
    print(closure_table)
    
    # Construct the adjacency list
    adjacency_list = document_constructor.construct_adjacency_list(nodes)
    print(adjacency_list)
```