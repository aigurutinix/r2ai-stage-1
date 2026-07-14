import tiktoken

class TokenCounter:
    def __init__(self, model_name="gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model_name)
    
    def count_tokens(self, text):
        tokens = self.encoding.encode(text)
        return len(tokens)
    
    def calculate_tokens_recursive(self, node):
        if 'children' not in node or not node['children']:
            node['num_tokens'] = self.count_tokens(node.get('content', ''))
            return node
        
        total_tokens = 0
        for child in node['children']:
            child_node = self.calculate_tokens_recursive(child)
            total_tokens += child_node['num_tokens']
        
        node['num_tokens'] = total_tokens + self.count_tokens(node.get('content', ''))
        return node
