import threading

class DocRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self.mapping = {}

    def update(self, new_mapping: dict):
        with self._lock:
            self.mapping.update(new_mapping)

    def get(self, key: str):
        with self._lock:
            return self.mapping.get(key)
            
    def get_all(self):
        with self._lock:
            return self.mapping.copy()

    def clear(self):
        with self._lock:
            self.mapping.clear()

global_doc_registry = DocRegistry()
