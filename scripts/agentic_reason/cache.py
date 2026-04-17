import os
import json

class CacheManager:
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        self.search_cache_path = os.path.join(cache_dir, 'search_cache.json')
        self.url_cache_path = os.path.join(cache_dir, 'url_cache.json')
        self._initialize_cache()
    
    def _initialize_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self.search_cache = self._load_cache(self.search_cache_path)
        self.url_cache = self._load_cache(self.url_cache_path)
    
    def _load_cache(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_caches(self):
        with open(self.search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
        with open(self.url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.url_cache, f, ensure_ascii=False, indent=2)