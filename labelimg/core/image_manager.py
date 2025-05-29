import os
import cv2
from typing import List, Optional

class ImageManager:
    def __init__(self, image_dir: Optional[str] = None, valid_exts=None):
        self.image_dir = image_dir
        self.valid_exts = valid_exts or ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
        self.image_list: List[str] = []
        self.current_index: int = -1
        self.current_image = None
        self.thumbnail_cache = {}
        self.preload_window = 2  # 預加載前後各2張
        if image_dir:
            self.scan_images(image_dir)

    def scan_images(self, image_dir: str):
        self.image_dir = image_dir
        self.image_list = []
        for filename in sorted(os.listdir(image_dir)):
            if filename.lower().endswith(self.valid_exts):
                img_path = os.path.join(image_dir, filename)
                # 輕量檢查
                if cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) is not None:
                    self.image_list.append(img_path)
        self.current_index = 0 if self.image_list else -1
        self.current_image = None
        self.thumbnail_cache = {}

    def get_image(self, idx: int):
        if 0 <= idx < len(self.image_list):
            img_path = self.image_list[idx]
            img = cv2.imread(img_path)
            return img
        return None

    def get_current_image(self):
        if self.current_index >= 0 and self.current_index < len(self.image_list):
            if self.current_image is None:
                self.current_image = self.get_image(self.current_index)
            return self.current_image
        return None

    def goto(self, idx: int):
        if 0 <= idx < len(self.image_list):
            self.current_index = idx
            self.current_image = self.get_image(idx)
            self.preload_images(idx)
            return True
        return False

    def next_image(self):
        return self.goto(self.current_index + 1)

    def prev_image(self):
        return self.goto(self.current_index - 1)

    def get_thumbnail(self, idx: int, size=(128, 128)):
        if idx in self.thumbnail_cache:
            return self.thumbnail_cache[idx]
        img = self.get_image(idx)
        if img is not None:
            thumb = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            self.thumbnail_cache[idx] = thumb
            return thumb
        return None

    def preload_images(self, center_idx: int):
        # 預加載前後幾張圖片到cache
        for i in range(center_idx - self.preload_window, center_idx + self.preload_window + 1):
            if 0 <= i < len(self.image_list) and i not in self.thumbnail_cache:
                img = self.get_image(i)
                if img is not None:
                    thumb = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    self.thumbnail_cache[i] = thumb

    def total(self):
        return len(self.image_list)

    def current_path(self):
        if 0 <= self.current_index < len(self.image_list):
            return self.image_list[self.current_index]
        return None

    def all_filenames(self):
        return [os.path.basename(p) for p in self.image_list] 