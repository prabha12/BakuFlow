class BoundingBox:
    def __init__(self, x, y, w, h, label=""):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.label = label

    def get_resize_handle(self, point, threshold=10):
        """Enhanced handle detection with visual size consideration"""
        handles = {
            'top-left': (self.x, self.y),
            'top-right': (self.x + self.w, self.y),
            'bottom-left': (self.x, self.y + self.h),
            'bottom-right': (self.x + self.w, self.y + self.h)
        }
        for handle_name, (hx, hy) in handles.items():
            if (abs(point.x() - hx) < threshold and
                abs(point.y() - hy) < threshold):
                return handle_name
        return None
        
    def contains(self, point, buffer=5):
        """Check if point is inside bbox with optional buffer zone"""
        # First check if point is fully inside the box
        if (self.x + buffer <= point.x() <= self.x + self.w - buffer and
            self.y + buffer <= point.y() <= self.y + self.h - buffer):
            return True
        # Then check if point is in buffer zone near edges
        return (self.x - buffer <= point.x() <= self.x + self.w + buffer and
                self.y - buffer <= point.y() <= self.y + self.h + buffer)
    
    def get_resize_handle(self, point, threshold=8): # Duplicated, will keep the first one.
        handles = {
            'top-left': (self.x, self.y),
            'top-right': (self.x + self.w, self.y),
            'bottom-left': (self.x, self.y + self.h),
            'bottom-right': (self.x + self.w, self.y + self.h)
        }
        for handle_name, (hx, hy) in handles.items():
            if abs(point.x() - hx) < threshold and abs(point.y() - hy) < threshold:
                return handle_name
        return None 