from PyQt5.QtWidgets import QWidget, QDialog, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QSpinBox, QCheckBox, QLineEdit, QPushButton, QDoubleSpinBox, QProgressDialog, QMessageBox, QListWidget, QListWidgetItem, QSlider, QGridLayout
from PyQt5.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QPen, QColor
import cv2
import numpy as np
import os
import random # For DataAugmentationDialog
from PyQt5.QtWidgets import QFileDialog # For DataAugmentationDialog
from ..core.localization import tr
import shutil

# 导出的类列表
__all__ = ['MagnifierWindow', 'DataAugmentationDialog', 'AutoLabelDialog', 'AutoLabelProgressDialog']

class MagnifierWindow(QWidget):
    # 添加信号
    zoom_changed = pyqtSignal(float)  # 用于通知倍率变化
    
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.magnifier_size = 200
        self.zoom_factor = 2.0  # 默认倍率
        self.setFixedSize(self.magnifier_size, self.magnifier_size)
        
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 180);
                border: 2px solid #666;
                border-radius: 100px;
            }
        """)
        
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.updatePosition)
        self.update_timer.start(16)
        
    def set_zoom_factor(self, factor):
        self.zoom_factor = factor
        self.update()
        
    def updatePosition(self):
        if self.parent():
            from PyQt5.QtWidgets import QApplication # Moved import here
            cursor_pos = QCursor.pos()
            screen = QApplication.primaryScreen().geometry()
            x = cursor_pos.x() + 20
            y = cursor_pos.y() + 20
            
            if x + self.magnifier_size > screen.right():
                x = cursor_pos.x() - self.magnifier_size - 20
            if y + self.magnifier_size > screen.bottom():
                y = cursor_pos.y() - self.magnifier_size - 20
                
            self.move(x, y)
            self.update() 
            
    def paintEvent(self, event):
        if not self.parent():
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        parent = self.parent()
        if not parent.current_image is None:
            cursor_pos = QCursor.pos()
            image_pos = parent.image_label.mapFromGlobal(cursor_pos)
            scaled_pos = parent.getScaledPoint(image_pos)
            
            source_size = int(self.magnifier_size / self.zoom_factor)
            source_x = max(0, int(scaled_pos.x() - source_size // 2))
            source_y = max(0, int(scaled_pos.y() - source_size // 2))
            
            h_img, w_img = parent.current_image.shape[:2]
            source_x = min(source_x, w_img - source_size)
            source_y = min(source_y, h_img - source_size)
            
            roi = parent.current_image[source_y:source_y + source_size, 
                                     source_x:source_x + source_size]
            if roi.size > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                q_img = QImage(roi_rgb.data, roi_rgb.shape[1], roi_rgb.shape[0],
                             roi_rgb.strides[0], QImage.Format_RGB888)
                
                painter.drawImage(self.rect(), q_img)
                
                painter.setPen(QPen(QColor(255, 0, 0), 1))
                center = self.rect().center()
                painter.drawLine(center.x(), 0, center.x(), self.height())
                painter.drawLine(0, center.y(), self.width(), center.y())
                
                if parent.drawing and parent.bbox_start and parent.bbox_end:
                    x1_draw = int((parent.bbox_start.x() - source_x) * self.zoom_factor)
                    y1_draw = int((parent.bbox_start.y() - source_y) * self.zoom_factor)
                    x2_draw = int((parent.bbox_end.x() - source_x) * self.zoom_factor)
                    y2_draw = int((parent.bbox_end.y() - source_y) * self.zoom_factor)
                    
                    pen_draw = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
                    painter.setPen(pen_draw)
                    painter.drawRect(min(x1_draw, x2_draw), min(y1_draw, y2_draw), abs(x2_draw - x1_draw), abs(y2_draw - y1_draw))
                
                for bbox in parent.bboxes:
                    x_bbox = int((bbox.x - source_x) * self.zoom_factor)
                    y_bbox = int((bbox.y - source_y) * self.zoom_factor)
                    w_bbox = int(bbox.w * self.zoom_factor)
                    h_bbox = int(bbox.h * self.zoom_factor)
                    
                    if (x_bbox + w_bbox > 0 and x_bbox < self.magnifier_size and 
                        y_bbox + h_bbox > 0 and y_bbox < self.magnifier_size):
                        color = parent.get_label_color(bbox.label)
                        pen_bbox = QPen(QColor(*color), 2)
                        if bbox == parent.selected_bbox:
                            pen_bbox.setStyle(Qt.DashLine)
                        painter.setPen(pen_bbox)
                        painter.drawRect(x_bbox, y_bbox, w_bbox, h_bbox)
                        
                        if bbox == parent.selected_bbox:
                            handle_size_bbox = int(parent.handle_visual_size * self.zoom_factor)
                            for handle in [(x_bbox, y_bbox), (x_bbox + w_bbox, y_bbox), (x_bbox, y_bbox + h_bbox), (x_bbox + w_bbox, y_bbox + h_bbox)]:
                                painter.fillRect(
                                    int(handle[0] - handle_size_bbox/2),
                                    int(handle[1] - handle_size_bbox/2),
                                    handle_size_bbox,
                                    handle_size_bbox,
                                    QColor(0, 255, 255)
                                )

class DataAugmentationWorker(QThread):
    progress = pyqtSignal(int, int)  # (current, total)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, image_list, classes, output_folder, aug_count, preserve_original, get_random_params, apply_augmentation):
        super().__init__()
        self.image_list = image_list
        self.classes = classes
        self.output_folder = output_folder
        self.aug_count = aug_count
        self.preserve_original = preserve_original
        self.get_random_params = get_random_params
        self.apply_augmentation = apply_augmentation

    def run(self):
        try:
            total_augmented = 0
            for i, image_path in enumerate(self.image_list):
                self.progress.emit(i + 1, len(self.image_list))
                import cv2, os
                image = cv2.imread(image_path)
                if image is None:
                    continue
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                label_path = os.path.splitext(image_path)[0] + '.txt'
                current_bboxes = []
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_idx = int(parts[0])
                                x_center, y_center, w_norm, h_norm = map(float, parts[1:])
                                img_h, img_w = image.shape[:2]
                                x = int((x_center - w_norm/2) * img_w)
                                y = int((y_center - h_norm/2) * img_h)
                                w_pixels = int(w_norm * img_w)
                                h_pixels = int(h_norm * img_h)
                                label = self.classes[class_idx] if class_idx < len(self.classes) else "unknown"
                                from ..core.bounding_box import BoundingBox
                                current_bboxes.append(BoundingBox(x, y, w_pixels, h_pixels, label))
                for aug_idx in range(self.aug_count):
                    params = self.get_random_params()
                    bboxes_to_augment = [b for b in current_bboxes]
                    aug_image, aug_bboxes = self.apply_augmentation(image.copy(), bboxes_to_augment, params)
                    aug_suffix = f"_aug{aug_idx + 1}"
                    aug_image_path = os.path.join(self.output_folder, base_name + aug_suffix + os.path.splitext(image_path)[1])
                    aug_label_path = os.path.join(self.output_folder, base_name + aug_suffix + '.txt')
                    cv2.imwrite(aug_image_path, aug_image)
                    if aug_bboxes:
                        aug_img_h, aug_img_w = aug_image.shape[:2]
                        with open(aug_label_path, 'w') as f_aug:
                            for bbox in aug_bboxes:
                                x_c = (bbox.x + bbox.w/2) / aug_img_w
                                y_c = (bbox.y + bbox.h/2) / aug_img_h
                                w_n = bbox.w / aug_img_w
                                h_n = bbox.h / aug_img_h
                                try:
                                    class_idx_aug = self.classes.index(bbox.label)
                                    f_aug.write(f"{class_idx_aug} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                                except ValueError:
                                    continue
                    total_augmented += 1
                if self.preserve_original:
                    dst_img = os.path.join(self.output_folder, os.path.basename(image_path))
                    if os.path.abspath(image_path) != os.path.abspath(dst_img):
                        shutil.copy2(image_path, dst_img)
                    if os.path.exists(label_path):
                        dst_label = os.path.join(self.output_folder, os.path.basename(label_path))
                        if os.path.abspath(label_path) != os.path.abspath(dst_label):
                            shutil.copy2(label_path, dst_label)
            self.finished.emit(total_augmented)
        except Exception as e:
            self.error.emit(str(e))

class DataAugmentationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent # Renamed to avoid clash with QWidget.parent()
        self.setWindowTitle(tr("data_aug_title"))
        self.setModal(True)
        if hasattr(self.parent_window, 'current_dir') and self.parent_window.current_dir:
            self.output_folder = self.parent_window.current_dir
        else:
            self.output_folder = ""
        self.aug_per_image = 1
        self.preserve_original = True
        self.rotation_range = (-30, 30)
        self.brightness_range = (-0.2, 0.2)
        self.contrast_range = (0.8, 1.2)
        self.saturation_range = (0.8, 1.2)
        self.initUI()
        self.progress_dialog = None
        self.worker = None
        
    def initUI(self):
        layout = QVBoxLayout()
        
        basic_group = QGroupBox(tr("aug_settings"))
        basic_layout = QVBoxLayout()
        
        aug_count_layout = QHBoxLayout()
        aug_count_label = QLabel(tr("aug_per_image"))
        self.aug_count_spin = QSpinBox()
        self.aug_count_spin.setRange(1, 99999)
        self.aug_count_spin.setValue(1)
        aug_count_layout.addWidget(aug_count_label)
        aug_count_layout.addWidget(self.aug_count_spin)
        basic_layout.addLayout(aug_count_layout)
        
        self.preserve_checkbox = QCheckBox(tr("preserve_original"))
        self.preserve_checkbox.setChecked(True)
        basic_layout.addWidget(self.preserve_checkbox)
        
        folder_layout = QHBoxLayout()
        folder_label = QLabel(tr("output_folder"))
        self.folder_path = QLineEdit()
        self.folder_path.setReadOnly(True)
        self.folder_path.setText(self.output_folder)
        folder_btn = QPushButton(tr("select_folder"))
        folder_btn.clicked.connect(self.select_output_folder)
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_path)
        folder_layout.addWidget(folder_btn)
        basic_layout.addLayout(folder_layout)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        random_group = QGroupBox(tr("aug_settings"))
        random_layout = QVBoxLayout()
        
        # 使用网格布局来确保对齐
        grid_layout = QGridLayout()
        
        # 设置列宽度比例，确保对齐
        grid_layout.setColumnStretch(0, 2)  # 标签列
        grid_layout.setColumnStretch(1, 1)  # 最小值列
        grid_layout.setColumnStretch(2, 0)  # "to"列
        grid_layout.setColumnStretch(3, 1)  # 最大值列
        
        # 旋转范围
        rotation_label = QLabel(tr("random_rotation"))
        self.rotation_min = QSpinBox()
        self.rotation_min.setRange(-180, 180)
        self.rotation_min.setValue(-30)
        self.rotation_max = QSpinBox()
        self.rotation_max.setRange(-180, 180)
        self.rotation_max.setValue(30)
        to_label1 = QLabel("to")
        to_label1.setAlignment(Qt.AlignCenter)
        
        grid_layout.addWidget(rotation_label, 0, 0)
        grid_layout.addWidget(self.rotation_min, 0, 1)
        grid_layout.addWidget(to_label1, 0, 2)
        grid_layout.addWidget(self.rotation_max, 0, 3)
        
        # 亮度范围
        brightness_label = QLabel(tr("random_brightness"))
        self.brightness_min = QDoubleSpinBox()
        self.brightness_min.setRange(-1.0, 1.0)
        self.brightness_min.setValue(-0.2)
        self.brightness_min.setSingleStep(0.1)
        self.brightness_max = QDoubleSpinBox()
        self.brightness_max.setRange(-1.0, 1.0)
        self.brightness_max.setValue(0.2)
        self.brightness_max.setSingleStep(0.1)
        to_label2 = QLabel("to")
        to_label2.setAlignment(Qt.AlignCenter)
        
        grid_layout.addWidget(brightness_label, 1, 0)
        grid_layout.addWidget(self.brightness_min, 1, 1)
        grid_layout.addWidget(to_label2, 1, 2)
        grid_layout.addWidget(self.brightness_max, 1, 3)
        
        # 对比度范围
        contrast_label = QLabel(tr("random_contrast"))
        self.contrast_min = QDoubleSpinBox()
        self.contrast_min.setRange(0.1, 2.0)
        self.contrast_min.setValue(0.8)
        self.contrast_min.setSingleStep(0.1)
        self.contrast_max = QDoubleSpinBox()
        self.contrast_max.setRange(0.1, 2.0)
        self.contrast_max.setValue(1.2)
        self.contrast_max.setSingleStep(0.1)
        to_label3 = QLabel("to")
        to_label3.setAlignment(Qt.AlignCenter)
        
        grid_layout.addWidget(contrast_label, 2, 0)
        grid_layout.addWidget(self.contrast_min, 2, 1)
        grid_layout.addWidget(to_label3, 2, 2)
        grid_layout.addWidget(self.contrast_max, 2, 3)
        
        # 饱和度范围
        saturation_label = QLabel(tr("random_saturation"))
        self.saturation_min = QDoubleSpinBox()
        self.saturation_min.setRange(0.1, 2.0)
        self.saturation_min.setValue(0.8)
        self.saturation_min.setSingleStep(0.1)
        self.saturation_max = QDoubleSpinBox()
        self.saturation_max.setRange(0.1, 2.0)
        self.saturation_max.setValue(1.2)
        self.saturation_max.setSingleStep(0.1)
        to_label4 = QLabel("to")
        to_label4.setAlignment(Qt.AlignCenter)
        
        grid_layout.addWidget(saturation_label, 3, 0)
        grid_layout.addWidget(self.saturation_min, 3, 1)
        grid_layout.addWidget(to_label4, 3, 2)
        grid_layout.addWidget(self.saturation_max, 3, 3)
        
        random_layout.addLayout(grid_layout)
        random_group.setLayout(random_layout)
        layout.addWidget(random_group)
        
        flip_group = QGroupBox(tr("flip_options"))
        flip_layout = QHBoxLayout()
        self.flip_ud_checkbox = QCheckBox(tr("flip_ud"))
        self.flip_lr_checkbox = QCheckBox(tr("flip_lr"))
        flip_layout.addWidget(self.flip_ud_checkbox)
        flip_layout.addWidget(self.flip_lr_checkbox)
        flip_group.setLayout(flip_layout)
        layout.addWidget(flip_group)
        
        button_layout = QHBoxLayout()
        apply_btn = QPushButton(tr("apply"))
        cancel_btn = QPushButton(tr("cancel"))
        button_layout.addWidget(apply_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        apply_btn.clicked.connect(self.process_images)
        cancel_btn.clicked.connect(self.reject)
        
        self.setLayout(layout)
        
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr("select_folder"))
        if folder:
            self.output_folder = folder
            self.folder_path.setText(folder)
            
    def get_random_params(self):
        return {
            'rotation': random.uniform(self.rotation_min.value(), self.rotation_max.value()),
            'brightness': random.uniform(self.brightness_min.value(), self.brightness_max.value()),
            'contrast': random.uniform(self.contrast_min.value(), self.contrast_max.value()),
            'saturation': random.uniform(self.saturation_min.value(), self.saturation_max.value()),
            'flip_ud': random.choice([True, False]) if self.flip_ud_checkbox.isChecked() else False,
            'flip_lr': random.choice([True, False]) if self.flip_lr_checkbox.isChecked() else False
        }
        
    def apply_augmentation(self, image, bboxes_in, params):
        # Ensure a deep copy of bboxes is made if they are to be modified
        # from ..core.bounding_box import BoundingBox # Placeholder for eventual import
        # bboxes = [BoundingBox(b.x,b.y,b.w,b.h,b.label) for b in bboxes_in]
        # For now, assume bboxes_in is a list of objects that can be directly modified or are value types
        bboxes = []
        for b_orig in bboxes_in:
            # Create a new BoundingBox object for each original to avoid modifying originals in place
            # This requires BoundingBox to be defined or imported.
            # For now, this will fail if BoundingBox is not accessible.
            # Placeholder: if BoundingBox is not defined, this will cause an error.
            # Assuming BoundingBox is defined as in the original single file for now.
            try: 
                from ..core.bounding_box import BoundingBox
            except ImportError:
                 # Fallback for direct execution or if structure isn't fully set up
                 # This is a temporary measure for the refactoring process.
                 class BoundingBoxPlaceholder:
                    def __init__(self, x, y, w, h, label=""):
                        self.x = int(x); self.y = int(y); self.w = int(w); self.h = int(h); self.label = label
                 BoundingBox = BoundingBoxPlaceholder
            bboxes.append(BoundingBox(b_orig.x, b_orig.y, b_orig.w, b_orig.h, b_orig.label))


        height, width = image.shape[:2]
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], params['saturation'])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        image = cv2.convertScaleAbs(image, alpha=params['contrast'], beta=params['brightness'] * 255)
        
        if params['rotation'] != 0:
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, params['rotation'], 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            for bbox in bboxes:
                corners = np.array([
                    [bbox.x, bbox.y],
                    [bbox.x + bbox.w, bbox.y],
                    [bbox.x + bbox.w, bbox.y + bbox.h],
                    [bbox.x, bbox.y + bbox.h]
                ])
                corners = cv2.transform(corners.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
                x_min = np.min(corners[:, 0])
                y_min = np.min(corners[:, 1])
                x_max = np.max(corners[:, 0])
                y_max = np.max(corners[:, 1])
                bbox.x = max(0, int(x_min))
                bbox.y = max(0, int(y_min))
                bbox.w = min(width - bbox.x, int(x_max - x_min))
                bbox.h = min(height - bbox.y, int(y_max - y_min))
        
        if params['flip_ud']:
            image = cv2.flip(image, 0)
            for bbox in bboxes:
                bbox.y = height - (bbox.y + bbox.h)
                
        if params['flip_lr']:
            image = cv2.flip(image, 1)
            for bbox in bboxes:
                bbox.x = width - (bbox.x + bbox.w)
                
        return image, bboxes
        
    def process_images(self):
        if not self.output_folder:
            QMessageBox.warning(self, "Error", tr("select_folder"))
            return
        if not self.parent_window or not self.parent_window.image_list:
            return
        self.progress_dialog = QProgressDialog(tr("processing"), tr("cancel"), 0, len(self.parent_window.image_list), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)
        self.worker = DataAugmentationWorker(
            image_list=self.parent_window.image_list,
            classes=self.parent_window.classes,
            output_folder=self.output_folder,
            aug_count=self.aug_count_spin.value(),
            preserve_original=self.preserve_checkbox.isChecked(),
            get_random_params=self.get_random_params,
            apply_augmentation=self.apply_augmentation
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.progress_dialog.canceled.connect(self.worker.terminate)
        self.worker.start()

    def on_progress(self, current, total):
        if self.progress_dialog:
            self.progress_dialog.setValue(current)
            self.progress_dialog.setLabelText(tr("aug_progress").format(current, total))

    def on_finished(self, total_augmented):
        if self.progress_dialog:
            self.progress_dialog.setValue(self.progress_dialog.maximum())
            self.progress_dialog.close()
        QMessageBox.information(self, tr("aug_complete"), tr("aug_success").format(total_augmented))
        if hasattr(self.parent_window, 'current_dir') and os.path.abspath(self.output_folder) == os.path.abspath(self.parent_window.current_dir):
            if hasattr(self.parent_window, 'openDirectory'):
                self.parent_window.openDirectory()
        self.accept()

    def on_error(self, msg):
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, tr("aug_error"), msg)

class AutoLabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(tr("auto_label_title"))
        self.setModal(True)
        self.prompt_indices = []  # 存储选中的prompt图片索引
        
        # 确保在显示对话框前同步最新的项目状态
        self.refresh_project_data()
        self.initUI()
        
    def refresh_project_data(self):
        """刷新项目数据，确保状态同步"""
        if not self.parent_window:
            return
            
        print("[Debug] 刷新自动标注对话框的项目数据...")
        
        # 重新扫描并同步标注缓存
        self.parent_window.label_cache = {}
        
        for idx, img_path in enumerate(self.parent_window.image_list):
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(txt_path):
                # 创建临时状态来加载标注
                original_current_index = self.parent_window.current_index
                original_bboxes = self.parent_window.bboxes.copy()
                original_current_image = self.parent_window.current_image
                
                try:
                    # 临时设置状态
                    self.parent_window.current_index = idx
                    self.parent_window.current_image = cv2.imread(img_path)
                    
                    if self.parent_window.current_image is not None:
                        self.parent_window.loadAnnotations(txt_path)
                        self.parent_window.label_cache[idx] = [
                            BoundingBox(b.x, b.y, b.w, b.h, b.label) 
                            for b in self.parent_window.bboxes
                        ]
                        print(f"[Debug] 缓存图片 {idx} 的 {len(self.parent_window.bboxes)} 个标注")
                    
                finally:
                    # 恢复原始状态
                    self.parent_window.current_index = original_current_index
                    self.parent_window.bboxes = original_bboxes
                    self.parent_window.current_image = original_current_image
        
        print(f"[Debug] 项目数据刷新完成，找到 {len(self.parent_window.label_cache)} 个已标注图片")
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 添加刷新按钮
        refresh_layout = QHBoxLayout()
        refresh_btn = QPushButton(tr("refresh_data"))
        refresh_btn.clicked.connect(self.refresh_and_update_ui)
        refresh_btn.setToolTip(tr("refresh_data_tooltip"))
        refresh_layout.addWidget(refresh_btn)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)
        
        # 显示项目信息
        info_group = QGroupBox(tr("project_info"))
        info_layout = QVBoxLayout()
        
        if hasattr(self.parent_window, 'current_dir') and self.parent_window.current_dir:
            project_name = os.path.basename(self.parent_window.current_dir)
            info_layout.addWidget(QLabel(f"{tr('current_project')}: {project_name}"))
        
        total_images = len(self.parent_window.image_list) if self.parent_window.image_list else 0
        labeled_count = len(self.parent_window.label_cache)
        info_layout.addWidget(QLabel(f"{tr('total_images')}: {total_images}"))
        info_layout.addWidget(QLabel(f"{tr('labeled_images')}: {labeled_count}"))
        info_layout.addWidget(QLabel(f"{tr('unlabeled_images')}: {total_images - labeled_count}"))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Prompt Images Section
        prompt_group = QGroupBox(tr("select_prompt_images"))
        prompt_layout = QVBoxLayout()
        
        self.prompt_list = QListWidget()
        self.update_prompt_list()
        self.prompt_list.setSelectionMode(QListWidget.MultiSelection)
        prompt_layout.addWidget(self.prompt_list)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # 批量自动标注按钮
        self.auto_label_all_btn = QPushButton(tr("auto_label_all_unlabeled"))
        self.auto_label_all_btn.clicked.connect(self.auto_label_all)
        layout.addWidget(self.auto_label_all_btn)

        # 取消按钮
        cancel_btn = QPushButton(tr("cancel"))
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)
        
        self.setLayout(layout)
        
    def update_prompt_list(self):
        """更新提示图片列表"""
        self.prompt_list.clear()
        
        # 重新计算已标注的图片索引
        self.labeled_indices = list(self.parent_window.label_cache.keys())
        self.labeled_indices.sort()  # 确保顺序一致
        
        if not self.labeled_indices:
            item = QListWidgetItem(tr("no_labeled_images_found"))
            item.setForeground(QColor(128, 128, 128))  # 灰色
            self.prompt_list.addItem(item)
            self.auto_label_all_btn.setEnabled(False)
            return
        
        # 添加已标注的图片到列表
        for idx in self.labeled_indices:
            if idx < len(self.parent_window.image_list):
                filename = os.path.basename(self.parent_window.image_list[idx])
                bbox_count = len(self.parent_window.label_cache[idx])
                
                item = QListWidgetItem(f"{filename} ({bbox_count} {tr('objects')})")
                item.setForeground(QColor(0, 128, 0))  # 绿色表示已标注
                self.prompt_list.addItem(item)
        
        self.auto_label_all_btn.setEnabled(len(self.labeled_indices) > 0)
        
    def refresh_and_update_ui(self):
        """刷新数据并更新UI"""
        self.refresh_project_data()
        self.update_prompt_list()
        
        # 显示刷新结果
        labeled_count = len(self.parent_window.label_cache)
        QMessageBox.information(
            self, 
            tr("refresh_complete"), 
            tr("refresh_complete_msg").format(labeled_count)
        )

    def auto_label_all(self):
        # 1. 強制刷新 classes.txt 和 label_cache
        self.parent_window.loadClasses()
        self.parent_window.label_cache = {}
        for idx, img_path in enumerate(self.parent_window.image_list):
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(txt_path):
                self.parent_window.loadAnnotations(txt_path)
                self.parent_window.label_cache[idx] = self.parent_window.bboxes.copy()

        prompt_image_paths, visuals, prompt_indices = self.get_prompt_data()
        if not prompt_image_paths:
            QMessageBox.warning(self, tr("warning"), tr("select_prompt_images_first"))
            return
        # 2. 檢查 prompt 是否出現在 target list
        prompt_indices_set = set(prompt_indices)

        # 3. 允許用戶選擇是否覆蓋已標註圖片
        reply = QMessageBox.question(self, tr("overwrite_labeled_images"), tr("overwrite_labeled_images_question"), QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            target_indices = [i for i in range(len(self.parent_window.image_list)) if i not in prompt_indices_set]
        else:
            target_indices = [i for i in range(len(self.parent_window.image_list))
                              if i not in self.parent_window.label_cache and i not in prompt_indices_set]
        if not target_indices:
            QMessageBox.information(self, tr("auto_label_title"), tr("all_images_labeled"))
            return
        target_image_paths = [self.parent_window.image_list[i] for i in target_indices]
        # 調用主窗口的批量自動標註方法
        try:
            self.parent_window.handle_batch_auto_label_with_vp(prompt_image_paths, visuals, target_image_paths, target_indices)
            QMessageBox.information(self, tr("auto_label_title"), tr("auto_label_success"))
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, tr("warning"), f"{tr('auto_label_error')}: {str(e)}") 

    def get_prompt_data(self):
        """获取提示图片数据"""
        prompt_image_paths = []
        prompt_indices = []
        bboxes_list = []
        cls_list = []
        
        print("[Debug] 开始获取提示图片数据...")
        
        # 修正：正確 mapping prompt_list row 到 image_list index
        for item in self.prompt_list.selectedItems():
            row = self.prompt_list.row(item)
            if row < len(self.labeled_indices):
                img_idx = self.labeled_indices[row]
                prompt_image_paths.append(self.parent_window.image_list[img_idx])
                prompt_indices.append(img_idx)
                print(f"[Debug] 添加提示图片: {self.parent_window.image_list[img_idx]}")
        
        if not prompt_image_paths:
            print("[Debug] 没有选择提示图片")
            return [], [], []
        
        # 🔍 修复：直接从YOLO标注文件读取真实的类别ID，而不是使用GUI中的索引
        for img_idx in prompt_indices:
            bboxes = []
            clses = []
            
            # 直接读取对应的YOLO标注文件
            img_path = self.parent_window.image_list[img_idx]
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            
            print(f"[Debug] 读取标注文件: {txt_path}")
            
            if os.path.exists(txt_path):
                try:
                    # 获取图片尺寸用于坐标转换
                    import cv2
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        
                        with open(txt_path, 'r') as f:
                            for line_num, line in enumerate(f):
                                line = line.strip()
                                if not line:
                                    continue
                                
                                parts = line.split()
                                if len(parts) != 5:
                                    print(f"[Debug] 跳过格式错误的行 {line_num+1}: {line}")
                                    continue
                                
                                try:
                                    # 🔍 关键修复：直接使用YOLO文件中的真实类别ID
                                    real_class_id = int(parts[0])
                                    center_x = float(parts[1])
                                    center_y = float(parts[2])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    
                                    # 转换为绝对坐标
                                    x1 = (center_x - width/2) * img_width
                                    y1 = (center_y - height/2) * img_height
                                    x2 = (center_x + width/2) * img_width
                                    y2 = (center_y + height/2) * img_height
                                    
                                    bboxes.append([x1, y1, x2, y2])
                                    clses.append(real_class_id)  # 使用真实的类别ID
                                    
                                    # 验证类别ID是否在合理范围内
                                    if hasattr(self.parent_window, 'classes') and real_class_id < len(self.parent_window.classes):
                                        class_name = self.parent_window.classes[real_class_id]
                                        print(f"[Debug] 添加对象: 真实类别ID={real_class_id} -> 名称='{class_name}'")
                                    else:
                                        print(f"[Debug] 添加对象: 真实类别ID={real_class_id} (超出当前类别范围)")
                                    
                                except (ValueError, IndexError) as e:
                                    print(f"[Debug] 解析标注行时出错 {line_num+1}: {e}")
                                    continue
                    else:
                        print(f"[Debug] 无法读取图片: {img_path}")
                        
                except Exception as e:
                    print(f"[Debug] 读取标注文件时出错: {e}")
            else:
                print(f"[Debug] 标注文件不存在: {txt_path}")
                # 如果标注文件不存在，尝试从缓存中获取
                if img_idx in self.parent_window.label_cache:
                    print(f"[Debug] 从缓存中获取标注")
                    for bbox in self.parent_window.label_cache[img_idx]:
                        try:
                            # 从缓存中获取时，需要将标签名称转换回类别ID
                            if bbox.label in self.parent_window.classes:
                                real_class_id = self.parent_window.classes.index(bbox.label)
                                bboxes.append([bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h])
                                clses.append(real_class_id)
                                print(f"[Debug] 从缓存添加: 标签='{bbox.label}' -> 类别ID={real_class_id}")
                            else:
                                print(f"[Debug] 跳过未知标签: {bbox.label}")
                        except ValueError as e:
                            print(f"[Debug] 处理缓存标注时出错: {e}")
                            continue
            
            # 确保数据格式正确
            if bboxes:  # 只添加非空的标注
                bboxes_list.append(np.array(bboxes, dtype=np.float32))
                cls_list.append(np.array(clses, dtype=np.int64))
                print(f"[Debug] 图片 {img_idx} 添加了 {len(bboxes)} 个标注")
            else:
                bboxes_list.append(np.zeros((0, 4), dtype=np.float32))
                cls_list.append(np.zeros((0,), dtype=np.int64))
                print(f"[Debug] 图片 {img_idx} 没有有效标注")
        
        # --- 保证与prompt_image_paths长度一致，且每个元素shape正确 ---
        for i in range(len(prompt_image_paths)):
            # bboxes
            if i >= len(bboxes_list):
                bboxes_list.append(np.zeros((0, 4), dtype=np.float32))
            else:
                arr = np.array(bboxes_list[i], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, 4)
                elif arr.ndim == 0:
                    arr = np.zeros((0, 4), dtype=np.float32)
                bboxes_list[i] = arr
            # cls
            if i >= len(cls_list):
                cls_list.append(np.zeros((0,), dtype=np.int64))
            else:
                arr = np.array(cls_list[i], dtype=np.int64)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                cls_list[i] = arr
        # ---------------------------------------------------
        
        # 构建视觉特征
        visuals = {'bboxes': bboxes_list, 'cls': cls_list}
        print(f"[Debug] 获取到 {len(bboxes_list)} 个视觉特征")
        
        # 🔍 添加最终数据验证
        print(f"[Debug] 最终数据验证:")
        for i, (bboxes, cls_arr) in enumerate(zip(bboxes_list, cls_list)):
            print(f"  图片{i+1}: bboxes形状={bboxes.shape}, cls形状={cls_arr.shape}")
            if cls_arr.size > 0:
                print(f"    类别ID: {cls_arr.tolist()}")
                for cls_id in cls_arr:
                    if hasattr(self.parent_window, 'classes') and 0 <= cls_id < len(self.parent_window.classes):
                        class_name = self.parent_window.classes[cls_id]
                        print(f"      ID {cls_id} -> '{class_name}'")
                    else:
                        print(f"      ID {cls_id} -> 超出范围!")
        
        return prompt_image_paths, visuals, prompt_indices

class AutoLabelProgressDialog(QProgressDialog):
    """自动标注进度对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("auto_label_progress_title"))
        self.setModal(True)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.setMinimumDuration(0)  # 立即显示
        self.setMinimumWidth(400)
        
        # 设置样式
        self.setStyleSheet("""
            QProgressDialog {
                font-size: 12px;
            }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        
        # 初始化状态
        self.current_image = 0
        self.total_images = 0
        self.current_step = 0
        self.total_steps = 0
        
    def setup_batch_progress(self, total_images):
        """设置批量处理进度"""
        self.total_images = total_images
        self.current_image = 0
        self.setMaximum(total_images * 4)  # 每张图片4个步骤
        self.setValue(0)
        self.update_display()
        
    def update_image_progress(self, image_index, image_name):
        """更新当前处理的图片"""
        self.current_image = image_index + 1
        self.current_step = 0
        self.update_display()
        
    def update_step_progress(self, step, total_steps, message):
        """更新当前步骤进度"""
        self.current_step = step
        self.total_steps = total_steps
        
        # 计算总体进度
        overall_progress = (self.current_image - 1) * 4 + step
        self.setValue(overall_progress)
        
        # 更新显示文本
        self.setLabelText(f"{message}\nProcessing image {self.current_image}/{self.total_images}")
        
    def update_display(self):
        """更新显示"""
        if self.total_images > 0:
            self.setLabelText(f"Preparing to process image {self.current_image}/{self.total_images}")
        else:
            self.setLabelText("Preparing to start auto labeling...")
            
    def set_completed(self, success_count, total_count):
        """设置完成状态"""
        self.setValue(self.maximum())
        if success_count == total_count:
            self.setLabelText(f"✅ Auto labeling completed!\nSuccessfully processed {success_count}/{total_count} images")
        else:
            self.setLabelText(f"⚠️ Auto labeling completed\nSuccessfully processed {success_count}/{total_count} images")
            
    def set_error(self, error_message):
        """设置错误状态"""
        self.setLabelText(f"❌ Auto labeling failed\nError: {error_message}")
        
    def closeEvent(self, event):
        """重写关闭事件，防止意外关闭"""
        # 可以在这里添加确认对话框
        event.accept() 