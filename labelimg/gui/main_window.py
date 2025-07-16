import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QListWidget, QCheckBox, QComboBox, QMenu, QInputDialog,
                            QSplitter, QFrame, QDialog, QSlider, QAction,
                            QGroupBox, QSpinBox, QDoubleSpinBox, QLineEdit, QProgressDialog)
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QCursor, QPainter, QPen, QColor
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtWidgets import QMessageBox
import random
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtGui import QFont

# Corrected relative imports after moving files
from ..core.localization import tr
from ..core.bounding_box import BoundingBox
from ..core.utils import draw_dashed_rect
from .widgets import MagnifierWindow, DataAugmentationDialog, AutoLabelDialog, AutoLabelProgressDialog
from ..controller.label_controller import LabelController
from labelimg.inference.autolableing import YOLOEWrapper


class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.magnifier_enabled = True
        self.magnifier = None
        self.magnifier_active = False
        self.min_box_size = 15
        self.handle_visual_size = 6
        self.handle_detection_threshold = 8
        self.label_font_scale = 0.4
        self.cursor_pos = None
        self.label_colors = {}
        self.autosave = True
        self.image_list = []
        self.current_index = -1
        self.current_image = None
        self.bboxes = []
        self.selected_bbox = None
        self.drawing = False
        self.dragging = False
        self.resize_handle = None
        self.bbox_start = None
        self.bbox_end = None
        self.scale_factor = 1.0
        self.classes = []
        self.label_cache = {}
        self.viewed_indices = set()
        self.total_images = 0
        self.drag_threshold = 5
        self.has_dragged = False
        self.original_bbox = None
        self.selected_bboxes = set()
        self.undo_stack = {}
        self.redo_stack = {}
        # YOLOEWrapper将在类别加载后延迟初始化
        self.yoloe_wrapper = None
        
        # 初始化 label_controller
        self.label_controller = LabelController(self)

        self.initUI()
        self.loadLogo()
        self.create_magnifier_controls()  # 添加放大镜控制面板

    def create_magnifier_controls(self):
        # 创建悬浮面板
        self.magnifier_controls = QFrame(self)
        self.magnifier_controls.setFrameShape(QFrame.StyledPanel)
        self.magnifier_controls.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,220);
                border: 1px solid #aaa;
                border-radius: 8px;
            }
        """)
        self.magnifier_controls.setFixedWidth(220)
        self.magnifier_controls.setFixedHeight(90)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        from ..core.localization import tr
        self.magnifier_checkbox = QCheckBox(tr("enable_magnifier"))
        self.magnifier_checkbox.setChecked(self.magnifier_enabled)
        self.magnifier_checkbox.stateChanged.connect(self.toggleMagnifier)
        layout.addWidget(self.magnifier_checkbox)

        zoom_layout = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(2, 8)  # 2-4倍放大
        self.zoom_slider.setValue(4)     # 初始值4对应2倍
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        self.zoom_value_label = QLabel("2.0x")
        self.zoom_value_label.setMinimumWidth(40)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_value_label)
        layout.addLayout(zoom_layout)

        self.magnifier_controls.setLayout(layout)
        self.magnifier_controls.show()
        self.magnifier_controls.raise_()

    def initUI(self):
        self.setWindowTitle('BakuFlow')
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        self.magnifier = MagnifierWindow(self)
        self.magnifier.hide()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.showContextMenu)
        self.image_label.setMinimumSize(100, 100)
        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #eee, stop:1 #ccc);
                border: 1px solid #777;
                width: 10px;
                height: 10px;
            }
        """)

        self.createMainMenu()

        left_panel = QWidget()
        control_layout = QVBoxLayout(left_panel)

        format_label = QLabel(tr("output_format"))
        format_label.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 2px;")
        control_layout.addWidget(format_label)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["YOLO", "VOC", "COCO"])
        self.format_combo.setCurrentText("YOLO")
        control_layout.addWidget(self.format_combo)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.onFileSelected)
        control_layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        open_btn = QPushButton(tr("open"))
        open_btn.clicked.connect(self.openDirectory)
        btn_layout.addWidget(open_btn)
        save_btn = QPushButton(tr("save"))
        save_btn.clicked.connect(self.saveAnnotations)
        btn_layout.addWidget(save_btn)
        control_layout.addLayout(btn_layout)

        class_label = QLabel(tr("classes"))
        class_label.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 2px;")
        control_layout.addWidget(class_label)
        class_combo_layout = QHBoxLayout()
        self.class_combo = QComboBox()
        self.class_combo.setEditable(True)
        class_combo_layout.addWidget(self.class_combo)
        class_load_btn = QPushButton("...")
        class_load_btn.setFixedWidth(28)
        class_load_btn.setToolTip(tr("load_custom_classes"))
        class_load_btn.clicked.connect(self.loadClassesFromFile)
        class_combo_layout.addWidget(class_load_btn)
        control_layout.addLayout(class_combo_layout)
        self.class_combo.lineEdit().editingFinished.connect(self.syncClassComboToClasses)

        status_options_frame = QFrame()
        status_options_frame.setFrameShape(QFrame.StyledPanel)
        status_options_frame.setStyleSheet("QFrame { border: 1px solid #bbb; border-radius: 4px; margin-top: 8px; margin-bottom: 8px; }")
        status_options_layout = QVBoxLayout(status_options_frame)
        status_options_layout.setContentsMargins(8, 4, 8, 4)
        status_options_layout.setSpacing(6)
        self.status_label = QLabel(tr("ready"))
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 2px 8px; background: transparent; border: none;")
        status_options_layout.addWidget(self.status_label)
        options_layout = QHBoxLayout()
        options_layout.setSpacing(12)
        autosave_checkbox = QCheckBox(tr("auto_save"))
        autosave_checkbox.setChecked(self.autosave)
        autosave_checkbox.stateChanged.connect(self.toggleAutosave)
        options_layout.addWidget(autosave_checkbox)
        self.copy_checkbox = QCheckBox(tr("label_propagation"))
        self.copy_checkbox.setChecked(False)
        options_layout.addWidget(self.copy_checkbox)
        status_options_layout.addLayout(options_layout)
        control_layout.addWidget(status_options_frame)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.image_label, stretch=1)

        legend_frame = QFrame()
        legend_frame.setFrameShape(QFrame.StyledPanel)
        legend_layout = QVBoxLayout(legend_frame)
        legend_layout.setContentsMargins(5, 5, 5, 5)
        legend_layout.setSpacing(2)
        legend_title = QLabel(tr("color_legend"))
        legend_title.setMaximumHeight(20)
        legend_layout.addWidget(legend_title)
        self.legend_list = QListWidget()
        self.legend_list.setMaximumHeight(200)
        # self.legend_list.status_level = 1 # This attribute is not standard for QListWidget
        self.legend_list.setStyleSheet("QListWidget::item { height: 18px; }")
        legend_layout.addWidget(self.legend_list)
        control_layout.addWidget(legend_frame)

        self.statusBar = self.statusBar() # This line should be self.statusBar() not self.statusBar
        self.statusBar.showMessage(tr("ready")) # Use tr() for "Ready"

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        quit_btn = QPushButton(tr("quit"))
        quit_btn.clicked.connect(self.close)
        quit_btn.setFixedSize(80, 30)
        button_layout.addWidget(quit_btn)
        right_layout.addLayout(button_layout)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 900])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        left_panel.setMinimumWidth(200)
        right_panel.setMinimumWidth(400)
        splitter.splitterMoved.connect(self.handleSplitterMoved)

        main_layout = QHBoxLayout(main_widget)
        main_layout.addWidget(splitter)

        shortcuts = [
            (QKeySequence("F"), self.nextImage),
            (QKeySequence("D"), self.prevImage),
            (QKeySequence("S"), self.saveAnnotations),
            (QKeySequence("O"), self.openDirectory),
            (QKeySequence("E"), self.deleteSelectedBox),
            (QKeySequence("Q"), self.close),
            (QKeySequence(Qt.Key_Delete), self.deleteSelectedBox),
            (QKeySequence(Qt.Key_Up), self.prevImage),
            (QKeySequence(Qt.Key_Down), self.nextImage),
            (QKeySequence("C"), self.toggle_label_propagation),
            (QKeySequence("Ctrl+Z"), self.undo),
            (QKeySequence("Ctrl+Y"), self.redo)
        ]
        for seq, func in shortcuts:
            QShortcut(seq, self, activated=func)

        # 在创建放大镜后连接信号
        if self.magnifier:
            self.magnifier.zoom_changed.connect(self.on_magnifier_zoom_changed)
            
        # 将放大镜控制面板添加到右下角
        if hasattr(self, 'magnifier_controls'):
            self.magnifier_controls.setParent(self)
            self.magnifier_controls.move(self.width() - self.magnifier_controls.width() - 20, 
                                       self.height() - self.magnifier_controls.height() - 20)

    def on_magnifier_zoom_changed(self, zoom_factor):
        # 更新倍率显示
        self.zoom_value_label.setText(tr("current_zoom").format(zoom_factor))
        if self.magnifier:
            self.magnifier.set_zoom_factor(zoom_factor)

    def createMainMenu(self):
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu(tr("file_menu"))
        load_classes_action = file_menu.addAction(tr("load_label_file"))
        load_classes_action.triggered.connect(self.loadClassesFromFile)
        
        # 添加自动标注菜单
        auto_label_menu = main_menu.addMenu(tr("auto_label_menu"))
        auto_label_action = QAction(tr("auto_label_title"), self)
        auto_label_action.triggered.connect(self.show_auto_label_dialog)
        auto_label_menu.addAction(auto_label_action)
        
        # 添加当前图像自动标注功能
        auto_label_current_action = QAction(tr("auto_label_current_image"), self)
        auto_label_current_action.triggered.connect(self.show_auto_label_current_dialog)
        auto_label_menu.addAction(auto_label_current_action)
        
        help_menu = main_menu.addMenu(tr("help_menu"))
        shortcut_action = help_menu.addAction(tr("hotkeys_menu"))
        shortcut_action.triggered.connect(self.showShortcutHelp)
        about_action = help_menu.addAction(tr("about_menu"))
        about_action.triggered.connect(self.showAboutDialog)

        data_aug_menu = main_menu.addMenu(tr("data_aug_menu"))
        data_aug_action = QAction(tr("data_aug_title"),self)
        data_aug_action.triggered.connect(self.show_data_augmentation_dialog)
        data_aug_menu.addAction(data_aug_action)

    def showShortcutHelp(self):
        shortcuts = [
            ("F", tr("next_image_hotkey")),
            ("D", tr("prev_image_hotkey")),
            ("S", tr("save_hotkey")),
            ("O", tr("open_hotkey")),
            ("E/Delete", tr("delete_hotkey")),
            ("Q", tr("quit_hotkey")),
            ("↑", tr("prev_image_hotkey")),
            ("↓", tr("next_image_hotkey")),
            ("C", tr("label_propagation_hotkey")),
            ("Ctrl+Click", tr("multi_select_hotkey")),
            ("Ctrl+Z", tr("undo_hotkey")),
            ("Ctrl+Y", tr("redo_hotkey"))
        ]
        # QShortcut(QKeySequence("Ctrl+H"), self, activated=self.showShortcutHelp) # Recursive shortcut
        # QShortcut(QKeySequence("Ctrl+A"), self, activated=self.showAboutDialog) # Usually for Select All
        text = f"<b>{tr('hotkeys_list')}</b><br><br>"
        text += "<table>"
        for key, desc in shortcuts:
            text += f"<tr><td><code>{key}</code></td><td> - </td><td>{desc}</td></tr>"
        text += "</table>"
        msg = QMessageBox(self)
        msg.setWindowTitle(tr("hotkeys_title"))
        msg.setTextFormat(Qt.RichText)
        msg.setText(text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def showAboutDialog(self):
        about_text = f"""
        <b>BakuAI Image Labeling Tool</b><br><br>
        {tr("version")} 1.1.0<br>
        © 2024 BakuAI AS, Norway. {tr("copyright")} <br><br>
        {tr("software_description")} <br><br>
        {tr("website")}：<a href='https://bakuai.no'>bakuai.no</a>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle(tr("about_title"))
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.setIcon(QMessageBox.Information)
        msg.setMinimumWidth(320)
        msg.setMinimumHeight(200)
        msg.setStyleSheet("""
            QMessageBox {
                font-size: 14px;
            }
            QMessageBox QLabel {
                min-width: 320px; /* Adjusted to ensure content fits */
                min-height: 150px; /* Adjusted based on typical content */
            }
        """)
        msg.exec_()

    def loadLogo(self):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)) # Adjust path to project root for resources
        
        logo_path = os.path.join(base_path, "resources", "logo.png") # Assuming logo will be in labelimg/resources
        # Fallback for initial setup if logo.png is in the same directory as the original script for now
        if not os.path.exists(logo_path):
            logo_path_alt = os.path.join(os.path.dirname(__file__), "logo.png") # Check in current dir
            if os.path.exists(logo_path_alt):
                logo_path = logo_path_alt
            else: # Fallback to project root if structure is not flat
                logo_path_proj_root = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "logo.png")
                if os.path.exists(logo_path_proj_root): logo_path = logo_path_proj_root


        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
        else:
            pixmap = QPixmap(400, 300)
            pixmap.fill(QColor(240, 240, 240))
            painter = QPainter(pixmap)
            painter.setPen(Qt.blue)
            painter.setFont(self.font()) # self.font() might not be ideal here if too small
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "BakuFlow\n\nOpen image folder to start")
            painter.end()
            print(f"Logo not found at {logo_path}, using default.") # Debug print
        
        # Ensure image_label is valid before scaling
        if self.image_label and self.image_label.size().isValid() and not self.image_label.size().isEmpty():
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else: # Fallback if image_label is not ready
            self.image_label.setPixmap(pixmap.scaled(600,400, Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def toggle_label_propagation(self):
        current_state = self.copy_checkbox.isChecked()
        self.copy_checkbox.setChecked(not current_state)
        if not current_state:
            self.status_label.setText(tr("label_propagation_activated"))
        else:
            self.status_label.setText(tr("label_propagation_disabled"))

    def update_class_stats(self): # Marked for removal by user
        pass

    def updateAnnotationStats(self): # Marked for removal by user
        pass

    def update_stats_display(self):
        nav_stats = [
            f"Viewed: {len(self.viewed_indices)}",
            f"Index: {self.current_index + 1}",
            f"Total: {self.total_images}"
        ]
        self.status_label.setText(" | ".join(nav_stats))

    def calculate_annotation_stats(self): # Marked for removal by user
        pass

    def onFileSelected(self, item):
        if not item or not self.image_list:
            return
            
        filename = item.text()
        # Assuming self.current_dir is set correctly when a directory is opened
        if not hasattr(self, 'current_dir') or not self.current_dir:
             print("Error: current_dir not set.") # Or handle more gracefully
             return

        dir_path = self.current_dir # Use self.current_dir
        full_path = os.path.join(dir_path, filename)
        
        try:
            # Find index based on full_path if image_list contains full paths
            # Or find index based on filename if image_list contains just basenames
            # Assuming image_list stores full paths as per openDirectory logic
            idx = self.image_list.index(full_path)
        except ValueError:
            # Fallback: try finding by basename if full_path fails (e.g. if image_list was populated differently)
            try:
                idx = [os.path.basename(p) for p in self.image_list].index(filename)
            except ValueError:
                print(f"Error: {filename} not found in image_list.")
                return
            
        if idx != self.current_index:
            if self.autosave:
                self.saveAnnotations()
            self.current_index = idx
            self.viewed_indices.add(idx)
            self.loadImage(self.image_list[idx]) # loadImage expects full path
            self.file_list.setCurrentRow(idx)
            self.updateStatusDisplay()
            self.updateFileListColors()

    def updateFileListColors(self):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if i in self.viewed_indices:
                item.setForeground(QColor(0, 128, 0))
            else:
                item.setForeground(QColor(0, 0, 0))

    def loadClasses(self, class_path=None):
        """Load classes from specified file or default location"""
        if class_path is None:
            class_path = os.path.join(getattr(self, 'current_dir', ''), 'classes.txt')
        # Add this check right after loading classes
        
        

        if not os.path.exists(class_path):
            # Your existing logic for when file doesn't exist
            reply = QMessageBox.question(self, tr("no_classes_file_title"), 
                                        tr("no_classes_file"), 
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                file_path_selected, _ = QFileDialog.getOpenFileName(
                    self, tr("select_class_file_title"), 
                    self.current_dir if hasattr(self, 'current_dir') else "", 
                    tr("text_files_filter"))
                if file_path_selected:
                    try:
                        shutil.copy(file_path_selected, class_path)
                        with open(class_path, 'r', encoding='utf-8') as f:
                            self.classes = [line.strip() for line in f if line.strip()]
                    except Exception as e:
                        QMessageBox.warning(self, tr("load_error_title"), str(e))
                        self.classes = []
                else:
                    self.classes = []
            else:
                text, ok = QInputDialog.getMultiLineText(self, tr("enter_classes_title"), 
                                                        tr("enter_classes_prompt"))
                if ok and text.strip():
                    self.classes = [line.strip() for line in text.splitlines() if line.strip()]
                    try:
                        with open(class_path, 'w', encoding='utf-8') as f:
                            f.write("\n".join(self.classes) + "\n")
                    except Exception as e:
                        QMessageBox.warning(self, tr("class_save_failed_title"), 
                                        tr("failed_to_write_classes").format(str(e)))
                else:
                    self.classes = []
        else:
            try:
                with open(class_path, 'r', encoding='utf-8') as f:
                    self.classes = [line.strip() for line in f if line.strip()]
            except Exception as e:
                QMessageBox.warning(self, tr("load_error_title"), 
                                f"Failed to read {class_path}: {e}")
                self.classes = []
        
        self.class_combo.clear()
        self.class_combo.addItems(self.classes)
        self.updateColorLegend()
        self._init_or_update_yoloe_wrapper()

    def _init_or_update_yoloe_wrapper(self):
        """在类别加载后初始化或更新YOLOEWrapper"""
        if self.classes:  # 只有在类别列表不为空时才初始化
            try:
                print(f"[Debug] 使用动态加载的类别初始化YOLOEWrapper: {self.classes}")
                if self.yoloe_wrapper is None:
                    self.yoloe_wrapper = YOLOEWrapper(class_names=self.classes)
                    print(f"[Debug] YOLOEWrapper初始化完成")
                else:
                    # 如果已经存在，更新类别名称
                    self.yoloe_wrapper.true_class_names = self.classes
                    print(f"[Debug] YOLOEWrapper类别已更新")
            except Exception as e:
                print(f"[Error] 初始化YOLOEWrapper时出错: {e}")
                self.yoloe_wrapper = None
        else:
            print("[Debug] 类别列表为空，跳过YOLOEWrapper初始化")

    def showContextMenu(self, position):
        if self.current_image is None:
            return
            
        scaled_pos = self.getScaledPoint(position)
        clicked_bbox = None
        for bbox_item in reversed(self.bboxes): # Iterate reversed to get topmost box
            if bbox_item.contains(scaled_pos):
                clicked_bbox = bbox_item
                break # Found the box under cursor

        if clicked_bbox:
            menu = QMenu()
            edit_action = menu.addAction(tr("edit_label"))
            delete_action = menu.addAction(tr("delete")) # This should be "Delete Selected Box(es)" or similar for consistency
            
            action = menu.exec_(self.image_label.mapToGlobal(position))
            
            if action == delete_action:
                self.push_undo() # Save state before deleting
                self.bboxes.remove(clicked_bbox)
                if clicked_bbox == self.selected_bbox: # if the deleted box was the single selected
                    self.selected_bbox = None
                if clicked_bbox in self.selected_bboxes: # if part of multi-selection
                    self.selected_bboxes.remove(clicked_bbox)
                self.updateColorLegend()
                self.updateDisplay()
                if self.autosave: self.saveAnnotations()

            elif action == edit_action:
                self.push_undo() # Save state before editing
                text, ok = QInputDialog.getText(self, tr("edit_label"), 
                                              tr("enter_new_label"), 
                                              text=clicked_bbox.label)
                if ok and text:
                    if text not in self.classes:
                        self.classes.append(text)
                        self.class_combo.addItem(text)
                        class_file_path = os.path.join(getattr(self, 'current_dir', ''), 'classes.txt')
                        try:
                            with open(class_file_path, 'w', encoding='utf-8') as f:
                                f.write("\n".join(self.classes) + "\n")
                        except Exception as e:
                            QMessageBox.warning(self, tr("class_save_failed"), tr("failed_to_write_classes").format(str(e)))
                    clicked_bbox.label = text
                    self.updateColorLegend()
                    self.updateDisplay()
                    if self.autosave: self.saveAnnotations()


    def openDirectory(self):
        # 如果当前有未保存的工作，提醒用户
        if hasattr(self, 'current_dir') and self.current_dir and self.bboxes:
            reply = QMessageBox.question(
                self, 
                tr("switch_project_title"), 
                tr("switch_project_confirm"), 
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # 保存当前状态
        if hasattr(self, 'current_dir') and self.current_dir and self.autosave:
            self.saveAnnotations()
        
        dir_path = QFileDialog.getExistingDirectory(self, tr("open_directory_title"))
        if dir_path:
            # 完整的项目状态重置
            self.reset_project_state()
            
            self.current_dir = dir_path
            self.image_list = []
            self.viewed_indices = set()
            valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
            
            # 更新窗口标题显示当前项目路径
            project_name = os.path.basename(dir_path)
            self.setWindowTitle(f'BakuFlow - {project_name}')
            
            # Scan for images
            found_images = []
            for filename in sorted(os.listdir(dir_path)):
                if filename.lower().endswith(valid_extensions):
                    img_path = os.path.join(dir_path, filename)
                    test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if test_img is not None:
                        found_images.append(img_path)
                    else:
                        print(f"Warning: Skipping unreadable or invalid image file: {img_path}")

            self.image_list = found_images
            self.total_images = len(self.image_list)
            
            self.file_list.clear()
            self.file_list.addItems([os.path.basename(f) for f in self.image_list])
            
            if self.image_list:
                self.current_index = 0
                self.viewed_indices.add(0)
                self.loadClasses() # Load classes.txt from the new directory
                self.loadImage(self.image_list[0])
                self.file_list.setCurrentRow(0)
                self.updateStatusDisplay()
                self.updateFileListColors()
                
                # 显示项目加载成功信息
                self.status_label.setText(tr("project_loaded").format(project_name, len(self.image_list)))
            else:
                self.current_index = -1
                self.current_image = None
                self.image_label.clear()
                self.updateDisplay()
                self.updateColorLegend()
                self.status_label.setText(tr("no_images_found"))
                QMessageBox.information(self, tr("no_images_title"), tr("no_images_message"))

    def reset_project_state(self):
        """完整重置项目状态"""
        # 清空所有相关状态
        self.label_cache = {}
        self.label_colors = {}
        self.undo_stack = {}
        self.redo_stack = {}
        self.bboxes = []
        self.selected_bbox = None
        self.selected_bboxes = set()
        self.classes = []
        self.yoloe_wrapper = None
        
        # 重置UI状态
        self.class_combo.clear()
        self.legend_list.clear()
        self.current_index = -1
        self.current_image = None
        
        # 重置选项状态
        self.copy_checkbox.setChecked(False)
        
        print("[Debug] 项目状态已完全重置")

    def updateStatusDisplay(self):
        if self.total_images > 0 and self.current_index >=0 :
            status_text = f"{tr('viewed_status')}: {len(self.viewed_indices)} | {tr('index_status')}: {self.current_index + 1} / {self.total_images}" # Corrected
        else:
            status_text = tr('ready')
        self.status_label.setText(status_text)


    def loadImage(self, image_path):
        try:
            self.image_label.clear() # Clear previous image first
            self.bboxes = []
            self.selected_bbox = None # Reset selection
            self.selected_bboxes = set()
            # QApplication.processEvents() # Usually not needed here, can cause issues

            new_image = cv2.imread(image_path)
            if new_image is None:
                QMessageBox.critical(self, tr("load_error"), tr("cannot_read_image").format(image_path)) # Corrected
                # Potentially remove this image from list or mark as bad
                if image_path in self.image_list:
                    # A more robust way would be to handle this in openDirectory or have a way to skip bad images
                    pass # For now, just log and continue, next/prev will skip.
                return

            self.current_image = new_image
            current_height, current_width = self.current_image.shape[:2]
            
            # Load annotations based on the selected format
            self.loadAnnotationsForCurrentImage()


            # Label propagation logic
            if self.copy_checkbox.isChecked() and self.current_index > 0:
                prev_img_idx = self.current_index -1 # Check if current_index is valid before this
                # Ensure prev_img_idx is valid and in label_cache
                if prev_img_idx >= 0 and prev_img_idx < len(self.image_list) and prev_img_idx in self.label_cache :
                    prev_bboxes_data = self.label_cache[prev_img_idx] # This is a list of BoundingBox objects
                    
                    # Need dimensions of previous image for scaling
                    # This could be slow if we re-read. Consider storing dims or not scaling if dims unknown.
                    # For now, let's assume we can get prev_image_dims if needed, or simplify.
                    # Simplified: If no prev_image_dims, propagate as is or warn.
                    # For a robust solution, prev_image_dims should be available or scaling logic adjusted.
                    # Let's assume for now prev_bboxes_data contains enough info or we don't scale.
                    # The original code reads the previous image just for dimensions.

                    # If self.selected_bboxes is for the *previous* image, this logic is complex.
                    # Assuming self.selected_bboxes refers to selection on *current* image before propagation,
                    # which is unlikely for "propagate from previous".
                    # The intent is likely to propagate bboxes *from* previous image *to* current.
                    
                    # If we are propagating *all* labels from previous:
                    propagated_bboxes_list = []
                    for b_data in prev_bboxes_data: # b_data is BoundingBox object
                        # Potentially scale b_data based on prev_image and current_image dimensions
                        # For simplicity, let's assume direct propagation first, scaling can be added
                        new_bbox = BoundingBox(b_data.x, b_data.y, b_data.w, b_data.h, b_data.label)
                        
                        # Boundary check for the new image
                        new_bbox.x = max(0, min(new_bbox.x, current_width - self.min_box_size))
                        new_bbox.y = max(0, min(new_bbox.y, current_height - self.min_box_size))
                        new_bbox.w = max(self.min_box_size, min(new_bbox.w, current_width - new_bbox.x))
                        new_bbox.h = max(self.min_box_size, min(new_bbox.h, current_height - new_bbox.y))

                        if new_bbox.w >= self.min_box_size and new_bbox.h >= self.min_box_size:
                           propagated_bboxes_list.append(new_bbox)

                    if self.copy_checkbox.isChecked(): # Check again, in case it was toggled
                        if not self.selected_bboxes: # If no specific boxes were selected on prev (difficult to track), propagate all
                            self.bboxes.extend(propagated_bboxes_list) # Add to existing or replace? Original replaces.
                            self.bboxes = propagated_bboxes_list # Replace
                            self.selected_bboxes = set(self.bboxes) # Select all propagated ones
                        else:
                            # This branch implies selected_bboxes was from *previous* image state.
                            # This is tricky. For now, let's assume simple "propagate all" if checkbox is on.
                            # The original code had a complex selected_bboxes interaction.
                            # Reverting to simpler: if propagation is on, previous bboxes are propagated.
                            self.bboxes = propagated_bboxes_list
                            self.selected_bboxes = set(self.bboxes)


            # Boundary check for all bboxes after loading/propagating
            for bbox_item in self.bboxes:
                bbox_item.x = max(0, min(bbox_item.x, current_width - self.min_box_size)) # Use min_box_size for safety
                bbox_item.y = max(0, min(bbox_item.y, current_height - self.min_box_size))
                bbox_item.w = max(self.min_box_size, min(bbox_item.w, current_width - bbox_item.x))
                bbox_item.h = max(self.min_box_size, min(bbox_item.h, current_height - bbox_item.y))
            
            # Filter out too small boxes after adjustment
            self.bboxes = [b for b in self.bboxes if b.w >= self.min_box_size and b.h >= self.min_box_size]


            self.push_undo() # Save initial state of this image
            self.updateDisplay()
            self.updateColorLegend()
            if self.autosave: self.saveAnnotations() # Save after loading and potentially propagating

        except Exception as e:
            QMessageBox.critical(self, tr("load_error"), tr("image_load_failed_detailed").format(image_path, str(e))) # Corrected

    def loadAnnotationsForCurrentImage(self):
        """Helper to load annotations for the self.current_image"""
        if self.current_image is None or self.current_index < 0:
            return

        image_path = self.image_list[self.current_index]
        fmt = self.format_combo.currentText()
        annotation_path = ""

        if fmt == "YOLO":
            annotation_path = os.path.splitext(image_path)[0] + '.txt'
        elif fmt == "VOC":
            annotation_path = os.path.splitext(image_path)[0] + '.xml'
        elif fmt == "COCO":
            # COCO is usually one file for a dataset.
            # The logic needs to find annotations for the current image within that JSON.
            folder = os.path.dirname(image_path)
            # Allow user to specify COCO file or look for a default name
            # For now, assume a fixed name or that it's handled by loadAnnotations
            annotation_path = os.path.join(folder, 'coco_annotations.json')


        if os.path.exists(annotation_path):
            self.loadAnnotations(annotation_path) # This method populates self.bboxes
        else:
            self.bboxes = [] # No annotation file found


    def nextImage(self):
        if self.current_index < len(self.image_list) - 1:
            if self.autosave:
                self.saveAnnotations() 
            
            self.current_index += 1
            self.viewed_indices.add(self.current_index)
            self.loadImage(self.image_list[self.current_index])
            self.file_list.setCurrentRow(self.current_index) # Sync file list selection
            self.updateStatusDisplay()
            self.updateFileListColors()


    def prevImage(self):
        if self.current_index > 0:
            if self.autosave:
                self.saveAnnotations()

            self.current_index -= 1
            self.viewed_indices.add(self.current_index) # Should always add, even if going back
            self.loadImage(self.image_list[self.current_index])
            self.file_list.setCurrentRow(self.current_index) # Sync file list selection
            self.updateStatusDisplay()
            self.updateFileListColors()
    
    def saveAnnotations(self):
        if self.current_image is None or self.current_index < 0:
            return False
        
        # Always save YOLO, as it's the primary internal format for caching too.
        yolo_saved_successfully = self.saveYOLOAnnotation()

        fmt = self.format_combo.currentText()
        if fmt == "YOLO":
            return yolo_saved_successfully # Already saved
        elif fmt == "VOC":
            return self.saveVOCAnnotation()
        elif fmt == "COCO":
            return self.saveCOCOAnnotation()
        return False # Should not happen if combo box is constrained

    def saveYOLOAnnotation(self):
        if self.current_index < 0 or self.current_index >= len(self.image_list): 
            print("[Debug] 保存失败: 无效的图片索引")
            return False
            
        img_path = self.image_list[self.current_index]
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        ## Ensure txt_path uses the correct path separator
        # Added this to avoid error in Windows
        txt_path = txt_path.replace('/', os.sep)
         
        print(f"[Debug] 准备保存标注到: {txt_path}")
        
        if self.current_image is None: 
            print("[Debug] 保存失败: 当前没有加载图片")
            return False
        
        try:
            height, width = self.current_image.shape[:2]
            if width == 0 or height == 0: 
                print("[Debug] 保存失败: 图片尺寸无效")
                return False


            valid_boxes_for_saving = []
            for bbox in self.bboxes:

                if not (bbox.w >= self.min_box_size and bbox.h >= self.min_box_size and \
                        0 <= bbox.x < width and 0 <= bbox.y < height and \
                        bbox.x + bbox.w <= width and bbox.y + bbox.h <= height):
                    print(f"[Debug] 边界框无效，跳过")
                    continue

                try:
                    if bbox.label not in self.classes:
                        print(f"[Debug] 跳过未知标签: {bbox.label}")
                        continue
                    class_idx = self.classes.index(bbox.label)
                except ValueError:
                    print(f"[Debug] 标签索引错误: {bbox.label}")
                    continue

                x_center = (bbox.x + bbox.w / 2) / width
                y_center = (bbox.y + bbox.h / 2) / height
                w_norm = bbox.w / width
                h_norm = bbox.h / height
                valid_boxes_for_saving.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                print(f"[Debug] 添加有效边界框: class={class_idx}, label={bbox.label}, x_center={x_center:.3f}, y_center={y_center:.3f}")

            if valid_boxes_for_saving:
                print(f"[Debug] 写入 {len(valid_boxes_for_saving)} 个边界框到文件")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for line in valid_boxes_for_saving:
                        f.write(line + "\n")
                self.label_cache[self.current_index] = [BoundingBox(b.x, b.y, b.w, b.h, b.label) for b in self.bboxes if b.w >= self.min_box_size and b.h >= self.min_box_size]
                print(f"[Debug] 标注保存成功: {txt_path}")
            else:
                print(f"[Debug] 没有有效的边界框需要保存")
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    print(f"[Debug] 删除空的标注文件: {txt_path}")
                if self.current_index in self.label_cache:
                    del self.label_cache[self.current_index]
                    print(f"[Debug] 清除缓存中的标注")
            
            return True
        except Exception as e:
            print(f"[Debug] 保存标注时出错: {str(e)}")
            QMessageBox.critical(self, tr("save_error"), tr("failed_to_save_yolo").format(str(e)))
            return False

    def saveVOCAnnotation(self):
        import xml.etree.ElementTree as ET # Local import
        if self.current_index < 0 or self.current_index >= len(self.image_list): return False
        img_path = self.image_list[self.current_index]
        xml_path = os.path.splitext(img_path)[0] + '.xml'

        if self.current_image is None: return False
        
        try:
            height, width = self.current_image.shape[:2]
            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'folder').text = os.path.basename(os.path.dirname(img_path))
            ET.SubElement(annotation, 'filename').text = os.path.basename(img_path)
            # Add path for VOC, often useful
            ET.SubElement(annotation, 'path').text = img_path 


            source = ET.SubElement(annotation, 'source')
            ET.SubElement(source, 'database').text = 'Unknown' # Or a project name

            size = ET.SubElement(annotation, 'size')
            ET.SubElement(size, 'width').text = str(width)
            ET.SubElement(size, 'height').text = str(height)
            ET.SubElement(size, 'depth').text = str(self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1) # Handle grayscale

            ET.SubElement(annotation, 'segmented').text = '0' # Standard for object detection

            for bbox in self.bboxes:
                if not (bbox.w >= self.min_box_size and bbox.h >= self.min_box_size): continue

                obj = ET.SubElement(annotation, 'object')
                ET.SubElement(obj, 'name').text = bbox.label
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0' # Heuristic: if box touches image edge, it might be truncated
                ET.SubElement(obj, 'difficult').text = '0'
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(bbox.x)
                ET.SubElement(bndbox, 'ymin').text = str(bbox.y)
                ET.SubElement(bndbox, 'xmax').text = str(bbox.x + bbox.w)
                ET.SubElement(bndbox, 'ymax').text = str(bbox.y + bbox.h)
            
            tree = ET.ElementTree(annotation)
            # Prettify XML output if possible (optional)
            try:
                from xml.dom import minidom
                xml_str = ET.tostring(annotation, 'utf-8')
                pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
                with open(xml_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml_str)
            except ImportError:
                tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            
            return True
        except Exception as e:
            QMessageBox.critical(self, tr("voc_save_error_title"), tr("voc_save_error_msg").format(str(e))) # Corrected
            return False

    def saveCOCOAnnotation(self):
        # COCO format is dataset-wide. This implies saving all annotations for all images in self.image_list.
        # This is a larger operation than per-image save.
        # The current self.bboxes is only for the current image.
        # We need to iterate all images, load/retrieve their bboxes (from cache or YOLO files), and compile.
        import json # Local import
        if not self.image_list or not hasattr(self, 'current_dir') or not self.current_dir:
            QMessageBox.warning(self, tr("coco_save_warn_title"), tr("coco_save_warn_msg_no_data")) # Corrected
            return False

        json_path = os.path.join(self.current_dir, 'coco_annotations.json')
        
        coco_output = {
            "info": {"description": "BakuLabel COCO export", "version": "1.0", "year": 2024, "contributor": "BakuLabel User", "date_created": ""}, # Add date
            "licenses": [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}], # Example license
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Populate categories
        for i, class_name in enumerate(self.classes):
            coco_output["categories"].append({"id": i, "name": class_name, "supercategory": "object"}) # supercategory can be more specific

        annotation_id_counter = 1
        
        # Iterate through all images in the current directory/list
        for img_idx, img_path_iter in enumerate(self.image_list):
            # Get image dimensions
            # temp_img = cv2.imread(img_path_iter) # Reading each image again can be slow
            # A better way: store dimensions when first loaded, or get from cache if possible
            # For now, let's try to get it from current_image if it's the one, else read.
            h_img, w_img = (0,0)
            if img_idx == self.current_index and self.current_image is not None:
                 h_img, w_img = self.current_image.shape[:2]
            else: # Need to read the image for dimensions
                temp_img_for_dim = cv2.imread(img_path_iter)
                if temp_img_for_dim is None:
                    print(f"Warning: Skipping {img_path_iter} for COCO export, cannot read for dimensions.")
                    continue
                h_img, w_img = temp_img_for_dim.shape[:2]

            if h_img == 0 or w_img == 0: continue # Skip if dimensions are invalid

            coco_output["images"].append({
                "id": img_idx, # Use current list index as image ID
                "file_name": os.path.basename(img_path_iter),
                "height": h_img,
                "width": w_img,
                "license": 1, # Example license id
                 # "flickr_url": "", "coco_url": "", "date_captured": "" # Optional fields
            })

            # Get bboxes for this image_path_iter
            # Priority: 1. self.bboxes if current image, 2. self.label_cache, 3. load from YOLO .txt
            bboxes_for_this_image = []
            if img_idx == self.current_index:
                bboxes_for_this_image = self.bboxes # Use current in-memory bboxes
            elif img_idx in self.label_cache:
                bboxes_for_this_image = self.label_cache[img_idx]
            else: # Try to load from its YOLO .txt file
                yolo_txt_path = os.path.splitext(img_path_iter)[0] + '.txt'
                if os.path.exists(yolo_txt_path):
                    # This is simplified load logic, assumes self.classes is populated
                    # Ideally, a dedicated YOLO load function for this context.
                    with open(yolo_txt_path, 'r', encoding='utf-8') as f_yolo:
                        for line in f_yolo:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                try:
                                    cls_idx, xc, yc, nw, nh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                    if 0 <= cls_idx < len(self.classes):
                                        label = self.classes[cls_idx]
                                        x_abs = int((xc - nw / 2) * w_img)
                                        y_abs = int((yc - nh / 2) * h_img)
                                        w_abs = int(nw * w_img)
                                        h_abs = int(nh * h_img)
                                        bboxes_for_this_image.append(BoundingBox(x_abs, y_abs, w_abs, h_abs, label))
                                except ValueError:
                                    continue # Skip malformed lines
            
            for bbox_item in bboxes_for_this_image:
                if not (bbox_item.w >= self.min_box_size and bbox_item.h >= self.min_box_size): continue
                try:
                    category_id = self.classes.index(bbox_item.label) # Use index in self.classes as category_id
                except ValueError:
                    continue # Skip if label not in master list

                coco_output["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": img_idx, # Matches image id above
                    "category_id": category_id,
                    "bbox": [float(bbox_item.x), float(bbox_item.y), float(bbox_item.w), float(bbox_item.h)],
                    "area": float(bbox_item.w * bbox_item.h),
                    "iscrowd": 0,
                    "segmentation": [] # Required by COCO, empty for bbox format
                })
                annotation_id_counter += 1
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f_json:
                json.dump(coco_output, f_json, indent=2) # Use indent for readability
            return True
        except Exception as e:
            QMessageBox.critical(self, tr("coco_save_error_title"), tr("coco_save_error_msg").format(str(e))) # Corrected
            return False


    def toggleAutosave(self, state):
        self.autosave = state == Qt.Checked
        # self.status_label.setText(f"Autosave {'Enabled' if self.autosave else 'Disabled'}") # This overwrites main status
        self.statusBar.showMessage(f"Autosave {'Enabled' if self.autosave else 'Disabled'}", 2000) # Show in status bar temporarily

    def loadAnnotations(self, annotation_path):
        """Load annotations from YOLO, VOC, or COCO format for the current image."""
        # This method is called by loadImage, self.bboxes should be cleared before calling this
        # self.bboxes = [] # Ensure bboxes are for the current file
        
        if not os.path.exists(annotation_path) or self.current_image is None:
            self.bboxes = [] # Ensure clean state if no file or image
            return

        height, width = self.current_image.shape[:2]
        ext = os.path.splitext(annotation_path)[1].lower()
        loaded_bboxes = []

        try:
            if ext == '.txt':  # YOLO
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5: continue
                        try:
                            class_idx, x_center, y_center, w_norm, h_norm = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            x = int((x_center - w_norm / 2) * width)
                            y = int((y_center - h_norm / 2) * height)
                            w_pixels = int(w_norm * width)
                            h_pixels = int(h_norm * height)
                            label = self.classes[class_idx] if 0 <= class_idx < len(self.classes) else "unknown"
                            loaded_bboxes.append(BoundingBox(x, y, w_pixels, h_pixels, label))
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing YOLO line: {line} - {str(e)}")
            
            elif ext == '.xml':  # VOC
                import xml.etree.ElementTree as ET
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    loaded_bboxes.append(BoundingBox(xmin, ymin, xmax - xmin, ymax - ymin, label))
            
            elif ext == '.json':  # COCO (specific to current image)
                import json
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                
                current_image_filename = os.path.basename(self.image_list[self.current_index])
                current_image_id = -1

                # Find image_id for current_image_filename
                for img_info in coco_data.get("images", []):
                    if img_info.get("file_name", "").lower() == current_image_filename.lower():
                        current_image_id = img_info["id"]
                        break
                
                if current_image_id != -1:
                    category_map = {cat['id']: cat['name'] for cat in coco_data.get("categories", [])}
                    for ann in coco_data.get("annotations", []):
                        if ann.get("image_id") == current_image_id:
                            coco_bbox = ann['bbox'] # [x, y, width, height]
                            category_id = ann['category_id']
                            label = category_map.get(category_id, "unknown")
                            
                            # Ensure label is in self.classes, add if not (important for COCO)
                            if label != "unknown" and label not in self.classes:
                                self.classes.append(label)
                                self.class_combo.addItem(label) # Keep UI in sync

                            x, y, w, h = [int(c) for c in coco_bbox]
                            loaded_bboxes.append(BoundingBox(x, y, w, h, label))
                else:
                    print(f"Warning: Image {current_image_filename} not found in COCO json file {annotation_path}")


            self.bboxes = loaded_bboxes # Assign loaded bboxes
            self.updateColorLegend() # Update legend after loading

        except Exception as e:
            QMessageBox.critical(self, tr("load_error"), tr("annotation_load_failed").format(annotation_path, str(e))) # Corrected
            self.bboxes = [] # Clear bboxes on error
            self.updateColorLegend()


    def toggleMagnifier(self, state):
        self.magnifier_enabled = state == Qt.Checked
        if self.magnifier:
            if self.magnifier_enabled and self.magnifier_active:
                self.magnifier.show()
            else:
                self.magnifier.hide()
            # 更新复选框状态
            self.magnifier_checkbox.setChecked(self.magnifier_enabled)

    def get_label_color(self, label):
        if label not in self.label_colors:
            # Simplified color generation, can be improved for more distinct colors
            random.seed(hash(label)) # Ensure consistent color for the same label
            r = random.randint(50, 200)
            g = random.randint(50, 200)
            b = random.randint(50, 200)
            # Ensure color is not too dark or too light for visibility (simple check)
            # if (r + g + b) / 3 < 80: r,g,b = r+50, g+50, b+50
            # if (r + g + b) / 3 > 180: r,g,b = r-50, g-50, b-50
            self.label_colors[label] = (max(0,min(r,255)), max(0,min(g,255)), max(0,min(b,255)))
        return self.label_colors[label]


    def deleteSelectedBox(self):
        deleted_something = False
        if self.selected_bboxes: # Multi-select delete
            self.push_undo()
            for bbox_to_delete in list(self.selected_bboxes): # Iterate over a copy
                if bbox_to_delete in self.bboxes:
                    self.bboxes.remove(bbox_to_delete)
                    deleted_something = True
            self.selected_bboxes.clear()
            self.selected_bbox = None # Clear single selection as well
        elif self.selected_bbox: # Single-select delete
            self.push_undo()
            if self.selected_bbox in self.bboxes:
                self.bboxes.remove(self.selected_bbox)
                deleted_something = True
            self.selected_bbox = None
        
        if deleted_something:
            self.updateDisplay()
            self.updateColorLegend()
            if self.autosave:
                self.saveAnnotations()
        # else: # No selection to delete
            # QMessageBox.information(self, tr("no_selection_title", "No Selection"), tr("no_selection_msg_delete", "Please select a bounding box to delete."))


    def keyPressEvent(self, event):
        # This keyPressEvent is for the main window. Specific widgets might handle their own.
        # Standard keys like Delete for selected items are often handled by QAction or QShortcut.
        # The existing shortcuts for E/Delete already call self.deleteSelectedBox.
        # This can be used for other global key presses if needed.
        
        # Example: Toggle magnifier with a key
        if event.key() == Qt.Key_M: # Example: M key for magnifier
            if self.magnifier: # Check if magnifier exists
                 self.magnifier_enabled = not self.magnifier_enabled
                 self.statusBar.showMessage(f"Magnifier {'Enabled' if self.magnifier_enabled else 'Disabled'}", 2000)
                 # Show/hide logic for magnifier might need adjustment based on mouse state
                 if not self.magnifier_enabled: self.magnifier.hide()
                 # If enabling, it will show on next mouse press/move if active

        super().keyPressEvent(event) # Call base class method


    def updateDisplay(self):
        if self.current_image is None:
            self.image_label.clear() # Clear if no image
            return

        # Make a mutable copy for drawing
        display_image_bgr = self.current_image.copy()
        # Convert to RGB for QImage display
        display_image_rgb = cv2.cvtColor(display_image_bgr, cv2.COLOR_BGR2RGB)
        
        height, width, _ = display_image_rgb.shape # Get dimensions from RGB image

        # Scale factor calculation needs to be robust
        if self.image_label.size().isEmpty() or width == 0 or height == 0 :
             self.scale_factor = 1.0 # Default if label size not ready
        else:
            label_size = self.image_label.size()
            width_ratio = label_size.width() / width
            height_ratio = label_size.height() / height
            self.scale_factor = min(width_ratio, height_ratio)
            if self.scale_factor <=0 : self.scale_factor = 1.0 # Safety for invalid scale

        font_scale = self.label_font_scale # Use class attribute
        thickness = 1

        # Draw existing bboxes
        for bbox_item in self.bboxes:
            color = self.get_label_color(bbox_item.label)
            # Determine if the box is selected (either single or multi-select)
            is_selected = (bbox_item == self.selected_bbox) or (bbox_item in self.selected_bboxes)

            if is_selected:
                # Draw a thicker, dashed rectangle for selected items
                # For multi-selected, use a common highlight color or slightly different from single-selected
                highlight_color = (255, 0, 0) if bbox_item in self.selected_bboxes and bbox_item != self.selected_bbox else color # Red for multi, box color for single
                
                # Create a filled overlay for selection emphasis (subtle)
                overlay = display_image_rgb.copy()
                cv2.rectangle(overlay, (bbox_item.x, bbox_item.y), (bbox_item.x + bbox_item.w, bbox_item.y + bbox_item.h), highlight_color, -1)
                alpha = 0.15 # Transparency of the fill
                cv2.addWeighted(overlay, alpha, display_image_rgb, 1 - alpha, 0, display_image_rgb)

                draw_dashed_rect(display_image_rgb, (bbox_item.x, bbox_item.y), 
                                 (bbox_item.x + bbox_item.w, bbox_item.y + bbox_item.h), 
                                 highlight_color, thickness +1, 8) # Thicker and dashed
                
                # Draw resize handles for the primary selected_bbox
                if bbox_item == self.selected_bbox:
                    handle_sz = self.handle_visual_size
                    # Top-left, top-right, bottom-left, bottom-right
                    handles_coords = [
                        (bbox_item.x, bbox_item.y), (bbox_item.x + bbox_item.w, bbox_item.y),
                        (bbox_item.x, bbox_item.y + bbox_item.h), (bbox_item.x + bbox_item.w, bbox_item.y + bbox_item.h)
                    ]
                    for hx, hy in handles_coords:
                        cv2.rectangle(display_image_rgb, (hx - handle_sz//2, hy - handle_sz//2), 
                                      (hx + handle_sz//2, hy + handle_sz//2), (0, 255, 255), -1) # Cyan handles
            else:
                # Normal box for non-selected items
                cv2.rectangle(display_image_rgb, (bbox_item.x, bbox_item.y), 
                              (bbox_item.x + bbox_item.w, bbox_item.y + bbox_item.h), color, thickness)

            # Draw label text (semi-transparent)
            text = bbox_item.label
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Text background for better readability (optional)
            # cv2.rectangle(display_image_rgb, (bbox_item.x, bbox_item.y - text_height - baseline -2), 
            #               (bbox_item.x + text_width + 2, bbox_item.y - baseline), (color[0]//2, color[1]//2, color[2]//2), -1) # Darker bg

            # Put text with slight offset from top-left corner of the box
            # Create a temporary layer for semi-transparent text to avoid issues with direct drawing
            text_layer_temp = np.zeros_like(display_image_rgb, dtype=np.uint8)
            cv2.putText(text_layer_temp, text, (bbox_item.x + 2, bbox_item.y - 5), # Adjusted position
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            
            text_alpha = 0.7 # Transparency of text
            text_mask = cv2.cvtColor(text_layer_temp, cv2.COLOR_RGB2GRAY) > 0
            display_image_rgb[text_mask] = (display_image_rgb[text_mask] * (1 - text_alpha) + text_layer_temp[text_mask] * text_alpha).astype(np.uint8)


        # Draw current drawing box (if any)
        if self.drawing and self.bbox_start and self.bbox_end:
            x1, y1 = self.bbox_start.x(), self.bbox_start.y()
            x2, y2 = self.bbox_end.x(), self.bbox_end.y()
            draw_dashed_rect(display_image_rgb, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), 
                             (0, 255, 255), 2) # Cyan dashed line for drawing

        # Convert to QPixmap for display
        bytes_per_line = display_image_rgb.strides[0]
        q_image = QImage(display_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        scaled_width = int(width * self.scale_factor)
        scaled_height = int(height * self.scale_factor)
        
        if scaled_width > 0 and scaled_height > 0 :
            pixmap = QPixmap.fromImage(q_image).scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        else: # Fallback if scaling results in zero size
            self.image_label.setPixmap(QPixmap.fromImage(q_image))


    def handleSplitterMoved(self, pos, index):
        # This might be called frequently during drag. Update display if needed.
        # Check if current_image exists to avoid errors if splitter is moved before image load
        if self.current_image is not None:
            self.updateDisplay() # Re-calculate scale and update pixmap

    def getScaledPoint(self, point_in_label_coords):
        if not self.image_label.pixmap() or self.image_label.pixmap().isNull() or self.scale_factor == 0:
            return QPoint(0, 0) # Return a default QPoint if no valid pixmap or scale

        pixmap_actual_size = self.image_label.pixmap().size() # Actual scaled pixmap size
        label_widget_size = self.image_label.size() # Size of the QLabel widget

        # Calculate offsets for centered pixmap
        x_offset = max(0, (label_widget_size.width() - pixmap_actual_size.width()) // 2)
        y_offset = max(0, (label_widget_size.height() - pixmap_actual_size.height()) // 2)

        # Adjust point to be relative to the pixmap itself (top-left of pixmap is 0,0)
        x_on_pixmap = point_in_label_coords.x() - x_offset
        y_on_pixmap = point_in_label_coords.y() - y_offset

        # Clamp to pixmap boundaries to avoid negative coords or coords outside scaled image
        x_on_pixmap_clamped = max(0, min(x_on_pixmap, pixmap_actual_size.width() -1))
        y_on_pixmap_clamped = max(0, min(y_on_pixmap, pixmap_actual_size.height() -1))


        # Scale back to original image coordinates
        original_x = int(x_on_pixmap_clamped / self.scale_factor)
        original_y = int(y_on_pixmap_clamped / self.scale_factor)
        
        return QPoint(original_x, original_y)


    def mousePressEvent(self, event):
        # This event is now directly on image_label, not the main window for drawing.
        if event.button() == Qt.LeftButton and self.current_image is not None:
            pos_on_label = event.pos() # Position relative to image_label widget
            scaled_pos = self.getScaledPoint(pos_on_label) # Convert to image coordinates
            
            self.has_dragged = False # Reset drag flag
            box_interaction_found = False
            ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier) # Simpler boolean

            # Check for resize handle first (higher priority than dragging whole box)
            if self.selected_bbox: # Only check handles if a box is already selected
                handle = self.selected_bbox.get_resize_handle(scaled_pos, self.handle_detection_threshold)
                if handle:
                    self.resize_handle = handle
                    self.drag_start_pos = scaled_pos # For calculating delta during resize
                    self.original_bbox_state = BoundingBox(self.selected_bbox.x, self.selected_bbox.y, 
                                                         self.selected_bbox.w, self.selected_bbox.h, 
                                                         self.selected_bbox.label)
                    box_interaction_found = True
            
            if not box_interaction_found: # If not resizing, check for click on a box
                clicked_on_existing_box = None
                # Iterate reversed: topmost boxes get priority
                for bbox_item in reversed(self.bboxes): 
                    if bbox_item.contains(scaled_pos):
                        clicked_on_existing_box = bbox_item
                        break 
                
                if clicked_on_existing_box:
                    box_interaction_found = True
                    self.push_undo() # Save state before potential modification (drag/selection change)
                    if ctrl_pressed: # Multi-selection toggle
                        if clicked_on_existing_box in self.selected_bboxes:
                            self.selected_bboxes.remove(clicked_on_existing_box)
                            # If the removed box was the primary selected_bbox, clear it
                            if clicked_on_existing_box == self.selected_bbox:
                                self.selected_bbox = self.selected_bboxes.copy().pop() if self.selected_bboxes else None
                        else:
                            self.selected_bboxes.add(clicked_on_existing_box)
                            self.selected_bbox = clicked_on_existing_box # Make it the primary for potential drag
                    else: # Single selection
                        self.selected_bboxes = {clicked_on_existing_box} # Set of one
                        self.selected_bbox = clicked_on_existing_box
                    
                    # Prepare for dragging the selected_bbox (even if part of multi-select, primary moves)
                    self.dragging = True 
                    self.drag_start_pos = scaled_pos
                    # Save original state of all selected boxes for consistent multi-drag
                    self.original_selected_bboxes_states = {
                        b: BoundingBox(b.x,b.y,b.w,b.h,b.label) for b in self.selected_bboxes
                    }
                    self.original_primary_bbox_drag_ref = BoundingBox( # Ref for primary selected box
                        self.selected_bbox.x, self.selected_bbox.y, self.selected_bbox.w, self.selected_bbox.h, self.selected_bbox.label
                    ) if self.selected_bbox else None

            if not box_interaction_found: # Start drawing a new box
                self.push_undo() # Save state before drawing new box
                self.drawing = True
                self.bbox_start = scaled_pos
                self.bbox_end = scaled_pos # Initialize end to start
                self.selected_bbox = None # Clear previous selection
                self.selected_bboxes = set()

            # Magnifier logic
            if self.magnifier_enabled:
                self.magnifier_active = True
                if self.magnifier: self.magnifier.show() # Ensure magnifier widget exists
            
            self.updateDisplay()


    def mouseMoveEvent(self, event):
        if self.current_image is None: return

        pos_on_label = event.pos()
        scaled_pos = self.getScaledPoint(pos_on_label)

        # Update cursor shape based on context (resize, drag, draw)
        current_cursor_shape = Qt.ArrowCursor # Default
        if self.selected_bbox and not self.drawing and not self.dragging and not self.resize_handle:
            handle_under_mouse = self.selected_bbox.get_resize_handle(scaled_pos, self.handle_detection_threshold)
            if handle_under_mouse:
                if handle_under_mouse in ('top-left', 'bottom-right'): current_cursor_shape = Qt.SizeFDiagCursor
                elif handle_under_mouse in ('top-right', 'bottom-left'): current_cursor_shape = Qt.SizeBDiagCursor
                # Add top, bottom, left, right cursors if you implement those handles
                # elif 'top' in handle_under_mouse or 'bottom' in handle_under_mouse: current_cursor_shape = Qt.SizeVerCursor
                # elif 'left' in handle_under_mouse or 'right' in handle_under_mouse: current_cursor_shape = Qt.SizeHorCursor
        
        self.image_label.setCursor(QCursor(current_cursor_shape))


        if self.drawing and self.bbox_start:
            self.bbox_end = scaled_pos
            self.has_dragged = True #Counts as drag for min_box_size check
        
        elif self.resize_handle and self.selected_bbox and self.original_bbox_state and self.drag_start_pos:
            dx = scaled_pos.x() - self.drag_start_pos.x()
            dy = scaled_pos.y() - self.drag_start_pos.y()
            
            orig = self.original_bbox_state
            new_x, new_y, new_w, new_h = orig.x, orig.y, orig.w, orig.h

            if 'left' in self.resize_handle:
                new_x = orig.x + dx
                new_w = orig.w - dx
            elif 'right' in self.resize_handle:
                new_w = orig.w + dx
            
            if 'top' in self.resize_handle:
                new_y = orig.y + dy
                new_h = orig.h - dy
            elif 'bottom' in self.resize_handle:
                new_h = orig.h + dy

            # Ensure minimum size and positive width/height
            if new_w < self.min_box_size:
                if 'left' in self.resize_handle: new_x = self.selected_bbox.x + self.selected_bbox.w - self.min_box_size
                new_w = self.min_box_size
            if new_h < self.min_box_size:
                if 'top' in self.resize_handle: new_y = self.selected_bbox.y + self.selected_bbox.h - self.min_box_size
                new_h = self.min_box_size
            
            # Boundary checks against image dimensions
            img_h, img_w = self.current_image.shape[:2]
            new_x = max(0, min(new_x, img_w - self.min_box_size if new_w == self.min_box_size else img_w - new_w))
            new_y = max(0, min(new_y, img_h - self.min_box_size if new_h == self.min_box_size else img_h - new_h))
            new_w = min(new_w, img_w - new_x)
            new_h = min(new_h, img_h - new_y)


            self.selected_bbox.x, self.selected_bbox.y = new_x, new_y
            self.selected_bbox.w, self.selected_bbox.h = new_w, new_h
            self.has_dragged = True

        elif self.dragging and self.selected_bboxes and self.original_primary_bbox_drag_ref and self.drag_start_pos:
            # Calculate delta from the primary selected box's original position
            dx = scaled_pos.x() - self.drag_start_pos.x()
            dy = scaled_pos.y() - self.drag_start_pos.y()

            img_h, img_w = self.current_image.shape[:2]

            for bbox_item in self.selected_bboxes:
                original_state = self.original_selected_bboxes_states.get(bbox_item)
                if not original_state: continue

                # Calculate new top-left for this box based on common delta
                new_x = original_state.x + dx
                new_y = original_state.y + dy

                # Boundary checks for each box being dragged
                new_x = max(0, min(new_x, img_w - bbox_item.w))
                new_y = max(0, min(new_y, img_h - bbox_item.h))
                
                bbox_item.x = new_x
                bbox_item.y = new_y
            self.has_dragged = True
            
        if self.has_dragged: # Only update display if something changed
            self.updateDisplay()
        
        if self.magnifier_enabled and self.magnifier_active and self.magnifier:
            self.magnifier.update() # Tell magnifier to repaint


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_image is not None:
            if self.magnifier_enabled and self.magnifier:
                self.magnifier_active = False
                self.magnifier.hide()

            if self.drawing and self.bbox_start and self.bbox_end:
                x1 = min(self.bbox_start.x(), self.bbox_end.x())
                y1 = min(self.bbox_start.y(), self.bbox_end.y())
                x2 = max(self.bbox_start.x(), self.bbox_end.x())
                y2 = max(self.bbox_start.y(), self.bbox_end.y())
                w, h = x2 - x1, y2 - y1

                if w >= self.min_box_size and h >= self.min_box_size:
                    # Determine label for the new box
                    current_default_label = self.class_combo.currentText() if self.class_combo.currentText() else (self.classes[0] if self.classes else "Default")
                    
                    # Check if a class is selected in combobox, else prompt
                    if not self.classes or not current_default_label:
                         label_text, ok = QInputDialog.getText(self, tr("new_box_label_title"), # Corrected
                                                             tr("new_box_label_prompt"), # Corrected
                                                             text="object")
                    else: # Use current class_combo selection or first class as default
                        items = [self.class_combo.itemText(i) for i in range(self.class_combo.count())]
                        if not items: # if combo is empty (e.g. no classes.txt)
                             label_text, ok = QInputDialog.getText(self, tr("new_box_label_title"), # Corrected
                                                             tr("new_box_label_prompt"),  # Corrected
                                                             text="object")
                        else:
                            label_text, ok = QInputDialog.getItem(self, tr("select_label_title"), # Corrected
                                                                tr("select_label_prompt"), # Corrected
                                                                items, 0, True) # editable=True

                    if ok and label_text:
                        # self.push_undo() was called at mousePress for drawing
                        if label_text not in self.classes:
                            self.classes.append(label_text)
                            self.class_combo.addItem(label_text)
                            self.syncClassComboToClasses(force_save=True) # Save new class to classes.txt
                        
                        new_bbox = BoundingBox(x1, y1, w, h, label_text)
                        self.bboxes.append(new_bbox)
                        self.selected_bbox = new_bbox # Select the new box
                        self.selected_bboxes = {new_bbox}
                        self.updateColorLegend()
                else: # Box too small, pop the undo state pushed at mousePress
                    idx = self.current_index
                    if idx >= 0 and idx in self.undo_stack and self.undo_stack[idx]:
                        self.undo_stack[idx].pop()


            # Reset states
            self.drawing = False
            self.dragging = False
            self.resize_handle = None
            self.bbox_start = None
            self.bbox_end = None
            self.original_bbox_state = None
            self.original_selected_bboxes_states = {}
            self.original_primary_bbox_drag_ref = None
            self.drag_start_pos = None

            if self.has_dragged: # If a drag/resize/draw happened
                # The undo state was already pushed at the start of the operation (mousePress)
                # So, no need to call self.push_undo() here again unless it's a new box creation fully completed.
                # For new box, it was pushed in mousePress, confirmed here.
                # For drag/resize, it was pushed in mousePress.
                if self.autosave:
                    self.saveAnnotations()
            
            self.updateDisplay()


    def loadClassesFromFile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, tr("select_label_file_title"), # Corrected
                                                 self.current_dir if hasattr(self, 'current_dir') else "", 
                                                 tr("text_files_filter")) # Corrected
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_classes = [line.strip() for line in f if line.strip()]
                
                self.classes = loaded_classes
                self.class_combo.clear()
                self.class_combo.addItems(self.classes)
                
                # Copy to current project directory if a directory is loaded
                if hasattr(self, 'current_dir') and self.current_dir:
                    target_class_path = os.path.join(self.current_dir, 'classes.txt')
                    import shutil
                    try:
                        shutil.copy(file_path, target_class_path)
                        QMessageBox.information(self, tr("load_success_title"), # Corrected
                                                tr("load_success_msg_copied").format(os.path.basename(file_path))) # Corrected
                    except Exception as e_copy:
                        QMessageBox.warning(self, tr("copy_error_title"), # Corrected
                                            tr("copy_error_msg_class_file").format(str(e_copy))) # Corrected
                else: # No current directory, just loaded in memory
                     QMessageBox.information(self, tr("load_success_title"), tr("load_success_msg").format(os.path.basename(file_path))) # Corrected
                self.updateColorLegend() # Update legend with new classes

            except Exception as e:
                QMessageBox.critical(self, tr("load_failed_title"), str(e)) # Corrected

    def syncClassComboToClasses(self, force_save=False):
        current_text = self.class_combo.currentText().strip()
        if current_text and current_text not in self.classes:
            self.classes.append(current_text)
            self.class_combo.blockSignals(True)
            self.class_combo.clear()
            self.class_combo.addItems(self.classes)
            self.class_combo.setCurrentText(current_text)
            self.class_combo.blockSignals(False)
            force_save = True
        if force_save and hasattr(self, 'current_dir') and self.current_dir:
            class_path_to_save = os.path.join(self.current_dir, 'classes.txt')
            try:
                with open(class_path_to_save, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.classes) + "\n")
                print(f"Updated {class_path_to_save}")
            except Exception as e:
                QMessageBox.warning(self, tr("class_save_failed_title"), tr("failed_to_write_classes").format(str(e)))
        self.updateColorLegend()
        # 类别更新后，更新YOLOEWrapper
        self._init_or_update_yoloe_wrapper()


    def updateColorLegend(self):
        self.legend_list.clear()
        current_image_counts = {}
        if self.current_image is not None: # Only count for current image
            for bbox_item in self.bboxes:
                current_image_counts[bbox_item.label] = current_image_counts.get(bbox_item.label, 0) + 1
        
        # Display all known classes, even if count is 0 for current image
        for label_text in self.classes: 
            color = self.get_label_color(label_text)
            count = current_image_counts.get(label_text, 0)
            
            item = QListWidgetItem(f"■ {label_text} ({count})")
            item.setForeground(QColor(*color)) # Use QColor correctly
            item.setFont(QFont("Arial", 10)) # Removed Bold for cleaner look, can be preference
            # 存储类别名称到item数据中，方便删除时使用
            item.setData(Qt.UserRole, label_text)
            self.legend_list.addItem(item)
        
        # 设置右键菜单
        self.legend_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.legend_list.customContextMenuRequested.connect(self.showLegendContextMenu)

    def showLegendContextMenu(self, position):
        """显示图例右键菜单"""
        item = self.legend_list.itemAt(position)
        if not item:
            return
            
        menu = QMenu()
        delete_action = menu.addAction(tr("delete_class"))
        
        action = menu.exec_(self.legend_list.mapToGlobal(position))
        
        if action == delete_action:
            label_text = item.data(Qt.UserRole)
            if label_text:
                self.delete_class(label_text)
    
    def delete_class(self, class_name):
        """删除指定的类别"""
        # 确认对话框
        reply = QMessageBox.question(
            self, 
            tr("confirm_delete_class_title"), 
            tr("confirm_delete_class_msg").format(class_name), 
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # 检查当前图像中是否有使用该类别的标注
            affected_bboxes = [bbox for bbox in self.bboxes if bbox.label == class_name]
            
            if affected_bboxes:
                # 询问是否删除相关标注
                bbox_reply = QMessageBox.question(
                    self,
                    tr("delete_related_annotations_title"),
                    tr("delete_related_annotations_msg").format(len(affected_bboxes), class_name),
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if bbox_reply == QMessageBox.Yes:
                    # 删除相关标注
                    self.push_undo()  # 保存撤销状态
                    self.bboxes = [bbox for bbox in self.bboxes if bbox.label != class_name]
                    
                    # 清除选择
                    if self.selected_bbox and self.selected_bbox.label == class_name:
                        self.selected_bbox = None
                    self.selected_bboxes = {bbox for bbox in self.selected_bboxes if bbox.label != class_name}
                else:
                    # 用户选择不删除标注，取消删除类别
                    return
            
            # 从类别列表中删除
            if class_name in self.classes:
                self.classes.remove(class_name)
                print(f"[Debug] 已删除类别: {class_name}")
            
            # 从下拉框中删除
            combo_index = self.class_combo.findText(class_name)
            if combo_index >= 0:
                self.class_combo.removeItem(combo_index)
            
            # 从颜色缓存中删除
            if class_name in self.label_colors:
                del self.label_colors[class_name]
            
            # 保存更新后的类别列表到文件
            self.save_classes_to_file()
            
            # 更新YOLOEWrapper
            self._init_or_update_yoloe_wrapper()
            
            # 更新显示
            self.updateColorLegend()
            self.updateDisplay()
            
            # 自动保存
            if self.autosave:
                self.saveAnnotations()
                
            print(f"[Debug] 类别 '{class_name}' 删除完成")
            
        except Exception as e:
            QMessageBox.critical(self, tr("error"), f"删除类别时出错: {str(e)}")
            print(f"[Error] 删除类别时出错: {e}")
    
    def save_classes_to_file(self):
        """保存类别列表到classes.txt文件"""
        if hasattr(self, 'current_dir') and self.current_dir:
            class_path = os.path.join(self.current_dir, 'classes.txt')
            try:
                with open(class_path, 'w', encoding='utf-8') as f:
                    for class_name in self.classes:
                        f.write(f"{class_name}\n")
                print(f"[Debug] 类别列表已保存到: {class_path}")
            except Exception as e:
                print(f"[Error] 保存类别文件失败: {e}")
                QMessageBox.warning(self, tr("save_error"), f"保存类别文件失败: {str(e)}")

    def push_undo(self):
        idx = self.current_index
        if idx < 0: return

        if idx not in self.undo_stack:
            self.undo_stack[idx] = []
        
        import copy # Keep local as it's specific here
        # Create a deep copy of the list of BoundingBox objects
        current_bboxes_copy = [copy.deepcopy(b) for b in self.bboxes]
        self.undo_stack[idx].append(current_bboxes_copy)
        
        if len(self.undo_stack[idx]) > 30: # Limit undo stack size
            self.undo_stack[idx] = self.undo_stack[idx][-30:]
        
        self.redo_stack[idx] = [] # Clear redo stack for this image on new action

    def undo(self):
        idx = self.current_index
        if idx < 0 or idx not in self.undo_stack or not self.undo_stack[idx]:
            return
        
        import copy
        if idx not in self.redo_stack: self.redo_stack[idx] = []
        
        current_bboxes_for_redo = [copy.deepcopy(b) for b in self.bboxes]
        self.redo_stack[idx].append(current_bboxes_for_redo)
        
        self.bboxes = [copy.deepcopy(b) for b in self.undo_stack[idx].pop()] # Restore with deep copies
        
        self.selected_bboxes = set() # Clear selections after undo
        self.selected_bbox = None
        self.updateDisplay()
        self.updateColorLegend()
        if self.autosave: self.saveAnnotations()


    def redo(self):
        idx = self.current_index
        if idx < 0 or idx not in self.redo_stack or not self.redo_stack[idx]:
            return

        import copy
        if idx not in self.undo_stack: self.undo_stack[idx] = []

        current_bboxes_for_undo = [copy.deepcopy(b) for b in self.bboxes]
        self.undo_stack[idx].append(current_bboxes_for_undo)

        self.bboxes = [copy.deepcopy(b) for b in self.redo_stack[idx].pop()]

        self.selected_bboxes = set()
        self.selected_bbox = None
        self.updateDisplay()
        self.updateColorLegend()
        if self.autosave: self.saveAnnotations()


    def show_data_augmentation_dialog(self):
        if self.current_image is not None and self.image_list : # Ensure there are images to augment
            # Pass self (LabelingTool instance) as parent to the dialog
            dialog = DataAugmentationDialog(parent=self) 
            dialog.exec_()
        else:
            QMessageBox.warning(self, tr("warning"), tr("no_image_for_augmentation")) # Corrected

    def show_auto_label_dialog(self):
        """Show auto labeling dialog"""
        if not hasattr(self, 'current_dir') or not self.current_dir:
            QMessageBox.warning(self, tr("warning"), tr("error_current_dir_not_set"))
            return
            
        dialog = AutoLabelDialog(self)
        dialog.exec_()

    def show_auto_label_current_dialog(self):
        """显示当前图像自动标注对话框"""
        if not hasattr(self, 'current_dir') or not self.current_dir:
            QMessageBox.warning(self, tr("warning"), tr("error_current_dir_not_set"))
            return
            
        if self.current_image is None or self.current_index < 0:
            QMessageBox.warning(self, tr("warning"), tr("no_current_image"))
            return
            
        # 检查当前图像是否有标注
        if not self.bboxes:
            QMessageBox.warning(self, tr("warning"), tr("current_image_no_annotations"))
            return
            
        # 确认对话框
        reply = QMessageBox.question(
            self,
            tr("auto_label_current_title"),
            tr("auto_label_current_confirm"),
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.handle_auto_label_current_image()

    def handle_auto_label_current_image(self):
        """处理当前图像的自动标注"""
        try:
            print(f"[Debug] 开始当前图像自动标注")
            
            # 确保YOLOEWrapper已经初始化
            if self.yoloe_wrapper is None:
                print(f"[Error] {tr('yoloe_wrapper_not_initialized')}")
                QMessageBox.warning(self, tr("error"), tr("yoloe_wrapper_not_initialized"))
                return
            
            # 获取当前图像路径
            current_image_path = self.image_list[self.current_index]
            print(f"[Debug] 当前图像: {current_image_path}")
            
            # 构建当前图像的视觉提示数据
            visuals = self._build_current_image_visuals()
            if not visuals['bboxes'] or not visuals['cls']:
                QMessageBox.warning(self, tr("warning"), tr("failed_to_build_visuals"))
                return
            
            print(f"[Debug] 构建的视觉提示: {visuals}")
            
            # 创建并显示进度对话框
            progress_dialog = AutoLabelProgressDialog(self)
            progress_dialog.setup_batch_progress(1)  # 只有一张图片
            progress_dialog.show()
            
            # 更新进度：当前处理的图片
            progress_dialog.update_image_progress(0, os.path.basename(current_image_path))
            
            try:
                # 定义进度回调函数
                def progress_callback(step, total_steps, message):
                    progress_dialog.update_step_progress(step, total_steps, message)
                    # 处理Qt事件，保持界面响应
                    from PyQt5.QtWidgets import QApplication
                    QApplication.processEvents()
                
                # 调用自动标注，使用当前图像作为源图像和目标图像
                predictions = self.yoloe_wrapper.auto_label_with_vp(
                    current_image_path,  # 源图像（提示图像）
                    visuals,             # 视觉提示
                    current_image_path,  # 目标图像（同一张图像）
                    progress_callback=progress_callback
                )
                
                print(f"[Debug] 获得 {len(predictions)} 个预测结果")
                
                # 保存当前状态用于撤销
                self.push_undo()
                
                # 清除现有标注（可选，或者可以添加到现有标注中）
                reply = QMessageBox.question(
                    self,
                    tr("replace_annotations_title"),
                    tr("replace_annotations_msg"),
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.bboxes.clear()  # 清除现有标注
                
                # 添加新的预测结果
                added_count = 0
                for pred in predictions:
                    x, y, w, h = pred['bbox']  # bbox 格式是 [x, y, width, height]
                    label_idx = pred['class_id']
                    confidence = pred.get('confidence', 0.0)
                    
                    # 获取标签名称
                    if label_idx < len(self.classes):
                        label = self.classes[label_idx]
                    else:
                        label = f"class_{label_idx}"
                        # 如果类别不存在，添加到类别列表
                        if label not in self.classes:
                            self.classes.append(label)
                            self.class_combo.addItem(label)
                    
                    # 创建新的边界框
                    success = self.add_new_bounding_box(label, x, y, x + w, y + h, confidence)
                    if success:
                        added_count += 1
                
                # 显示完成状态
                progress_dialog.set_completed(1, 1)
                print(f"[Debug] 当前图像自动标注完成，添加了 {added_count} 个标注")
                
                # 更新显示
                self.updateDisplay()
                self.updateColorLegend()
                
                # 自动保存
                if self.autosave:
                    self.saveAnnotations()
                
                # 显示结果消息
                QMessageBox.information(
                    self,
                    tr("auto_label_current_title"),
                    tr("auto_label_current_success").format(added_count)
                )
                
            except Exception as e:
                error_msg = f"当前图像自动标注失败: {str(e)}"
                print(f"[Error] {error_msg}")
                progress_dialog.set_error(str(e))
                QMessageBox.critical(self, tr("error"), error_msg)
            
            finally:
                # 延迟关闭进度对话框
                QTimer.singleShot(3000, progress_dialog.close)  # 3秒后自动关闭
                
        except Exception as e:
            error_msg = f"处理当前图像自动标注时出错: {str(e)}"
            print(f"[Error] {error_msg}")
            QMessageBox.critical(self, tr("error"), error_msg)

    def _build_current_image_visuals(self):
        """构建当前图像的视觉提示数据"""
        bboxes_list = []
        cls_list = []
        
        print(f"[Debug] 构建当前图像的视觉提示，当前有 {len(self.bboxes)} 个标注")
        
        if not self.bboxes:
            return {'bboxes': [], 'cls': []}
        
        # 从当前图像的标注构建视觉提示
        bboxes = []
        clses = []
        
        for bbox in self.bboxes:
            # 转换为绝对坐标 [x1, y1, x2, y2]
            x1 = bbox.x
            y1 = bbox.y
            x2 = bbox.x + bbox.w
            y2 = bbox.y + bbox.h
            
            bboxes.append([x1, y1, x2, y2])
            
            # 获取类别ID
            try:
                if bbox.label in self.classes:
                    class_id = self.classes.index(bbox.label)
                    clses.append(class_id)
                    print(f"[Debug] 添加标注: 标签='{bbox.label}' -> 类别ID={class_id}")
                else:
                    print(f"[Debug] 跳过未知标签: {bbox.label}")
                    continue
            except ValueError as e:
                print(f"[Debug] 处理标签时出错: {e}")
                continue
        
        if bboxes and clses:
            bboxes_array = np.array(bboxes, dtype=np.float32)
            cls_array = np.array(clses, dtype=np.int64)
            
            bboxes_list.append(bboxes_array)
            cls_list.append(cls_array)
            
            print(f"[Debug] 构建的视觉提示: bboxes形状={bboxes_array.shape}, cls形状={cls_array.shape}")
        else:
            print(f"[Debug] 没有有效的标注用于构建视觉提示")
        
        return {'bboxes': bboxes_list, 'cls': cls_list}

    def add_new_bounding_box(self, label: str, xmin: float, ymin: float, xmax: float, ymax: float, score: float = None, difficult: bool = False):
        """Add new bounding box annotation
        
        Args:
            label: Annotation class name
            xmin: Top-left x coordinate
            ymin: Top-left y coordinate
            xmax: Bottom-right x coordinate
            ymax: Bottom-right y coordinate
            score: Confidence score (optional)
            difficult: Whether it's a difficult sample (optional)
        """
        if self.current_image is None:
            print("错误: 当前没有加载图像")
            return False
            
        try:
            # 确保坐标在图像范围内
            height, width = self.current_image.shape[:2]
            xmin = max(0, min(float(xmin), width - self.min_box_size))
            ymin = max(0, min(float(ymin), height - self.min_box_size))
            xmax = max(xmin + self.min_box_size, min(float(xmax), width))
            ymax = max(ymin + self.min_box_size, min(float(ymax), height))
            
            # 计算宽度和高度
            w = xmax - xmin
            h = ymax - ymin
            
            # 检查最小尺寸
            if w < self.min_box_size or h < self.min_box_size:
                print(f"警告: 边界框尺寸过小 ({w}x{h})，最小要求为 {self.min_box_size}x{self.min_box_size}")
                return False
            
            # 确保标签在类别列表中
            if label not in self.classes:
                self.classes.append(label)
                self.class_combo.addItem(label)
                # 保存更新后的类别列表
                if hasattr(self, 'current_dir') and self.current_dir:
                    class_path = os.path.join(self.current_dir, 'classes.txt')
                    try:
                        with open(class_path, 'w', encoding='utf-8') as f:
                            f.write("\n".join(self.classes) + "\n")
                    except Exception as e:
                        print(f"警告: 无法保存类别列表: {e}")
            
            # 创建新的边界框
            new_bbox = BoundingBox(xmin, ymin, w, h, label)
            
            # 保存当前状态用于撤销
            self.push_undo()
            
            # 添加到边界框列表
            self.bboxes.append(new_bbox)
            
            # 更新显示
            self.updateColorLegend()
            self.updateDisplay()
            
            # 如果启用了自动保存，保存标注
            if self.autosave:
                self.saveAnnotations()
                
            return True
            
        except Exception as e:
            print(f"添加边界框时出错: {e}")
            return False

    def handle_batch_auto_label_with_vp(self, prompt_image_paths, visuals, target_image_paths, target_indices):
        print(f"[Debug] {tr('batch_auto_label_start')}")
        print(f"[Debug] 提示图片数量: {len(prompt_image_paths)}")
        print(f"[Debug] 目标图片数量: {len(target_image_paths)}")
        
        # 确保YOLOEWrapper已经初始化
        if self.yoloe_wrapper is None:
            print(f"[Error] {tr('yoloe_wrapper_not_initialized')}")
            QMessageBox.warning(self, tr("error"), tr("yoloe_wrapper_not_initialized"))
            return
        
        # 创建并显示进度对话框
        progress_dialog = AutoLabelProgressDialog(self)
        progress_dialog.setup_batch_progress(len(target_image_paths))
        progress_dialog.show()
        
        # 处理结果统计
        success_count = 0
        error_count = 0
        
        try:
            for i, (idx, img_path) in enumerate(zip(target_indices, target_image_paths)):
                # 检查用户是否取消
                if progress_dialog.wasCanceled():
                    print("[Debug] 用户取消了自动标注")
                    break
                
                # 更新进度：当前处理的图片
                image_name = os.path.basename(img_path)
                progress_dialog.update_image_progress(i, image_name)
                
                print(f"\n[Debug] 正在处理图片 {i+1}/{len(target_image_paths)}: {img_path}")
                
                try:
                    # 定义进度回调函数
                    def progress_callback(step, total_steps, message):
                        progress_dialog.update_step_progress(step, total_steps, message)
                        # 处理Qt事件，保持界面响应
                        from PyQt5.QtWidgets import QApplication
                        QApplication.processEvents()
                    
                    # 调用自动标注，传入进度回调
                    bboxes = self.yoloe_wrapper.auto_label_with_vp(
                        prompt_image_paths, 
                        visuals, 
                        img_path,
                        progress_callback=progress_callback
                    )
                    
                    # 处理预测结果
                    bbox_objs = []
                    for pred in bboxes:
                        x, y, w, h = pred['bbox']  # bbox 格式是 [x, y, width, height]
                        label_idx = pred['class_id']
                        label = self.classes[label_idx] if label_idx < len(self.classes) else str(label_idx)
                        bbox_objs.append(BoundingBox(x, y, w, h, label))
           
                    print(f"[Debug] 更新缓存和当前索引: idx={idx}")
                    self.label_cache[idx] = bbox_objs
                    self.current_index = idx
                    self.bboxes = bbox_objs
                    
                    print(f"[Debug] 开始保存标注...")
                    save_result = self.saveYOLOAnnotation()
                    print(f"[Debug] 保存结果: {'成功' if save_result else '失败'}")
                    
                    if save_result:
                        success_count += 1
                    else:
                        error_count += 1
                    
                except Exception as e:
                    print(f"[Debug] 处理图片时出错: {str(e)}")
                    error_count += 1
                    continue
            
            # 显示完成状态
            if not progress_dialog.wasCanceled():
                progress_dialog.set_completed(success_count, len(target_image_paths))
                print(f"[Debug] 批量自动标注完成: 成功 {success_count}, 失败 {error_count}")
            
        except Exception as e:
            error_msg = f"批量自动标注失败: {str(e)}"
            print(error_msg)
            progress_dialog.set_error(str(e))
            
        finally:
            # 更新显示
            self.updateDisplay()
            self.updateColorLegend()
            
            # 延迟关闭进度对话框，让用户看到结果
            QTimer.singleShot(3000, progress_dialog.close)  # 3秒后自动关闭

    def on_zoom_slider_changed(self, value):
        # 将滑动条的值2-8转换为实际的放大倍率2.0x-4.0x
        # 映射关系：2->2.0x, 4->2.67x, 6->3.33x, 8->4.0x
        zoom_factor = 2.0 + (value - 2) * (2.0 / 6)  # 线性映射到2.0-4.0
        self.zoom_value_label.setText(f"{zoom_factor:.1f}x")
        if self.magnifier:
            self.magnifier.set_zoom_factor(zoom_factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 保持右下角
        if hasattr(self, 'magnifier_controls'):
            x = self.width() - self.magnifier_controls.width() - 20
            y = self.height() - self.magnifier_controls.height() - 20
            self.magnifier_controls.move(x, y)

# Entry point will be handled by __main__.py
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = LabelingTool()
#     ex.show()
#     sys.exit(app.exec_()) 