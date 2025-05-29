import os
import traceback

# 导入必要的模块 - 根据您的labelimg项目结构调整路径
from labelimg.io.prompt_converter import PromptConverter
from labelimg.inference.autolableing import YOLOEWrapper
from labelimg.core.localization import tr

# Qt相关的导入，通常 labelImg 使用 PyQt5 或 PySide
# from PyQt5.QtCore import QPointF # 示例
# from labelimg.libs.shape import Shape # 示例，labelImg的Shape对象

class LabelController:
    def __init__(self, main_window_ref):
        """初始化标签控制器
        
        Args:
            main_window_ref: labelImg 主窗口 (MainWindow) 的引用
        """
        self.main_window = main_window_ref
        self._model = None  # 延迟初始化 YOLOE 模型
        
    def _get_model(self) -> YOLOEWrapper:
        """获取或初始化 YOLOE 模型
        
        Returns:
            YOLOEWrapper: YOLOE 模型包装器实例
        """
        if self._model is None:
            # 从主窗口获取动态加载的类别名称
            class_names = getattr(self.main_window, 'classes', [])
            if not class_names:
                print("[Warning] 主窗口中没有加载类别，使用空的类别列表")
            print(f"[Debug] 使用主窗口的类别初始化YOLOE模型: {class_names}")
            self._model = YOLOEWrapper(class_names=class_names)
        return self._model
        
    def _get_class_mapping_and_reverse(self) -> tuple[dict[str, int], dict[int, str]]:
        """从主窗口获取类别列表并创建正向和反向映射
        
        Returns:
            tuple: (类别名到ID的映射, ID到类别名的映射)
        """
        name_to_id = {}
        id_to_name = {}
        
        if hasattr(self.main_window, 'label_hist') and self.main_window.label_hist:
            class_names = self.main_window.label_hist
            name_to_id = {name: i for i, name in enumerate(class_names)}
            id_to_name = {i: name for i, name in enumerate(class_names)}
        else:
            error_msg = tr("error_no_class_list")
            print(error_msg)
            self._show_status_message(error_msg)
            
        return name_to_id, id_to_name

    def _get_actual_annotation_loader(self):
        """获取标注加载器实例
        
        Returns:
            object: 标注加载器实例
        """
        # 方案1: 如果主窗口直接提供了兼容的加载方法
        if hasattr(self.main_window, 'load_annotations_for_prompt_converter'): 
            return self.main_window.load_annotations_for_prompt_converter
        
        # 方案2: 使用 labelImg 内置的 Reader
        class GenericAnnotationLoaderWrapper:
            def __init__(self, main_window_instance):
                self.mw = main_window_instance
                self.reader_load_method = None
                
                # 尝试获取可用的标注读取器
                if hasattr(self.mw, 'pascal_voc_reader') and hasattr(self.mw.pascal_voc_reader, 'loadFile'):
                    self.reader_load_method = self.mw.pascal_voc_reader.loadFile
                    print("Info: 使用 Pascal VOC 读取器")
                elif hasattr(self.mw, 'yolo_reader') and hasattr(self.mw.yolo_reader, 'loadFile'):
                    self.reader_load_method = self.mw.yolo_reader.loadFile
                    print("Info: 使用 YOLO 读取器")
                else:
                    print("警告: 未找到兼容的标注读取器")
                    
            def load(self, annotation_file_path: str):
                """加载标注文件
                
                Args:
                    annotation_file_path: 标注文件路径
                    
                Returns:
                    tuple: (文件名, 形状列表, 图像数据, ...)
                """
                if self.reader_load_method:
                    try:
                        return self.reader_load_method(annotation_file_path)
                    except Exception as e:
                        print(f"加载标注文件失败: {e}")
                        return (annotation_file_path, [], None)
                return (annotation_file_path, [], None)
                
        return GenericAnnotationLoaderWrapper(self.main_window)
        
    def _show_status_message(self, message: str, timeout: int = 0):
        """显示状态栏消息
        
        Args:
            message: 要显示的消息
            timeout: 消息显示时间（毫秒），0表示一直显示
        """
        if hasattr(self.main_window, 'statusBar') and callable(self.main_window.statusBar):
            self.main_window.statusBar().showMessage(message, timeout)
            
    def _validate_image_paths(self, prompt_paths: list[str], target_path: str) -> bool:
        """验证图像路径
        
        Args:
            prompt_paths: 提示图像路径列表
            target_path: 目标图像路径
            
        Returns:
            bool: 所有路径是否有效
        """
        if not os.path.exists(target_path):
            self._show_status_message(tr("error_target_image_not_exist").format(target_path))
            return False
            
        for path in prompt_paths:
            if not os.path.exists(path):
                self._show_status_message(tr("error_prompt_image_not_exist").format(path))
                return False
                
        return True

    def handle_auto_label_with_vp(self, prompt_image_paths: list[str], target_image_path: str):
        """处理视觉提示自动标注
        
        Args:
            prompt_image_paths: 提示图像路径列表
            target_image_path: 目标图像路径
        """
        try:
            self._show_status_message(tr("processing_vp_auto_label"))
            print("[Debug] 开始处理视觉提示自动标注...")
            
            # 验证输入
            if not self._validate_image_paths(prompt_image_paths, target_image_path):
                return
                
            # 获取类别映射和标注加载器
            class_mapping, id_to_name = self._get_class_mapping_and_reverse()
            if not class_mapping:
                return
                
            print(f"[Debug] 类别映射: {class_mapping}")
            print(f"[Debug] ID到类别映射: {id_to_name}")
            
            annotation_loader = self._get_actual_annotation_loader()
            
            # 初始化转换器和模型
            converter = PromptConverter(class_mapping, annotation_loader)
            model = self._get_model()
            
            # 转换标注
            self._show_status_message(tr("converting_annotations"))
            print("[Debug] 正在转换标注...")
            visuals = converter.convert(prompt_image_paths)
            print(f"[Debug] 转换后的visuals: {visuals}")
            
            # 执行预测
            self._show_status_message(tr("executing_prediction"))
            print("[Debug] 正在执行预测...")
            predictions = model.auto_label_with_vp(
                prompt_image_paths[0],
                visuals,
                target_image_path,
                conf_threshold=0.25
            )
            print(f"[Debug] 预测结果: {predictions}")
            
            # 添加预测结果到主窗口
            skipped = 0
            added = 0
            
            for pred in predictions:
                try:
                    # 获取类别名称
                    class_id = pred['class_id']
                    if class_id not in id_to_name:
                        print(tr("warning_unknown_class_id").format(class_id))
                        skipped += 1
                        continue
                        
                    class_name = id_to_name[class_id]
                    print(f"[Debug] 处理预测框: class_id={class_id}, class_name={class_name}, bbox={pred['bbox']}")
                    
                    # 添加边界框
                    result = self.main_window.add_new_bounding_box(
                        pred['bbox'],
                        class_name,
                        pred['confidence']
                    )
                    print(f"[Debug] 添加边界框结果: {result}")
                    added += 1
                    
                except Exception as e:
                    print(tr("error_adding_bbox").format(e))
                    skipped += 1
            
            # 更新状态
            if added > 0:
                print(f"[Debug] 尝试保存标注，当前图片: {target_image_path}")
                # 保存标注
                save_result = self.main_window.saveAnnotations()
                print(f"[Debug] 保存标注结果: {save_result}")
                
                self._show_status_message(tr("auto_label_completed").format(added) + 
                                        (tr("skipped_invalid_labels").format(skipped) if skipped > 0 else ""))
            else:
                self._show_status_message(tr("auto_label_completed_no_valid_labels"))
            
        except Exception as e:
            error_msg = tr("auto_label_failed").format(str(e))
            print(error_msg)
            self._show_status_message(error_msg)
            raise

    def handle_batch_auto_label_with_vp(self, prompt_image_paths: list[str], target_dir: str):
        """批量處理視覺提示自動標註
        
        Args:
            prompt_image_paths: 提示圖像路徑列表
            target_dir: 目標圖像目錄
        """
        try:
            self._show_status_message("開始批量自動標註...")
            print("[Debug] 开始批量自动标注...")
            
            # 驗證輸入
            if not os.path.exists(target_dir):
                self._show_status_message(f"錯誤: 目標目錄不存在: {target_dir}")
                return
            
            for path in prompt_image_paths:
                if not os.path.exists(path):
                    self._show_status_message(f"錯誤: 提示圖像不存在: {path}")
                    return
            
            # 獲取所有未標記的圖像
            unlabeled_images = self._get_unlabeled_images(target_dir)
            if not unlabeled_images:
                self._show_status_message("沒有找到需要標註的圖像")
                return
            
            print(f"[Debug] 找到 {len(unlabeled_images)} 张未标注图片")
            
            # 獲取類別映射和標註加載器
            class_mapping, id_to_name = self._get_class_mapping_and_reverse()
            if not class_mapping:
                return
            
            print(f"[Debug] 类别映射: {class_mapping}")
            print(f"[Debug] ID到类别映射: {id_to_name}")
            
            annotation_loader = self._get_actual_annotation_loader()
            
            # 初始化轉換器和模型
            converter = PromptConverter(class_mapping, annotation_loader)
            model = self._get_model()
            
            # 轉換標註
            self._show_status_message("正在轉換標註...")
            print("[Debug] 正在转换标注...")
            visuals = converter.convert(prompt_image_paths)
            print(f"[Debug] 转换后的visuals: {visuals}")
            
            # 批量處理圖像
            total_images = len(unlabeled_images)
            total_boxes = 0
            skipped_images = 0
            
            for i, image_path in enumerate(unlabeled_images, 1):
                try:
                    self._show_status_message(f"正在處理第 {i}/{total_images} 張圖像: {os.path.basename(image_path)}")
                    print(f"[Debug] 处理第 {i}/{total_images} 张图片: {image_path}")
                    
                    # 執行預測
                    predictions = model.auto_label_with_vp(
                        prompt_image_paths[0],
                        visuals,
                        image_path,
                        conf_threshold=0.25
                    )
                    print(f"[Debug] 预测结果: {predictions}")
                    
                    # 添加預測結果
                    boxes_added = 0
                    for pred in predictions:
                        try:
                            # 獲取類別名稱
                            class_id = pred['class_id']
                            if class_id not in id_to_name:
                                print(f"警告: 未知類別ID {class_id}")
                                continue
                            
                            class_name = id_to_name[class_id]
                            print(f"[Debug] 处理预测框: class_id={class_id}, class_name={class_name}, bbox={pred['bbox']}")
                            
                            # 添加邊界框
                            result = self.main_window.add_new_bounding_box(
                                pred['bbox'],
                                class_name,
                                pred['confidence']
                            )
                            print(f"[Debug] 添加边界框结果: {result}")
                            boxes_added += 1
                            
                        except Exception as e:
                            print(f"添加邊界框時出錯: {e}")
                            continue
                    
                    if boxes_added > 0:
                        total_boxes += boxes_added
                        print(f"[Debug] 尝试保存标注，当前图片: {image_path}")
                        # 保存当前图片的标注
                        save_result = self.main_window.saveAnnotations()
                        print(f"[Debug] 保存标注结果: {save_result}")
                    else:
                        skipped_images += 1
                        print(f"[Debug] 跳过图片 {image_path}，没有添加任何边界框")
                    
                except Exception as e:
                    print(f"處理圖像 {image_path} 時出錯: {e}")
                    skipped_images += 1
                    continue
            
            # 更新狀態
            status_msg = f"批量自動標註完成: 處理了 {total_images} 張圖像"
            if total_boxes > 0:
                status_msg += f"，添加了 {total_boxes} 個標註"
            if skipped_images > 0:
                status_msg += f"，跳過了 {skipped_images} 張圖像"
            self._show_status_message(status_msg)
            print(f"[Debug] {status_msg}")
            
        except Exception as e:
            error_msg = f"批量自動標註失敗: {str(e)}"
            print(error_msg)
            self._show_status_message(error_msg)
            raise
        
    def _get_unlabeled_images(self, target_dir: str) -> list[str]:
        """獲取未標記的圖像列表
        
        Args:
            target_dir: 目標目錄
            
        Returns:
            list[str]: 未標記圖像的路徑列表
        """
        unlabeled_images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for filename in os.listdir(target_dir):
            # 檢查是否為圖像文件
            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_extensions:
                continue
            
            image_path = os.path.join(target_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            # 檢查是否存在對應的標註文件
            has_annotation = False
            for annot_ext in ['.xml', '.txt', '.json']:
                annot_path = os.path.join(target_dir, base_name + annot_ext)
                if os.path.exists(annot_path):
                    has_annotation = True
                    break
            
            if not has_annotation:
                unlabeled_images.append(image_path)
            
        return unlabeled_images 

    def saveYOLOAnnotation(self):
        img_path = self.image_list[self.current_index]
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        print(f"[Debug] 尝试保存标注到: {txt_path}")
        print(f"[Debug] 当前边界框数量: {len(self.bboxes)}")
        try:
            height, width = self.current_image.shape[:2]
            valid_boxes = 0
            with open(txt_path, 'w') as f:
                for bbox in self.bboxes:
                    print(f"[Debug] 处理边界框: x={bbox.x}, y={bbox.y}, w={bbox.w}, h={bbox.h}, label={bbox.label}")
                    if (bbox.w < self.min_box_size or bbox.h < self.min_box_size or 
                        bbox.x < 0 or bbox.y < 0 or 
                        bbox.x + bbox.w > width or bbox.y + bbox.h > height):
                        print(f"[Debug] 边界框被过滤: 尺寸或位置无效")
                        continue
                    try:
                        class_idx = self.classes.index(bbox.label)
                    except ValueError:
                        print(f"[Debug] 边界框被过滤: 类别 {bbox.label} 不在类别列表中")
                        continue
                    x_center = (bbox.x + bbox.w/2) / width
                    y_center = (bbox.y + bbox.h/2) / height
                    w_norm = bbox.w / width
                    h_norm = bbox.h / height
                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    valid_boxes += 1
        except Exception as e:
            print(f"保存标注失败: {str(e)}") 