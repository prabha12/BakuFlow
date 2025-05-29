import numpy as np
import os
# 假设的导入路径，需要根据您项目中实际的 Shape 和 AnnotationLoader/Reader 的位置进行调整
# from labelimg.libs.shape import Shape # 示例
# from PyQt5.QtCore import QPointF # 示例，如果Shape使用QPointF
from labelimg.core.localization import tr

class PromptConverter:
    def __init__(self, class_name_to_id_map: dict, annotation_loader):
        """初始化提示转换器
        
        Args:
            class_name_to_id_map: 类别名称到ID的映射字典
            annotation_loader: 标注加载器实例
        """
        self.class_name_to_id_map = class_name_to_id_map
        self.annotation_loader = annotation_loader
        
    def convert(self, prompt_image_paths: list[str]) -> dict:
        """将标注转换为YOLOE所需的格式
        
        Args:
            prompt_image_paths: 提示图像路径列表
            
        Returns:
            dict: 包含边界框和类别的字典
        """
        bboxes_list = []
        cls_list = []
        
        for image_path in prompt_image_paths:
            # 获取标注文件路径
            annot_path = self._get_annotation_file_path(image_path)
            if not annot_path:
                print(tr("warning_no_annotation_file").format(image_path))
                continue
                
            # 加载标注
            try:
                # 使用标注加载器加载标注
                shapes = self._load_annotations_for_image(annot_path)
                if not shapes:
                    print(tr("warning_no_valid_annotations").format(annot_path))
                    continue
                    
                # 转换标注格式
                image_bboxes = []
                image_cls = []
                
                for shape in shapes:
                    # 获取类别ID
                    class_name = shape.label
                    if class_name not in self.class_name_to_id_map:
                        print(tr("warning_unknown_class").format(class_name))
                        continue
                        
                    class_id = self.class_name_to_id_map[class_name]
                    
                    # 获取边界框坐标
                    try:
                        x, y, w, h = shape.x, shape.y, shape.w, shape.h
                        x2, y2 = x + w, y + h
                        
                        # 验证坐标有效性
                        if not (0 <= x < x2 and 0 <= y < y2):
                            print(tr("warning_invalid_bbox_coords").format(x, y, x2, y2))
                            continue
                            
                        # 添加到列表
                        image_bboxes.append([float(x), float(y), float(x2), float(y2)])
                        image_cls.append(int(class_id))
                    except AttributeError:
                        print(tr("warning_cannot_get_bbox_coords"))
                        continue
                
                if image_bboxes:
                    bboxes_list.append(np.array(image_bboxes, dtype=np.float32))
                    cls_list.append(np.array(image_cls, dtype=np.int32))
                    
            except Exception as e:
                print(tr("error_processing_annotation").format(annot_path, e))
                continue
        
        return {
            "bboxes": bboxes_list,
            "cls": cls_list
        }

    def _get_annotation_file_path(self, image_path: str) -> str | None:
        """根据图像路径推断其对应的标注文件路径
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str | None: 标注文件路径，如果未找到则返回 None
        """
        # 尝试几种常见的标注文件后缀
        base, _ = os.path.splitext(image_path)
        possible_exts = ['.xml', '.txt', '.json']  # 支持更多格式
        
        for ext in possible_exts:
            annot_path = base + ext
            if os.path.exists(annot_path):
                return annot_path
        
        return None

    def _load_annotations_for_image(self, annotation_file_path: str) -> list:
        """加载单个图像的标注数据
        
        Args:
            annotation_file_path: 标注文件路径
            
        Returns:
            list: 标注形状列表
        """
        try:
            # 调用标注加载器加载标注
            result = self.annotation_loader.load(annotation_file_path)
            
            # 处理加载结果
            if isinstance(result, tuple) and len(result) >= 2:
                shapes = result[1]  # 假设第二个元素是shapes列表
            else:
                shapes = result
                
            if not shapes:
                return []
                
            # 验证shapes格式
            valid_shapes = []
            for shape in shapes:
                if hasattr(shape, 'label') and hasattr(shape, 'x') and hasattr(shape, 'y') and hasattr(shape, 'w') and hasattr(shape, 'h'):
                    valid_shapes.append(shape)
                else:
                    print(f"警告: 无效的标注形状格式: {shape}")
                    
            return valid_shapes
            
        except Exception as e:
            print(f"加载标注文件失败: {e}")
            return [] 