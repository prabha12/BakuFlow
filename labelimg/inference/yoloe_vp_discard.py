import sys
import os
import torch
import torch.nn.functional as F
import numpy as np # 確保 numpy 已導入
import traceback # 用於異常處理

# 添加 yoloe 目錄到 Python 路徑
# __file__ 可能在某些環境 (如 Jupyter notebook) 中未定義，請注意
try:
    yoloe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yoloe'))
    if yoloe_path not in sys.path:
        sys.path.insert(0, yoloe_path)  # 使用 insert(0, ...) 確保優先使用 yoloe 目錄中的包
        print(f"Added yoloe path: {yoloe_path}")
except NameError:
    print("[Warning] __file__ not defined, skipping yoloe_path modification. Ensure 'yoloe' is in PYTHONPATH if needed.")

print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# 修改導入路徑
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

# --- Placeholder for actual YOLOE model and predictor (user's original placeholders) ---
class MockYOLOEModel: # This is the user's top-level mock, not directly used by YOLOEWrapper unless YOLOE import fails
    def predict(self, images_or_target, prompts=None, predictor=None, conf=None):
        print(f"MockYOLOEModel.predict called with: images_or_target={images_or_target}, prompts_provided={prompts is not None}, predictor_exists={predictor is not None}, conf={conf}")
        if predictor: 
            print("MockYOLOEModel: VP-Seg predict call (通常用于设置上下文或初步预测).")
            return [] 
        elif conf is not None: 
            print("MockYOLOEModel: Final predict call for target image.")
            return [
                (50.0, 50.0, 150.0, 150.0, 0.9, 0), 
                (70.0, 70.0, 180.0, 180.0, 0.85, 1),
            ]
        return []

class MockYOLOEVPSegPredictor: # User's top-level mock
    def __init__(self):
        print("MockYOLOEVPSegPredictor initialized")
# --- End of Placeholder ---
# --- Helper Function: Calculate IoU (user's original helper) ---
def calculate_iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_width = x2_inter - x1_inter
    inter_height = y2_inter - y1_inter
    if inter_width <= 0 or inter_height <= 0:
        return 0.0
    inter_area = inter_width * inter_height
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    iou = inter_area / union_area
    return iou

class YOLOEWrapper:
    def __init__(self, model_path="/Users/patrick/Desktop/labeling/pretrain/yoloe-11l-seg.pt", 
                 class_names=None):
        """初始化 YOLOE 模型
        
        Args:
            model_path: YOLOE 模型路径
            class_names: 真实的类别名称列表，如 ['energydrink', 'pepsiMax', ...]
        """
        try:
            self.model = YOLOE(model_path) 
        except NameError:
            print("[CRITICAL ERROR] The 'YOLOE' class from 'ultralytics' is not defined or imported correctly.")
            print("[INFO] Using internal MockYOLOE_for_init for YOLOEWrapper. Define YOLOE and YOLOEVPSegPredictor for real operation.")
            class MockYOLOE_for_init:
                def __init__(self, path): self.predictor = None; print(f"Internal MockYOLOE_for_init loaded from {path}")
                def predict(self, *args, **kwargs): 
                    print("[Internal MockYOLOE_for_init.predict] called")
                    if kwargs.get('return_vpe'): # Simulate VPE generation part
                        if isinstance(kwargs.get('predictor'), type): # if predictor class is passed
                             self.predictor = kwargs['predictor']() 
                        elif kwargs.get('predictor') is not None: # if instance is passed
                             self.predictor = kwargs.get('predictor')

                        if self.predictor:
                            num_prompt_images = len(args[0]) if isinstance(args[0], list) else 1
                            max_objects = 1
                            prompts_data = kwargs.get('prompts')
                            if prompts_data and prompts_data.get('bboxes'):
                                valid_bbox_arrays = [b_arr for b_arr in prompts_data['bboxes'] if isinstance(b_arr, np.ndarray) and b_arr.ndim == 2 and b_arr.shape[0] > 0]
                                if valid_bbox_arrays:
                                    max_objects = max(b_arr.shape[0] for b_arr in valid_bbox_arrays)
                            max_objects = max(1, max_objects)
                            self.predictor.vpe = torch.rand(num_prompt_images, max_objects, 512)
                    # Simulate detection results
                    class MockBox_internal:
                        def __init__(self, xyxy, conf, cls_id): self.xyxy = torch.tensor([xyxy],dtype=torch.float); self.conf = torch.tensor([conf],dtype=torch.float); self.cls = torch.tensor([cls_id],dtype=torch.float)
                    class MockResults_internal: boxes = [MockBox_internal([10,10,50,50],0.8,0), MockBox_internal([60,60,100,100],0.75,1)] if not kwargs.get('return_vpe') else []
                    return [MockResults_internal()]
                def set_classes(self, *args, **kwargs): print("[Internal MockYOLOE_for_init.set_classes] called")
            self.model = MockYOLOE_for_init(model_path)
        
        # 真实类别名称（按ID索引）
        self.true_class_names = class_names or []
        
        # 类别映射管理
        self.class_mapping = {}  # 类别ID -> 类别名称的映射
        self.reverse_class_mapping = {}  # 类别名称 -> 类别ID的映射
        self.class_id_to_index = {}  # 类别ID -> 在initial_object_set中的索引
        self.index_to_class_id = {}  # 在initial_object_set中的索引 -> 类别ID
        self.embedding_dim = 512  # VPE特征维度，默认512
        
        # VPE相关属性
        self.final_refined_embeddings_tensor = None
        self.initial_object_set = None
        self.num_classes = 0  # 初始化为0，会根据实际类别数量更新

        self.visual_prompts = {
            'image_paths': [],
            'bboxes': [],      
            'cls': [],         
            'confidences': [], 
            'is_initial': []   
        }
        self.max_prompts = 15
        self.conf_thresh_for_update = 1

    def _validate_and_update_class_mapping(self, new_cls_arrays):
        """验证并更新类别映射，严格保持原始出现顺序
        
        Args:
            new_cls_arrays: 新的类别ID数组列表
        """
        # 按照第一次出现的顺序收集类别ID（不排序）
        seen_cls_ids = set()
        ordered_unique_cls = []
        
        # 遍历所有数组，按出现顺序记录类别ID
        for cls_arr in new_cls_arrays:
            if isinstance(cls_arr, np.ndarray) and cls_arr.size > 0:
                for cls_id in cls_arr.astype(int):
                    if cls_id not in seen_cls_ids:
                        seen_cls_ids.add(cls_id)
                        ordered_unique_cls.append(cls_id)
        
        # 更新类别映射，按原始出现顺序（绝不排序）
        classes_added = []
        for cls_id in ordered_unique_cls:  # 使用原始顺序，不用sorted!
            if cls_id not in self.class_mapping:
                # 使用真实的类别名称（如果有的话）
                if (self.true_class_names and 
                    0 <= cls_id < len(self.true_class_names)):
                    class_name = self.true_class_names[cls_id]
                else:
                    class_name = f"object_{cls_id}"
                
                self.class_mapping[cls_id] = class_name
                self.reverse_class_mapping[class_name] = cls_id
                classes_added.append(cls_id)
        
        # 更新核心类别属性以确保一致性
        if self.class_mapping:
            # 使用排序后的类别ID列表作为权威顺序
            # 这与代码中其他部分（如备用VPE创建）的行为一致
            authoritative_class_ids = sorted(list(self.class_mapping.keys()))
            
            self.num_classes = len(authoritative_class_ids)
            self.initial_object_set = [self.class_mapping[class_id] for class_id in authoritative_class_ids]
            self.index_to_class_id = {idx: class_id for idx, class_id in enumerate(authoritative_class_ids)}
            self.class_id_to_index = {class_id: idx for idx, class_id in enumerate(authoritative_class_ids)}
        else:
            self.num_classes = 0
            self.initial_object_set = []
            self.class_id_to_index = {}
            self.index_to_class_id = {}
        
        if classes_added:
            print(f"[Info] 新增类别: {classes_added}, 当前总类别数: {self.num_classes}")
        
        # print(f"[Debug] 当前类别映射: {self.class_mapping}")
        # print(f"[Debug] 索引映射: {self.index_to_class_id}")
        return True

    def _validate_class_consistency(self):
        """验证类别映射的一致性"""
        if len(self.class_mapping) != len(self.reverse_class_mapping):
            print("[ERROR] 类别映射不一致！正向和反向映射长度不匹配")
            return False
        
        if len(self.class_mapping) != self.num_classes:
            print(f"[ERROR] 类别数量不一致！映射中有{len(self.class_mapping)}个类别，但num_classes={self.num_classes}")
            return False
        
        if self.initial_object_set and len(self.initial_object_set) != self.num_classes:
            print(f"[ERROR] 对象集合长度不一致！对象集合有{len(self.initial_object_set)}个，但num_classes={self.num_classes}")
            return False
        
        return True

    def _update_vpe_with_new_annotations(self):
        """使用新标注更新VPE (Overwrite Strategy)"""
        # 缓存当前的VPE相关状态，以备在保持现有VPE时恢复
        cached_num_classes = self.num_classes
        cached_initial_object_set = self.initial_object_set.copy() if self.initial_object_set is not None else []
        cached_index_to_class_id = self.index_to_class_id.copy()
        cached_class_id_to_index = self.class_id_to_index.copy()
        cached_vpe_tensor = self.final_refined_embeddings_tensor # Tensor本身不需要深拷贝，因为如果替换，也是整个替换

        if not self.visual_prompts['image_paths']:
            print("[Info] _update_vpe: No visual prompts available. Skipping VPE update.")
            return True 
            
        # print("\\n[Debug] _update_vpe: Current visual prompts for VPE calculation:")
        for i in range(len(self.visual_prompts['image_paths'])):
            num_objects_in_prompt = len(self.visual_prompts['cls'][i]) if isinstance(self.visual_prompts['cls'][i], np.ndarray) else 'N/A'
            # print(f"  {i+1}. {os.path.basename(self.visual_prompts['image_paths'][i])} (类别: {self.visual_prompts['cls'][i]}, Objects: {num_objects_in_prompt}, 置信度: {self.visual_prompts['confidences'][i]:.3f}, Initial: {self.visual_prompts['is_initial'][i]})")
        
        # 首先更新类别映射
        self._validate_and_update_class_mapping(self.visual_prompts['cls'])
        
        # 验证类别一致性
        if not self._validate_class_consistency():
            print("[ERROR] _update_vpe: 类别一致性验证失败")
            return False
        
        new_visuals = {
            'bboxes': self.visual_prompts['bboxes'],
            'cls': self.visual_prompts['cls']
        }
        
        self.model.predictor = None 
        try:
            _ = YOLOEVPSegPredictor 
        except NameError:
            print("[CRITICAL ERROR] _update_vpe: 'YOLOEVPSegPredictor' class from 'ultralytics' is not defined or imported.")
            print("[INFO] Using internal MockYOLOEVPSegPredictor for this run.")
            class MockYOLOEVPSegPredictor_for_update: 
                 def __init__(self, model=None, overrides=None, _callbacks=None): self.vpe = None
            globals()['YOLOEVPSegPredictor'] = MockYOLOEVPSegPredictor_for_update

        try:
            self.model.predict(
                self.visual_prompts['image_paths'], 
                prompts=new_visuals,                
                predictor=YOLOEVPSegPredictor, 
                return_vpe=True,
                verbose=False 
            )
        except Exception as model_predict_error:
            print(f"[Warning] _update_vpe: VPE更新时模型预测出错: {str(model_predict_error)}")
            print("[Info] _update_vpe: 尝试使用现有VPE或创建备用VPE")
            
            # 如果已有VPE且形状合适，保持不变，并恢复与之匹配的旧映射状态
            if (cached_vpe_tensor is not None and 
                cached_vpe_tensor.shape[1] == cached_num_classes and # 使用缓存的num_classes进行比较
                len(cached_initial_object_set) == cached_num_classes): # 确保缓存状态本身是自洽的
                print("[Info] _update_vpe: 成功恢复并保持现有VPE及其映射")
                self.final_refined_embeddings_tensor = cached_vpe_tensor
                self.initial_object_set = cached_initial_object_set
                self.num_classes = cached_num_classes
                self.index_to_class_id = cached_index_to_class_id
                self.class_id_to_index = cached_class_id_to_index
                return True
            
            # 否则创建新的零VPE (此时 self.num_classes, self.initial_object_set 等已被 _validate_and_update_class_mapping 更新为基于完整类别映射的权威状态)
            print("[Info] _update_vpe: 创建新的零VPE作为备用")
            if self.num_classes == 0 or self.embedding_dim == 0 : # 如果没有类别或维度为0，则无法创建
                 print(f"[Error] _update_vpe:无法创建零VPE，因为 num_classes ({self.num_classes}) 或 embedding_dim ({self.embedding_dim}) 为0")
                 # 尝试恢复到之前的状态，如果可能的话
                 self.final_refined_embeddings_tensor = cached_vpe_tensor
                 self.initial_object_set = cached_initial_object_set
                 self.num_classes = cached_num_classes
                 self.index_to_class_id = cached_index_to_class_id
                 self.class_id_to_index = cached_class_id_to_index
                 return False # 指示VPE更新失败

            processed_vpe_for_update = torch.zeros((1, self.num_classes, self.embedding_dim), 
                                                   dtype=torch.float32, 
                                                   device='cpu' if not torch.cuda.is_available() else 'cuda')
            self.final_refined_embeddings_tensor = processed_vpe_for_update
            # initial_object_set, num_classes, index_to_class_id 等已由 _validate_and_update_class_mapping 设置为与 self.num_classes 一致
            
            # print(f"[Debug] _update_vpe: 备用VPE创建完成，形状: {self.final_refined_embeddings_tensor.shape}")
            # print(f"[Debug] _update_vpe: 备用VPE索引映射: {self.index_to_class_id}")
            # print(f"[Debug] _update_vpe: 备用VPE对象集合: {self.initial_object_set}")
            return True
        
        if self.model.predictor is None or not hasattr(self.model.predictor, 'vpe') or self.model.predictor.vpe is None:
            print("[ERROR] _update_vpe: New VPE calculation failed. Predictor or VPE tensor is None.")
            
            # 🔧 修复：即使VPE计算失败，也要确保索引映射正确设置
            if self.class_mapping:
                print("[Info] _update_vpe: VPE计算失败，但仍设置索引映射以避免预测时出现unknown")
                sorted_class_ids = sorted(self.class_mapping.keys())
                self.index_to_class_id = {idx: class_id for idx, class_id in enumerate(sorted_class_ids)}
                self.class_id_to_index = {class_id: idx for idx, class_id in enumerate(sorted_class_ids)}
                self.initial_object_set = [self.class_mapping[class_id] for class_id in sorted_class_ids]
                self.num_classes = len(sorted_class_ids)
                
                # 创建一个最小的零VPE以避免后续错误
                if self.embedding_dim > 0:
                    self.final_refined_embeddings_tensor = torch.zeros(
                        (1, self.num_classes, self.embedding_dim), 
                        dtype=torch.float32, 
                        device='cpu' if not torch.cuda.is_available() else 'cuda'
                    )
                    # print(f"[Debug] _update_vpe: 创建最小零VPE，形状: {self.final_refined_embeddings_tensor.shape}")
                    # print(f"[Debug] _update_vpe: 设置索引映射: {self.index_to_class_id}")
                    # print(f"[Debug] _update_vpe: 设置对象集合: {self.initial_object_set}")
                    return True
                else:
                    print("[ERROR] _update_vpe: 嵌入维度为0，无法创建VPE")
            else:
                print("[ERROR] _update_vpe: 没有类别映射，无法设置索引映射")
            
            return False
            
        raw_new_vpe = self.model.predictor.vpe 
        # print(f"[Debug] _update_vpe: Raw new VPE shape from predictor: {raw_new_vpe.shape}")
        
        # 更新嵌入维度
        if raw_new_vpe.numel() > 0:
            self.embedding_dim = raw_new_vpe.shape[-1]
        
        processed_vpe_for_update = None
        
        # 如果没有类别映射，清空VPE
        if not self.class_mapping:
            print("[Warning] _update_vpe: No class mapping available. VPE will be cleared.")
            self.final_refined_embeddings_tensor = None
            self.initial_object_set = []
            self.num_classes = 0
            return True 
        
        # 使用类别映射中的实际类别数量
        num_target_classes = self.num_classes
        # print(f"[Debug] _update_vpe: 目标类别数量: {num_target_classes}")
        # print(f"[Debug] _update_vpe: 类别映射: {self.class_mapping}")
        
        if raw_new_vpe.numel() == 0:
             if self.embedding_dim == 0:
                print(f"[ERROR] _update_vpe: 嵌入维度为0，无法创建VPE. Raw VPE shape: {raw_new_vpe.shape}")
                return False
             else: # numel is 0 but embedding_dim is valid
                print(f"[Warning] _update_vpe: Raw new VPE is empty {raw_new_vpe.shape}, 为所有类别创建零向量VPE.")
                processed_vpe_for_update = torch.zeros((1, num_target_classes, self.embedding_dim), dtype=torch.float32, device='cpu' if not torch.cuda.is_available() else 'cuda')
                # Skip to assignment as no raw VPE data to process
                self.final_refined_embeddings_tensor = processed_vpe_for_update
                
                # 关键修复：创建零VPE时也要设置索引映射！
                sorted_class_ids = sorted(self.class_mapping.keys())
                self.index_to_class_id = {idx: class_id for idx, class_id in enumerate(sorted_class_ids)}
                self.class_id_to_index = {class_id: idx for idx, class_id in enumerate(sorted_class_ids)}
                self.initial_object_set = [self.class_mapping[class_id] for class_id in sorted_class_ids]
                
                # print(f"[Debug] _update_vpe: VPE updated with zeros due to empty raw VPE. Current class count: {self.num_classes}")
                # print(f"[Debug] _update_vpe: 零VPE索引映射: {self.index_to_class_id}")
                # print(f"[Debug] _update_vpe: 零VPE对象集合: {self.initial_object_set}")
                if self.final_refined_embeddings_tensor is not None:
                    # print(f"[Debug] _update_vpe: Final embeddings tensor shape after update: {self.final_refined_embeddings_tensor.shape}")
                    pass
                return True

        embedding_dim = raw_new_vpe.shape[-1]

        if raw_new_vpe.dim() == 3 and raw_new_vpe.shape[0] > 0: 
            # 关键修复：按照类别ID顺序来构建VPE，确保VPE索引与类别ID对应
            # 这样VPE索引0对应类别ID=0，VPE索引1对应类别ID=1，依此类推
            
            first_prompt_cls = new_visuals['cls'][0] if len(new_visuals['cls']) > 0 else np.array([])
            
            if len(first_prompt_cls) == 0:
                print("[Warning] _update_vpe: No class information in first prompt")
                processed_vpe_for_update = torch.zeros((1, 1, embedding_dim), dtype=raw_new_vpe.dtype, device=raw_new_vpe.device)
                vpe_class_order = []
            else:
                max_objects = min(len(first_prompt_cls), raw_new_vpe.shape[1])
                selected_vpe = raw_new_vpe[:, :max_objects, :]
                
                # 🔧 修复：按照原始出现顺序收集类别ID，不进行排序！
                seen_class_ids = set()
                unique_class_ids_in_order = []
                for cls_id in first_prompt_cls[:max_objects]:
                    cls_id = int(cls_id)
                    if cls_id not in seen_class_ids:
                        seen_class_ids.add(cls_id)
                        unique_class_ids_in_order.append(cls_id)
                
                # print(f"[Debug] _update_vpe: 发现的唯一类别ID（按原始出现顺序）: {unique_class_ids_in_order}")
                # print(f"[Debug] _update_vpe: 原始prompt中的类别序列: {first_prompt_cls[:max_objects].tolist()}")
                
                # 🔧 额外调试：显示每个类别ID对应的名称
                # print(f"[Debug] _update_vpe: 类别ID对应的名称:")
                for cls_id in unique_class_ids_in_order:
                    cls_name = self.class_mapping.get(cls_id, f'unknown_{cls_id}')
                    true_name = self.true_class_names[cls_id] if (self.true_class_names and cls_id < len(self.true_class_names)) else f'class_{cls_id}'
                    # print(f"    类别ID {cls_id} -> 映射名称'{cls_name}' | 真实名称'{true_name}'")
                
                # 🔧 额外调试：检查所有visual prompts的内容
                # print(f"[Debug] _update_vpe: 检查所有visual prompts:")
                for i, (img_path, cls_arr) in enumerate(zip(self.visual_prompts['image_paths'], self.visual_prompts['cls'])):
                    # print(f"    Prompt {i+1}: {os.path.basename(img_path)} -> 类别: {cls_arr.tolist() if hasattr(cls_arr, 'tolist') else cls_arr}")
                    if hasattr(cls_arr, 'tolist'):
                        for j, cls_id in enumerate(cls_arr.tolist()):
                            cls_name = self.class_mapping.get(cls_id, f'unknown_{cls_id}')
                            # print(f"        对象{j+1}: 类别ID{cls_id} -> 名称'{cls_name}'")
                
                # print(f"[Debug] _update_vpe: 当前使用的是第1个prompt的类别序列进行VPE构建")
                
                # 创建类别ID到原始位置的映射
                class_id_to_positions = {}
                for obj_idx in range(max_objects):
                    class_id = int(first_prompt_cls[obj_idx])
                    if class_id not in class_id_to_positions:
                        class_id_to_positions[class_id] = []
                    class_id_to_positions[class_id].append(obj_idx)
                
                # print(f"[Debug] _update_vpe: 类别ID到位置映射: {class_id_to_positions}")
                
                vpe_class_order = []
                class_embeddings_list = []
                
                # 🔧 修复：按照原始出现顺序处理每个类别，而不是排序后的顺序
                for class_id in unique_class_ids_in_order:  # 使用原始顺序！
                    positions = class_id_to_positions[class_id]
                    # print(f"[Debug] _update_vpe: 处理类别{class_id}，原始位置: {positions}")
                    
                    # 如果一个类别有多个实例，取第一个实例的特征（或平均值）
                    if len(positions) == 1:
                        # 单个实例，直接使用
                        pos = positions[0]
                        class_embeddings = []
                        for img_idx in range(selected_vpe.shape[0]):
                            class_embeddings.append(selected_vpe[img_idx, pos])
                    else:
                        # 多个实例，计算平均
                        # print(f"[Debug] _update_vpe: 类别{class_id}有{len(positions)}个实例，计算平均特征")
                        all_embeddings = []
                        for pos in positions:
                            for img_idx in range(selected_vpe.shape[0]):
                                all_embeddings.append(selected_vpe[img_idx, pos])
                        class_embeddings = [torch.stack(all_embeddings, dim=0).mean(dim=0)]
                    
                    # 计算该类别的平均嵌入
                    avg_emb = torch.stack(class_embeddings, dim=0).mean(dim=0, keepdim=True)
                    avg_emb_normalized = F.normalize(avg_emb, dim=-1, p=2)
                    
                    class_embeddings_list.append(avg_emb_normalized)
                    vpe_class_order.append(class_id)
                    
                    # print(f"[Debug] _update_vpe: VPE索引{len(vpe_class_order)-1} -> 类别{class_id} ({self.class_mapping.get(class_id, 'unknown')})")
                
                if class_embeddings_list:
                    processed_vpe_for_update = torch.cat(class_embeddings_list, dim=0).unsqueeze(0)
                else:
                    processed_vpe_for_update = torch.zeros((1, 1, embedding_dim), dtype=raw_new_vpe.dtype, device=raw_new_vpe.device)
                    vpe_class_order = []
            
            # 🔧 修复：更新索引映射，按照原始出现顺序而不是类别ID排序
            self.index_to_class_id = {idx: class_id for idx, class_id in enumerate(vpe_class_order)}
            self.class_id_to_index = {class_id: idx for idx, class_id in enumerate(vpe_class_order)}
            
            # 根据VPE顺序设置对象集合
            self.initial_object_set = [self.class_mapping.get(class_id, f"unknown_{class_id}") 
                                     for class_id in vpe_class_order]
            self.num_classes = len(vpe_class_order)  # 更新为实际VPE中的类别数量
            
            # print(f"[Debug] _update_vpe: 最终VPE索引映射（按原始出现顺序）: {self.index_to_class_id}")
            # print(f"[Debug] _update_vpe: 对象集合: {self.initial_object_set}")
            # print(f"[Debug] _update_vpe: VPE形状: {processed_vpe_for_update.shape}")
            print(f"[Info] _update_vpe: ✅ 模型将直接预测类别ID，不依赖VPE索引映射")
            # print(f"[Debug] _update_vpe: VPE构建验证:")
            for idx, class_id in self.index_to_class_id.items():
                expected_class_name = self.true_class_names[class_id] if (self.true_class_names and class_id < len(self.true_class_names)) else f"class_{class_id}"
                actual_class_name = self.class_mapping.get(class_id, 'unknown')
                match = "✓" if expected_class_name == actual_class_name else "✗"
                # print(f"         VPE位置{idx} 包含类别ID{class_id} -> 名称'{actual_class_name}' | 预期'{expected_class_name}' | {match}")
                
                # ✅ 重点说明：模型预测类别ID时的逻辑
                # print(f"         💡 当模型预测类别ID{class_id}时，将直接映射到'{actual_class_name}' ✅")
        
        elif raw_new_vpe.dim() == 2 and raw_new_vpe.shape[0] == num_target_classes : 
            # print(f"[Debug] _update_vpe: Raw VPE shape {raw_new_vpe.shape}. Assuming [num_classes, dim], unsqueezing to [1, num_classes, dim].")
            processed_vpe_for_update = raw_new_vpe.unsqueeze(0) 
            processed_vpe_for_update = F.normalize(processed_vpe_for_update, dim=-1, p=2)
            
            # 🔧 修复：确保索引映射正确设置
            sorted_class_ids = sorted(self.class_mapping.keys())
            self.index_to_class_id = {idx: class_id for idx, class_id in enumerate(sorted_class_ids)}
            self.class_id_to_index = {class_id: idx for idx, class_id in enumerate(sorted_class_ids)}
            self.initial_object_set = [self.class_mapping[class_id] for class_id in sorted_class_ids]
            # print(f"[Debug] _update_vpe: 设置索引映射 (2D情况): {self.index_to_class_id}")
            # print(f"[Debug] _update_vpe: 设置对象集合 (2D情况): {self.initial_object_set}")

        elif raw_new_vpe.dim() == 3 and raw_new_vpe.shape[0] == 1 and raw_new_vpe.shape[1] == num_target_classes: 
            # print(f"[Debug] _update_vpe: Raw VPE shape {raw_new_vpe.shape}. Assuming already [1, num_classes, dim].")
            processed_vpe_for_update = raw_new_vpe 
            processed_vpe_for_update = F.normalize(processed_vpe_for_update, dim=-1, p=2)
            
            # 🔧 修复：确保索引映射正确设置
            sorted_class_ids = sorted(self.class_mapping.keys())
            self.index_to_class_id = {idx: class_id for idx, class_id in enumerate(sorted_class_ids)}
            self.class_id_to_index = {class_id: idx for idx, class_id in enumerate(sorted_class_ids)}
            self.initial_object_set = [self.class_mapping[class_id] for class_id in sorted_class_ids]
            # print(f"[Debug] _update_vpe: 设置索引映射 (3D情况): {self.index_to_class_id}")
            # print(f"[Debug] _update_vpe: 设置对象集合 (3D情况): {self.initial_object_set}")
        else:
            print(f"[ERROR] _update_vpe: Raw VPE has unhandled shape: {raw_new_vpe.shape} for num_target_classes={num_target_classes}. VPE update failed.")
            
            # 🔧 即使VPE形状不匹配，也要尝试设置基本的索引映射
            if self.class_mapping:
                print("[Info] _update_vpe: VPE形状不匹配，但仍设置基本索引映射")
                sorted_class_ids = sorted(self.class_mapping.keys())
                self.index_to_class_id = {idx: class_id for idx, class_id in enumerate(sorted_class_ids)}
                self.class_id_to_index = {class_id: idx for idx, class_id in enumerate(sorted_class_ids)}
                self.initial_object_set = [self.class_mapping[class_id] for class_id in sorted_class_ids]
                # print(f"[Debug] _update_vpe: 紧急设置索引映射: {self.index_to_class_id}")
                # print(f"[Debug] _update_vpe: 紧急设置对象集合: {self.initial_object_set}")
            
            return False

        self.final_refined_embeddings_tensor = processed_vpe_for_update
        
        # print(f"[Debug] _update_vpe: VPE updated via overwrite. Current class count: {self.num_classes}")
        # print(f"[Debug] _update_vpe: Class mapping maintained: {self.class_mapping}")
        if self.final_refined_embeddings_tensor is not None:
            # print(f"[Debug] _update_vpe: Final embeddings tensor shape after update: {self.final_refined_embeddings_tensor.shape}")
            # print(f"[Debug] _update_vpe: Object set: {self.initial_object_set}")
            pass
        else:
            # print("[Debug] _update_vpe: Final embeddings tensor is None after update.")
            pass
        
        # 最终验证
        if not self._validate_class_consistency():
            print("[ERROR] _update_vpe: 最终类别一致性验证失败")
            return False
            
        return True

    def _add_or_replace_prompt(self, image_path, bboxes_np_array, cls_ids_np_array, group_confidence, is_initial=False):
        """
        添加或替换 visual prompt。
        bboxes_np_array: np.array of shape [M, 4] (M bboxes for this single prompt entry)
        cls_ids_np_array: np.array of shape [M] (M class IDs for this single prompt entry)
        group_confidence: float, 代表这组bboxes/cls_ids的整体置信度
        """
        if not isinstance(bboxes_np_array, np.ndarray) or bboxes_np_array.ndim != 2 or bboxes_np_array.shape[1] != 4:
            print(f"[ERROR AddPrompt] Invalid bboxes_np_array shape: {bboxes_np_array.shape if isinstance(bboxes_np_array, np.ndarray) else type(bboxes_np_array)}. Expected [M, 4].")
            return
        if not isinstance(cls_ids_np_array, np.ndarray) or cls_ids_np_array.ndim != 1:
            print(f"[ERROR AddPrompt] Invalid cls_ids_np_array shape: {cls_ids_np_array.shape if isinstance(cls_ids_np_array, np.ndarray) else type(cls_ids_np_array)}. Expected [M].")
            return
        if bboxes_np_array.shape[0] != cls_ids_np_array.shape[0]:
            print(f"[ERROR AddPrompt] Mismatch between number of bboxes ({bboxes_np_array.shape[0]}) and class IDs ({cls_ids_np_array.shape[0]}).")
            return


        if is_initial:
            self.visual_prompts['image_paths'].append(image_path)
            self.visual_prompts['bboxes'].append(bboxes_np_array) 
            self.visual_prompts['cls'].append(cls_ids_np_array)     
            self.visual_prompts['confidences'].append(group_confidence)
            self.visual_prompts['is_initial'].append(True)
            # print(f"[Debug AddPrompt] Added initial visual prompt (Objects: {len(cls_ids_np_array)}). Total prompts: {len(self.visual_prompts['image_paths'])}")
            return

        if len(self.visual_prompts['image_paths']) < self.max_prompts:
            self.visual_prompts['image_paths'].append(image_path)
            self.visual_prompts['bboxes'].append(bboxes_np_array)
            self.visual_prompts['cls'].append(cls_ids_np_array)
            self.visual_prompts['confidences'].append(group_confidence)
            self.visual_prompts['is_initial'].append(False)
            # print(f"[Debug AddPrompt] Added new dynamic visual prompt (Objects: {len(cls_ids_np_array)}). Total prompts: {len(self.visual_prompts['image_paths'])}")
        else:
            dynamic_indices = [i for i, is_init in enumerate(self.visual_prompts['is_initial']) if not is_init]
            if not dynamic_indices:
                # print("[Debug AddPrompt] Max prompts reached, but no dynamic (non-initial) prompts to replace.")
                return
                
            min_conf_idx_in_dynamics = min(dynamic_indices, key=lambda i: self.visual_prompts['confidences'][i])
            min_conf_dynamic = self.visual_prompts['confidences'][min_conf_idx_in_dynamics]
            
            if group_confidence > min_conf_dynamic:
                # print(f"[Debug AddPrompt] Replacing dynamic prompt (Index: {min_conf_idx_in_dynamics}, Old_Conf: {min_conf_dynamic:.3f}) with new (Objects: {len(cls_ids_np_array)}, Group_Conf: {group_confidence:.3f}).")
                self.visual_prompts['image_paths'][min_conf_idx_in_dynamics] = image_path
                self.visual_prompts['bboxes'][min_conf_idx_in_dynamics] = bboxes_np_array
                self.visual_prompts['cls'][min_conf_idx_in_dynamics] = cls_ids_np_array
                self.visual_prompts['confidences'][min_conf_idx_in_dynamics] = group_confidence
                self.visual_prompts['is_initial'][min_conf_idx_in_dynamics] = False
            else:
                # print(f"[Debug AddPrompt] New dynamic prompt group conf ({group_confidence:.3f}) not higher than lowest dynamic ({min_conf_dynamic:.3f}). Not replaced.")
                pass

    def auto_label_with_vp(self, prompt_image_paths: list, visuals: dict, target_image_path: str, 
                           conf_thresh: float = 0.4, iou_match_thresh: float = 0.5):
        """
        Auto-labeling with refined embeddings.
        Supports adding all high-confidence detections from one image as a single multi-object prompt.
        """
        try:
            # 🔧 保护检查：确保类别映射存在
            if not self.class_mapping and self.true_class_names:
                print("[Warning auto_label_vp] 类别映射为空，尝试从true_class_names重建")
                # 从true_class_names重建基本的类别映射
                for cls_id, cls_name in enumerate(self.true_class_names):
                    self.class_mapping[cls_id] = cls_name
                    self.reverse_class_mapping[cls_name] = cls_id
                print(f"[Info auto_label_vp] 重建了 {len(self.class_mapping)} 个类别映射")
            
            is_first_call_or_vpe_missing = self.final_refined_embeddings_tensor is None or self.initial_object_set is None

            if is_first_call_or_vpe_missing:
                print("[Info auto_label_vp] No cached VPE. Will generate from provided initial prompts.")
                if not prompt_image_paths:
                    print("[ERROR auto_label_vp] Initial prompt_image_paths is empty. Cannot generate initial VPE.")
                    return []
                if not (visuals and 'bboxes' in visuals and 'cls' in visuals and \
                        len(prompt_image_paths) == len(visuals['bboxes']) == len(visuals['cls'])):
                    bbox_len = len(visuals.get('bboxes',[])) if visuals else -1
                    cls_len = len(visuals.get('cls',[])) if visuals else -1
                    print(f"[ERROR auto_label_vp] Initial input list lengths mismatch or visuals incomplete: paths={len(prompt_image_paths)}, bboxes={bbox_len}, cls={cls_len}")
                    return []

                # print("\n[Debug auto_label_vp] Adding initial visual prompts from input:")
                self.visual_prompts = {key: [] for key in self.visual_prompts} 

                for i, img_path in enumerate(prompt_image_paths):
                    current_bboxes = np.asarray(visuals['bboxes'][i])
                    current_cls = np.asarray(visuals['cls'][i])
                    
                    # Ensure bboxes is [M, 4]
                    if current_bboxes.ndim == 1:
                        if current_bboxes.shape[0] == 4: # A single bbox [x1,y1,x2,y2]
                            current_bboxes = current_bboxes.reshape(1, 4)
                        else: # Malformed
                            print(f"[ERROR auto_label_vp] Initial prompt {i} bbox has incorrect shape {current_bboxes.shape}. Skipping.")
                            continue
                    elif current_bboxes.ndim == 2 and current_bboxes.shape[1] != 4: # [M, X] where X!=4
                         print(f"[ERROR auto_label_vp] Initial prompt {i} bboxes have incorrect column count {current_bboxes.shape[1]}. Skipping.")
                         continue
                    elif current_bboxes.ndim > 2:
                         print(f"[ERROR auto_label_vp] Initial prompt {i} bboxes have too many dimensions {current_bboxes.ndim}. Skipping.")
                         continue

                    # Ensure cls is [M]
                    if current_cls.ndim == 0: 
                        current_cls = current_cls.reshape(1)
                    elif current_cls.ndim > 1:
                        print(f"[ERROR auto_label_vp] Initial prompt {i} cls array has too many dimensions {current_cls.ndim}. Skipping.")
                        continue
                    
                    if current_bboxes.shape[0] == 0 and current_cls.shape[0] == 0: # Empty prompt, allow if consistent
                        pass # Will be handled by _add_or_replace_prompt if needed
                    elif current_bboxes.shape[0] != current_cls.shape[0]:
                        print(f"[ERROR auto_label_vp] Mismatch in number of bboxes ({current_bboxes.shape[0]}) and class IDs ({current_cls.shape[0]}) for initial prompt {i}. Skipping.")
                        continue
                    
                    self._add_or_replace_prompt(
                        img_path,
                        current_bboxes, 
                        current_cls,    
                        1.0, 
                        is_initial=True
                    )
                
                update_success = self._update_vpe_with_new_annotations()
                if not update_success or self.final_refined_embeddings_tensor is None:
                    print("[ERROR auto_label_vp] Failed to generate initial VPE. Cannot proceed with auto-labeling.")
                    return []
            
            if self.final_refined_embeddings_tensor is None or self.initial_object_set is None:
                 print("[ERROR auto_label_vp] VPE or object set is still None after VPE update/check. Cannot predict.")
                 return []

            print("\n[Info auto_label_vp] Using current VPE for inference on target image.")
            if self.final_refined_embeddings_tensor.numel() > 0:
                # print(f"[Debug auto_label_vp] VPE Tensor mean: {self.final_refined_embeddings_tensor.mean().item()}")
                # print(f"[Debug auto_label_vp] VPE Tensor std: {self.final_refined_embeddings_tensor.std().item()}")
                # print(f"[Debug auto_label_vp] VPE Tensor min: {self.final_refined_embeddings_tensor.min().item()}")
                # print(f"[Debug auto_label_vp] VPE Tensor max: {self.final_refined_embeddings_tensor.max().item()}")
                # print(f"[Debug auto_label_vp] Is NaN present: {torch.isnan(self.final_refined_embeddings_tensor).any().item()}")
                pass
            else:
                # print("[Debug auto_label_vp] VPE Tensor is empty.")
                pass
            # print(f"[Debug auto_label_vp] VPE Shape: {self.final_refined_embeddings_tensor.shape}")
            # print(f"[Debug auto_label_vp] Object set for VPE: {self.initial_object_set}")
            # print(f"[Debug auto_label_vp] 当前类别映射: {self.class_mapping}")

            # ✅ 简化逻辑：不需要复杂的索引映射，模型直接预测类别ID
            # 只需要确保基本的类别映射存在
            if not self.class_mapping:
                print("[Error auto_label_vp] 没有类别映射，无法进行预测")
                return []
            
            print(f"[Info auto_label_vp] 使用直接类别ID映射逻辑 ✅")

            self.model.predictor = None
            self.model.set_classes(self.initial_object_set, self.final_refined_embeddings_tensor)
            
            final_results_list = self.model.predict(target_image_path, save=False, verbose=False)
            final_results = final_results_list[0] if final_results_list else None

            predictions = []
            collected_bboxes_for_new_prompt = []
            collected_cls_ids_for_new_prompt = []
            collected_scores_for_new_prompt = []

            if final_results and hasattr(final_results, 'boxes') and final_results.boxes:
                for box_obj in final_results.boxes:
                    x1, y1, x2, y2 = box_obj.xyxy[0].cpu().numpy().tolist()
                    score = float(box_obj.conf[0].cpu().numpy())
                    predicted_index = int(box_obj.cls[0].cpu().numpy())
                    
                    # ✅ 恢复到原始的直接映射逻辑
                    if predicted_index >= len(self.initial_object_set) or predicted_index < 0:
                        print(f"[Warning auto_label_vp] Predicted index {predicted_index} is out of bounds for current object set size {len(self.initial_object_set)}. Skipping this prediction.")
                        continue
                    
                    actual_class_id = predicted_index # 直接映射
                    class_name = self.class_mapping.get(actual_class_id, f"unknown_{actual_class_id}")
                    
                    # print(f"[Debug auto_label_vp] 预测映射: 模型预测索引 {predicted_index} -> 真实类别ID {actual_class_id} -> 名称 '{class_name}' ✅")
                    
                    if self.true_class_names and 0 <= actual_class_id < len(self.true_class_names):
                        expected_name = self.true_class_names[actual_class_id]
                        if class_name == expected_name:
                            print(f"[Info auto_label_vp] 类别验证: ID{actual_class_id} -> '{class_name}' ✅ 正确")
                        else:
                            print(f"[Warning auto_label_vp] 类别验证: ID{actual_class_id} -> 映射'{class_name}' vs 期望'{expected_name}'")
                    else:
                        print(f"[Info auto_label_vp] 类别ID{actual_class_id}超出真实类别范围，使用映射名称'{class_name}'")
                    
                    if score >= conf_thresh: 
                        predictions.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': actual_class_id, 
                            'class_name': class_name, 
                            'confidence': score
                        })
                        
                        # 收集高置信度的检测结果作为新的visual prompt
                        if score >= self.conf_thresh_for_update: 
                            collected_bboxes_for_new_prompt.append([x1, y1, x2, y2])
                            collected_cls_ids_for_new_prompt.append(actual_class_id)
                            collected_scores_for_new_prompt.append(score)
            
            print(f"[Info auto_label_vp] Found {len(predictions)} predictions for target image: {os.path.basename(target_image_path)}")

            # 检查是否有高置信度的检测结果需要作为新的visual prompt
            if collected_bboxes_for_new_prompt: 
                bboxes_np = np.array(collected_bboxes_for_new_prompt)
                cls_ids_np = np.array(collected_cls_ids_for_new_prompt)
                group_confidence = np.mean(collected_scores_for_new_prompt) if collected_scores_for_new_prompt else 0.0

                print(f"[Info auto_label_vp] 发现{len(collected_bboxes_for_new_prompt)}个高置信度检测结果，将其作为新的visual prompt")
                print(f"[Info auto_label_vp] 平均置信度: {group_confidence:.3f}")
                
                # 添加新的visual prompt
                self._add_or_replace_prompt(
                    target_image_path,
                    bboxes_np,
                    cls_ids_np,
                    group_confidence, 
                    is_initial=False
                )
                
                # 更新VPE
                print("[Info auto_label_vp] 更新VPE以包含新的visual prompt")
                update_success = self._update_vpe_with_new_annotations()
                if update_success:
                    print("[Info auto_label_vp] VPE更新成功")
                    # 使用更新后的VPE重新进行预测
                    print("[Info auto_label_vp] 使用更新后的VPE重新进行预测")
                    self.model.predictor = None
                    self.model.set_classes(self.initial_object_set, self.final_refined_embeddings_tensor)
                    
                    # 重新预测
                    final_results_list = self.model.predict(target_image_path, save=False, verbose=False)
                    final_results = final_results_list[0] if final_results_list else None
                    
                    # 清空之前的预测结果
                    predictions = []
                    
                    # 处理新的预测结果
                    if final_results and hasattr(final_results, 'boxes') and final_results.boxes:
                        for box_obj in final_results.boxes:
                            x1, y1, x2, y2 = box_obj.xyxy[0].cpu().numpy().tolist()
                            score = float(box_obj.conf[0].cpu().numpy())
                            predicted_index = int(box_obj.cls[0].cpu().numpy())
                            
                            # ✅ 恢复到原始的直接映射逻辑 (re-predict loop)
                            if predicted_index >= len(self.initial_object_set) or predicted_index < 0:
                                print(f"[Warning auto_label_vp RE-PREDICT] Predicted index {predicted_index} is out of bounds for current object set size {len(self.initial_object_set)}. Skipping this prediction.")
                                continue
                            
                            actual_class_id = predicted_index # 直接映射
                            class_name = self.class_mapping.get(actual_class_id, f"unknown_{actual_class_id}")

                            # print(f"[Debug auto_label_vp RE-PREDICT] 预测映射: 模型预测索引 {predicted_index} -> 真实类别ID {actual_class_id} -> 名称 '{class_name}' ✅")

                            if score >= conf_thresh:
                                predictions.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'class_id': actual_class_id,
                                    'class_name': class_name,
                                    'confidence': score
                                })
                    
                    print(f"[Info auto_label_vp] 使用更新后的VPE重新预测，得到{len(predictions)}个结果")
                else:
                    print("[Warning auto_label_vp] VPE更新失败，使用原始预测结果")
            
            return predictions

        except Exception as e:
            print("--- Exception in auto_label_with_vp ---")
            traceback.print_exc()
            prompt_paths_str = str([os.path.basename(p) for p in prompt_image_paths]) if prompt_image_paths else "[]"
            visuals_str = f"bboxes_count: {len(visuals.get('bboxes',[])) if visuals else 'N/A'}, cls_count: {len(visuals.get('cls',[])) if visuals else 'N/A'}"
            print(f"[ERROR CONTEXT] prompt_image_paths: {prompt_paths_str}")
            print(f"[ERROR CONTEXT] visuals: {visuals_str}")
            print(f"[ERROR CONTEXT] target_image_path: {os.path.basename(target_image_path) if target_image_path else 'N/A'}")
            return []

    def debug_class_mapping_and_vpe_alignment(self):
        """
        调试函数：检查visual prompt的类别排列和VPE embedding的对应关系
        """
        print("\n" + "="*80)
        print("🔍 调试：Visual Prompt类别排列和VPE Embedding对应关系检查")
        print("="*80)
        
        # 1. 检查当前visual prompts的类别信息
        print("\n📋 1. 当前Visual Prompts详细信息:")
        if not self.visual_prompts['image_paths']:
            print("   ❌ 没有visual prompts")
            return
            
        for i, img_path in enumerate(self.visual_prompts['image_paths']):
            cls_array = self.visual_prompts['cls'][i]
            bbox_array = self.visual_prompts['bboxes'][i]
            is_initial = self.visual_prompts['is_initial'][i]
            confidence = self.visual_prompts['confidences'][i]
            
            print(f"\n   Prompt {i+1}: {os.path.basename(img_path)}")
            print(f"   - 是否初始prompt: {is_initial}")
            print(f"   - 置信度: {confidence:.3f}")
            print(f"   - 对象数量: {len(cls_array) if isinstance(cls_array, np.ndarray) else 'N/A'}")
            
            if isinstance(cls_array, np.ndarray) and cls_array.size > 0:
                print(f"   - 类别ID序列: {cls_array.tolist()}")
                print(f"   - 对应类别名称: {[self.class_mapping.get(int(cls_id), f'unknown_{cls_id}') for cls_id in cls_array]}")
                
                # 显示每个对象的详细信息
                for j, cls_id in enumerate(cls_array):
                    bbox = bbox_array[j] if isinstance(bbox_array, np.ndarray) and j < len(bbox_array) else "N/A"
                    class_name = self.class_mapping.get(int(cls_id), f'unknown_{cls_id}')
                    print(f"     对象{j+1}: 类别ID={int(cls_id)} -> 名称='{class_name}' | BBox={bbox}")
        
        # 2. 检查类别映射
        print(f"\n🗂️  2. 类别映射信息:")
        print(f"   - 总类别数: {self.num_classes}")
        print(f"   - 类别映射 (ID -> 名称): {self.class_mapping}")
        print(f"   - 反向映射 (名称 -> ID): {self.reverse_class_mapping}")
        
        if self.true_class_names:
            print(f"   - 真实类别名称列表: {self.true_class_names}")
            print("   - 类别ID与真实名称对照:")
            for cls_id, mapped_name in self.class_mapping.items():
                true_name = self.true_class_names[cls_id] if cls_id < len(self.true_class_names) else f"超出范围_{cls_id}"
                match = "✓" if mapped_name == true_name else "✗"
                print(f"     ID {cls_id}: 映射='{mapped_name}' | 真实='{true_name}' | {match}")
        
        # 3. 检查VPE索引映射
        print(f"\n🔗 3. VPE索引映射:")
        print(f"   - VPE索引 -> 类别ID: {self.index_to_class_id}")
        print(f"   - 类别ID -> VPE索引: {self.class_id_to_index}")
        print(f"   - Initial Object Set: {self.initial_object_set}")
        
        # 4. 检查VPE tensor信息
        print(f"\n🧠 4. VPE Tensor信息:")
        if self.final_refined_embeddings_tensor is not None:
            vpe_shape = self.final_refined_embeddings_tensor.shape
            print(f"   - VPE形状: {vpe_shape}")
            print(f"   - 嵌入维度: {self.embedding_dim}")
            
            if len(vpe_shape) >= 2:
                print(f"   - VPE中的类别数: {vpe_shape[1] if len(vpe_shape) > 1 else vpe_shape[0]}")
                
                # 检查每个VPE索引对应的类别
                print("   - VPE索引对应关系:")
                vpe_class_count = vpe_shape[1] if len(vpe_shape) > 1 else vpe_shape[0]
                for vpe_idx in range(vpe_class_count):
                    if vpe_idx in self.index_to_class_id:
                        class_id = self.index_to_class_id[vpe_idx]
                        class_name = self.class_mapping.get(class_id, f'unknown_{class_id}')
                        true_name = self.true_class_names[class_id] if (self.true_class_names and class_id < len(self.true_class_names)) else f"class_{class_id}"
                        
                        # 计算该索引的embedding统计信息
                        if len(vpe_shape) == 3:  # [batch, classes, dim]
                            emb_vector = self.final_refined_embeddings_tensor[0, vpe_idx, :]
                        else:  # [classes, dim]
                            emb_vector = self.final_refined_embeddings_tensor[vpe_idx, :]
                        
                        emb_norm = torch.norm(emb_vector).item()
                        emb_mean = emb_vector.mean().item()
                        
                        print(f"     VPE[{vpe_idx}] -> 类别ID{class_id} -> '{class_name}' (真实:'{true_name}') | 范数:{emb_norm:.4f} | 均值:{emb_mean:.4f}")
                    else:
                        print(f"     VPE[{vpe_idx}] -> ❌ 没有对应的类别ID映射")
        else:
            print("   ❌ VPE tensor为None")
        
        # 5. 一致性检查
        print(f"\n✅ 5. 一致性检查:")
        
        # 检查映射长度一致性
        mapping_consistent = len(self.class_mapping) == len(self.reverse_class_mapping) == self.num_classes
        print(f"   - 映射长度一致性: {mapping_consistent} (正向:{len(self.class_mapping)}, 反向:{len(self.reverse_class_mapping)}, 计数:{self.num_classes})")
        
        # 检查VPE形状一致性
        if self.final_refined_embeddings_tensor is not None:
            vpe_shape = self.final_refined_embeddings_tensor.shape
            vpe_class_count = vpe_shape[1] if len(vpe_shape) > 1 else vpe_shape[0]
            vpe_shape_consistent = vpe_class_count == self.num_classes
            print(f"   - VPE形状一致性: {vpe_shape_consistent} (VPE类别数:{vpe_class_count}, 映射类别数:{self.num_classes})")
        else:
            print(f"   - VPE形状一致性: ❌ (VPE为None)")
        
        # 检查索引映射完整性
        index_mapping_complete = (len(self.index_to_class_id) == len(self.class_id_to_index) == self.num_classes)
        print(f"   - 索引映射完整性: {index_mapping_complete} (索引->ID:{len(self.index_to_class_id)}, ID->索引:{len(self.class_id_to_index)})")
        
        # 检查initial_object_set一致性
        if self.initial_object_set:
            object_set_consistent = len(self.initial_object_set) == self.num_classes
            print(f"   - Object Set一致性: {object_set_consistent} (Object Set长度:{len(self.initial_object_set)})")
        else:
            print(f"   - Object Set一致性: ❌ (Object Set为空)")
        
        # 6. 潜在问题警告
        print(f"\n⚠️  6. 潜在问题检查:")
        
        # 检查类别ID是否连续
        if self.class_mapping:
            class_ids = sorted(self.class_mapping.keys())
            expected_ids = list(range(len(class_ids)))
            if class_ids != expected_ids:
                print(f"   ⚠️  类别ID不连续: 实际={class_ids}, 期望={expected_ids}")
            else:
                print(f"   ✓ 类别ID连续: {class_ids}")
        
        # 检查VPE索引是否从0开始连续
        if self.index_to_class_id:
            vpe_indices = sorted(self.index_to_class_id.keys())
            expected_indices = list(range(len(vpe_indices)))
            if vpe_indices != expected_indices:
                print(f"   ⚠️  VPE索引不连续: 实际={vpe_indices}, 期望={expected_indices}")
            else:
                print(f"   ✓ VPE索引连续: {vpe_indices}")
        
        print("\n" + "="*80)
        print("🔍 调试检查完成")
        print("="*80)

    def debug_actual_data_flow(self, prompt_image_paths, visuals, target_image_path):
        """
        调试实际数据流：检查从GUI传入的数据到最终预测的完整流程
        """
        print("\n" + "🔍" * 80)
        print("🔍 实际数据流调试 - 从GUI到预测的完整流程")
        print("🔍" * 80)
        
        # 1. 检查输入数据
        print(f"\n📥 1. 输入数据检查:")
        print(f"   - 提示图片数量: {len(prompt_image_paths)}")
        print(f"   - 目标图片: {os.path.basename(target_image_path) if target_image_path else 'None'}")
        
        if 'bboxes' in visuals and 'cls' in visuals:
            print(f"   - Visuals中的bboxes数量: {len(visuals['bboxes'])}")
            print(f"   - Visuals中的cls数量: {len(visuals['cls'])}")
            
            for i, (bboxes, cls_arr) in enumerate(zip(visuals['bboxes'], visuals['cls'])):
                print(f"   - 第{i+1}个prompt:")
                print(f"     * Bboxes形状: {bboxes.shape if hasattr(bboxes, 'shape') else type(bboxes)}")
                print(f"     * Cls形状: {cls_arr.shape if hasattr(cls_arr, 'shape') else type(cls_arr)}")
                print(f"     * Cls内容: {cls_arr.tolist() if hasattr(cls_arr, 'tolist') else cls_arr}")
                
                # 检查类别ID是否在合理范围内
                if hasattr(cls_arr, 'tolist'):
                    cls_list = cls_arr.tolist()
                    for j, cls_id in enumerate(cls_list):
                        if self.true_class_names and 0 <= cls_id < len(self.true_class_names):
                            true_name = self.true_class_names[cls_id]
                            print(f"       对象{j+1}: 类别ID={cls_id} -> 真实名称='{true_name}'")
                        else:
                            print(f"       对象{j+1}: 类别ID={cls_id} -> ⚠️ 超出真实类别范围!")
        
        # 2. 检查当前YOLOEWrapper状态
        print(f"\n🧠 2. YOLOEWrapper当前状态:")
        print(f"   - 真实类别名称: {self.true_class_names}")
        print(f"   - 当前类别映射: {self.class_mapping}")
        print(f"   - 当前索引映射: {self.index_to_class_id}")
        print(f"   - 当前对象集合: {self.initial_object_set}")
        print(f"   - VPE是否存在: {self.final_refined_embeddings_tensor is not None}")
        if self.final_refined_embeddings_tensor is not None:
            print(f"   - VPE形状: {self.final_refined_embeddings_tensor.shape}")
        
        # 3. 模拟添加prompts的过程
        print(f"\n📝 3. 模拟添加prompts过程:")
        
        # 备份当前状态
        original_visual_prompts = {
            'image_paths': self.visual_prompts['image_paths'].copy(),
            'bboxes': self.visual_prompts['bboxes'].copy(),
            'cls': self.visual_prompts['cls'].copy(),
            'confidences': self.visual_prompts['confidences'].copy(),
            'is_initial': self.visual_prompts['is_initial'].copy()
        }
        original_class_mapping = self.class_mapping.copy()
        original_reverse_class_mapping = self.reverse_class_mapping.copy()
        original_index_mapping = self.index_to_class_id.copy()
        original_class_id_to_index = self.class_id_to_index.copy()
        original_initial_object_set = self.initial_object_set.copy() if self.initial_object_set else []
        original_num_classes = self.num_classes
        
        try:
            # 清空当前prompts（模拟重新开始）
            self.visual_prompts = {
                'image_paths': [],
                'bboxes': [],
                'cls': [],
                'confidences': [],
                'is_initial': []
            }
            self.class_mapping = {}
            self.reverse_class_mapping = {}
            self.index_to_class_id = {}
            self.class_id_to_index = {}
            
            # 逐个添加prompts
            for i, (img_path, bboxes, cls_arr) in enumerate(zip(prompt_image_paths, visuals['bboxes'], visuals['cls'])):
                print(f"\n   添加第{i+1}个prompt: {os.path.basename(img_path)}")
                print(f"   - 输入类别: {cls_arr.tolist() if hasattr(cls_arr, 'tolist') else cls_arr}")
                
                self._add_or_replace_prompt(img_path, bboxes, cls_arr, 1.0, is_initial=True)
                
                print(f"   - 添加后visual_prompts['cls']: {[arr.tolist() if hasattr(arr, 'tolist') else arr for arr in self.visual_prompts['cls']]}")
            
            # 4. 检查类别映射更新过程
            print(f"\n🗂️ 4. 类别映射更新过程:")
            print(f"   更新前类别映射: {self.class_mapping}")
            
            self._validate_and_update_class_mapping(self.visual_prompts['cls'])
            
            print(f"   更新后类别映射: {self.class_mapping}")
            print(f"   更新后索引映射: {self.index_to_class_id}")
            
            # 5. 检查VPE更新过程
            print(f"\n🧠 5. VPE更新过程:")
            vpe_update_success = self._update_vpe_with_new_annotations()
            print(f"   VPE更新成功: {vpe_update_success}")
            
            if vpe_update_success:
                print(f"   最终索引映射: {self.index_to_class_id}")
                print(f"   最终对象集合: {self.initial_object_set}")
                
                # 6. 验证映射正确性
                print(f"\n✅ 6. 映射正确性验证:")
                all_correct = True
                for vpe_idx, class_id in self.index_to_class_id.items():
                    expected_name = self.true_class_names[class_id] if (self.true_class_names and class_id < len(self.true_class_names)) else f"class_{class_id}"
                    actual_name = self.class_mapping.get(class_id, 'unknown')
                    is_correct = expected_name == actual_name
                    if not is_correct:
                        all_correct = False
                    
                    status = "✓" if is_correct else "✗"
                    print(f"   VPE索引{vpe_idx} -> 类别ID{class_id} -> 映射名称'{actual_name}' | 期望名称'{expected_name}' | {status}")
                
                print(f"\n   总体映射正确性: {'✓ 全部正确' if all_correct else '✗ 存在错误'}")
                
                # 7. 模拟预测过程
                print(f"\n🎯 7. 模拟预测映射:")
                print("   假设模型预测以下索引，检查映射结果:")
                
                for pred_idx in range(min(len(self.initial_object_set), 10)):  # 最多检查10个
                    if pred_idx in self.index_to_class_id:
                        mapped_class_id = self.index_to_class_id[pred_idx]
                        mapped_name = self.class_mapping.get(mapped_class_id, 'unknown')
                        expected_name = self.true_class_names[mapped_class_id] if (self.true_class_names and mapped_class_id < len(self.true_class_names)) else f"class_{mapped_class_id}"
                        
                        is_correct = mapped_name == expected_name
                        status = "✓" if is_correct else "✗"
                        
                        print(f"   预测索引{pred_idx} -> 类别ID{mapped_class_id} -> 名称'{mapped_name}' (期望:'{expected_name}') | {status}")
                        
                        if not is_correct:
                            print(f"     ⚠️ 错误映射！这会导致预测结果错误")
                    else:
                        print(f"   预测索引{pred_idx} -> ❌ 没有对应的类别ID")
                
        finally:
            # 恢复原始状态
            self.visual_prompts = original_visual_prompts
            self.class_mapping = original_class_mapping
            self.reverse_class_mapping = original_reverse_class_mapping
            self.index_to_class_id = original_index_mapping
            self.class_id_to_index = original_class_id_to_index
            if self.class_mapping:
                self.initial_object_set = original_initial_object_set
                self.num_classes = original_num_classes
        
        print("\n" + "🔍" * 80)
        print("🔍 实际数据流调试完成")
        print("🔍" * 80)

    def debug_vpe_construction_step_by_step(self, prompt_image_paths, visuals):
        """
        逐步调试VPE构建过程，特别关注索引映射
        """
        print("\n" + "🔧" * 80)
        print("🔧 逐步调试VPE构建过程")
        print("🔧" * 80)
        
        # 1. 检查输入数据
        print(f"\n📥 1. 输入数据:")
        print(f"   - 提示图片数量: {len(prompt_image_paths)}")
        for i, (img_path, bboxes, cls_arr) in enumerate(zip(prompt_image_paths, visuals['bboxes'], visuals['cls'])):
            print(f"   - 第{i+1}个prompt: {os.path.basename(img_path)}")
            print(f"     * 类别ID: {cls_arr.tolist() if hasattr(cls_arr, 'tolist') else cls_arr}")
            print(f"     * 对象数量: {len(cls_arr) if hasattr(cls_arr, '__len__') else 'N/A'}")
        
        # 2. 模拟类别映射构建过程
        print(f"\n🗂️ 2. 类别映射构建过程:")
        temp_class_mapping = {}
        temp_reverse_mapping = {}
        
        # 按照第一次出现的顺序收集类别ID（不排序）
        seen_cls_ids = set()
        ordered_unique_cls = []
        
        for cls_arr in visuals['cls']:
            if isinstance(cls_arr, np.ndarray) and cls_arr.size > 0:
                for cls_id in cls_arr.astype(int):
                    if cls_id not in seen_cls_ids:
                        seen_cls_ids.add(cls_id)
                        ordered_unique_cls.append(cls_id)
                        print(f"   发现新类别ID: {cls_id}")
        
        print(f"   按出现顺序的类别ID: {ordered_unique_cls}")
        
        # 构建类别映射
        for cls_id in ordered_unique_cls:
            if self.true_class_names and 0 <= cls_id < len(self.true_class_names):
                class_name = self.true_class_names[cls_id]
            else:
                class_name = f"object_{cls_id}"
            
            temp_class_mapping[cls_id] = class_name
            temp_reverse_mapping[class_name] = cls_id
            print(f"   映射: 类别ID {cls_id} -> 名称 '{class_name}'")
        
        # 3. 模拟VPE索引映射构建过程
        print(f"\n🔗 3. VPE索引映射构建过程:")
        
        # 方法1：按照类别ID排序（当前实现）
        sorted_class_ids = sorted(temp_class_mapping.keys())
        method1_index_to_class_id = {idx: class_id for idx, class_id in enumerate(sorted_class_ids)}
        print(f"   方法1（按类别ID排序）:")
        for idx, class_id in method1_index_to_class_id.items():
            class_name = temp_class_mapping[class_id]
            print(f"     VPE索引{idx} -> 类别ID{class_id} -> 名称'{class_name}'")
        
        # 方法2：按照出现顺序（可能更正确）
        method2_index_to_class_id = {idx: class_id for idx, class_id in enumerate(ordered_unique_cls)}
        print(f"   方法2（按出现顺序）:")
        for idx, class_id in method2_index_to_class_id.items():
            class_name = temp_class_mapping[class_id]
            print(f"     VPE索引{idx} -> 类别ID{class_id} -> 名称'{class_name}'")
        
        # 4. 检查两种方法的差异
        print(f"\n⚠️ 4. 两种方法的差异:")
        if method1_index_to_class_id == method2_index_to_class_id:
            print("   ✅ 两种方法结果相同")
        else:
            print("   ❌ 两种方法结果不同！这可能是问题所在")
            for idx in range(max(len(method1_index_to_class_id), len(method2_index_to_class_id))):
                cls1 = method1_index_to_class_id.get(idx, "N/A")
                cls2 = method2_index_to_class_id.get(idx, "N/A")
                if cls1 != cls2:
                    name1 = temp_class_mapping.get(cls1, "N/A") if cls1 != "N/A" else "N/A"
                    name2 = temp_class_mapping.get(cls2, "N/A") if cls2 != "N/A" else "N/A"
                    print(f"     索引{idx}: 方法1={cls1}('{name1}') vs 方法2={cls2}('{name2}')")
        
        # 5. 模拟预测映射
        print(f"\n🎯 5. 模拟预测映射:")
        print("   如果模型预测VPE索引0，会映射到:")
        
        if 0 in method1_index_to_class_id:
            cls_id = method1_index_to_class_id[0]
            cls_name = temp_class_mapping[cls_id]
            print(f"     方法1: 类别ID{cls_id} -> 名称'{cls_name}'")
        
        if 0 in method2_index_to_class_id:
            cls_id = method2_index_to_class_id[0]
            cls_name = temp_class_mapping[cls_id]
            print(f"     方法2: 类别ID{cls_id} -> 名称'{cls_name}'")
        
        # 6. 推荐解决方案
        print(f"\n💡 6. 问题分析和建议:")
        if ordered_unique_cls != sorted_class_ids:
            print("   ❌ 问题确认：类别ID的出现顺序与排序后的顺序不同")
            print(f"     出现顺序: {ordered_unique_cls}")
            print(f"     排序后: {sorted_class_ids}")
            print("   💡 建议：使用出现顺序而不是排序后的顺序来构建VPE索引映射")
            
            # 显示具体的修复效果
            print(f"\n   修复效果预览:")
            print(f"     当前实现（错误）: energydrink(ID=0) -> VPE索引{sorted_class_ids.index(0) if 0 in sorted_class_ids else 'N/A'}")
            print(f"     修复后（正确）: energydrink(ID=0) -> VPE索引{ordered_unique_cls.index(0) if 0 in ordered_unique_cls else 'N/A'}")
        else:
            print("   ✅ 类别ID的出现顺序与排序后的顺序相同，问题可能在其他地方")
        
        print("\n" + "🔧" * 80)
        print("🔧 VPE构建调试完成")
        print("🔧" * 80)

# It's good practice to have a main execution block for testing if this file is run directly.
# However, since you provided a file that seems like a module, I will not add an __main__ block here.
# You can copy the __main__ block from my previous complete code example if you need to test this file standalone.

def test_debug_class_mapping():
    """
    测试函数：创建一个简单的YOLOEWrapper实例并测试调试功能
    """
    print("🧪 测试调试功能...")
    
    # 创建测试实例
    wrapper = YOLOEWrapper(
        model_path="/Users/patrick/Desktop/labeling/pretrain/yoloe-11l-seg.pt",
        class_names=['energydrink', 'pepsiMax', 'cocacola']  # 示例类别名称
    )
    
    # 模拟添加一些visual prompts
    print("\n📝 添加测试visual prompts...")
    
    # 模拟第一个prompt：包含类别0和1
    test_bboxes_1 = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
    test_cls_1 = np.array([0, 1])
    wrapper._add_or_replace_prompt(
        "test_image_1.jpg", 
        test_bboxes_1, 
        test_cls_1, 
        0.9, 
        is_initial=True
    )
    
    # 模拟第二个prompt：包含类别1和2
    test_bboxes_2 = np.array([[20, 20, 80, 80], [90, 90, 150, 150]])
    test_cls_2 = np.array([1, 2])
    wrapper._add_or_replace_prompt(
        "test_image_2.jpg", 
        test_bboxes_2, 
        test_cls_2, 
        0.85, 
        is_initial=True
    )
    
    # 更新VPE
    print("\n🔄 更新VPE...")
    wrapper._update_vpe_with_new_annotations()
    
    # 运行调试检查
    print("\n🔍 运行调试检查...")
    wrapper.debug_class_mapping_and_vpe_alignment()
    
    return wrapper

if __name__ == "__main__":
    # 如果直接运行此文件，执行测试
    test_wrapper = test_debug_class_mapping()
    print("\n✅ 测试完成！你可以检查上面的输出来了解类别排列和VPE对应关系。")