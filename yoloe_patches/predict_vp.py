from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.data.augment import LetterBox, LoadVisualPrompt
from ultralytics.utils.instance import Instances
import numpy as np
import torch
from copy import deepcopy

from ultralytics.utils.torch_utils import select_device

class YOLOEVPPredictorMixin:
    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        self.model = model.to(device)

        self.device = device  # update device
        self.model.fp16 = False
        self.args.half = False
        self.model.eval()
        
        self.done_warmup = True
        self.return_vpe = False
        
    def set_return_vpe(self, return_vpe):
        self.return_vpe = return_vpe
    
    def set_prompts(self, prompts):
        self.prompts = deepcopy(prompts)
    
    def load_vp(self, label):
        label["img"] = label["img"].transpose(2, 0, 1)
        load_vp = LoadVisualPrompt(nc=len(label["cls"]), augment=False)
        label = load_vp(label)
        label["img"] = label["img"].transpose(1, 2, 0)
        return label
    
    def process_box_label(self, img, bboxes, cls, letterbox):
        label = dict(
            img=img,
            instances=Instances(bboxes=bboxes.astype(np.float32), 
                                segments=np.zeros((0, 1000, 2), dtype=np.float32), 
                                bbox_format="xyxy", normalized=False),
            cls=torch.tensor(cls).unsqueeze(-1)
        )
        label = letterbox(label)
        instances = label.pop("instances")
        h, w = label["img"].shape[:2]
        instances.normalize(w, h)
        instances.convert_bbox(format="xywh")
        label["bboxes"] = torch.from_numpy(instances.bboxes)
        return self.load_vp(label)
    
    def process_mask_label(self, img, masks, cls, letterbox):
        img = letterbox(image=img)
        masks = np.stack([letterbox(image=mask) for mask in masks])
        masks[masks == 114] = 0
        label = dict(
            img=img,
            masks=masks,
            cls=torch.tensor(cls).unsqueeze(-1)
        )
        return self.load_vp(label)
        
    def create_class_count_mask(self, cls_arrays, unified_nc):
        """创建类别计数掩码，记录每个类别在多少个prompt中出现
        
        Args:
            cls_arrays: list of np.array，每个prompt的类别ID数组
            unified_nc: 统一的类别数量
            
       pad_and_align_vpe Returns:
            torch.Tensor: [unified_nc] 计数掩码，记录每个类别出现的次数
        """
        class_count_mask = torch.zeros(unified_nc, dtype=torch.float32)
        
        for cls_array in cls_arrays:
            if isinstance(cls_array, np.ndarray) and cls_array.size > 0:
                unique_classes = np.unique(cls_array)
                for class_id in unique_classes:
                    if 0 <= class_id < unified_nc:
                        class_count_mask[class_id] += 1.0
        
        print(f"[Debug] 类别计数掩码: {class_count_mask}")
        for i in range(unified_nc):
            if class_count_mask[i] > 0:
                print(f"[Debug] 类别{i}: 出现在{int(class_count_mask[i])}个prompt中")
                
        return class_count_mask
    
    def combine_vpe_by_class_averaging_correct(self, raw_vpe, class_count_mask):
        """正确的通道级别到类别级别VPE转换
        
        基于实际的VPE形状：[batch, num_channels, embed_dim]
        其中 num_channels = num_prompts * nc
        
        Args:
            raw_vpe: torch.Tensor [batch, num_channels, embed_dim] 通道级别的VPE
            class_count_mask: torch.Tensor [nc] 类别计数掩码
            
        Returns:
            torch.Tensor: [batch, nc, embed_dim] 类别级别的VPE
        """
        batch_size, num_channels, embed_dim = raw_vpe.shape
        
        # 计算统一的类别数量和prompt数量
        nc = len(class_count_mask)
        num_prompts = len(self.prompts_cls)
        expected_channels = num_prompts * nc
        
        print(f"[Debug] 通道级别到类别级别VPE转换:")
        print(f"  输入VPE形状: {raw_vpe.shape}")
        print(f"  预期通道数: {expected_channels} (num_prompts={num_prompts} × nc={nc})")
        print(f"  实际通道数: {num_channels}")
        print(f"  类别计数: {class_count_mask}")
        
        if num_channels != expected_channels:
            print(f"[Warning] 通道数不匹配，使用实际通道数进行处理")
            # 重新计算nc，假设所有prompt都有相同的类别数
            nc = num_channels // num_prompts
            print(f"[Debug] 重新计算的nc: {nc}")
        
        # 创建类别级别的VPE
        combined_vpe = torch.zeros(batch_size, nc, embed_dim, 
                                 device=raw_vpe.device, dtype=raw_vpe.dtype)
        
        print(f"[Debug] 通道到类别映射:")
        
        # 对每个类别进行处理
        for class_id in range(nc):
            # 找到该类别对应的所有通道
            class_channels = []
            class_weights = []
            
            for prompt_idx in range(num_prompts):
                channel_idx = prompt_idx * nc + class_id
                
                if channel_idx < num_channels:
                    # 检查该prompt是否包含这个类别
                    if prompt_idx < len(self.prompts_cls):
                        cls_array = self.prompts_cls[prompt_idx]
                        if isinstance(cls_array, np.ndarray) and class_id in cls_array:
                            class_channels.append(channel_idx)
                            class_weights.append(1.0)
                            print(f"    类别{class_id}: 通道{channel_idx} (来自Prompt {prompt_idx}) ✓")
                        else:
                            print(f"    类别{class_id}: 通道{channel_idx} (来自Prompt {prompt_idx}) ✗ (不包含该类别)")
                    else:
                        # 如果没有cls信息，假设所有通道都有效
                        class_channels.append(channel_idx)
                        class_weights.append(1.0)
                        print(f"    类别{class_id}: 通道{channel_idx} (来自Prompt {prompt_idx}) ? (无cls信息)")
            
            # 对该类别的所有有效通道进行加权平均
            if class_channels:
                class_weights = torch.tensor(class_weights, device=raw_vpe.device, dtype=raw_vpe.dtype)
                class_weights = class_weights / class_weights.sum()  # 归一化
                
                class_vpe = torch.zeros(batch_size, embed_dim, device=raw_vpe.device, dtype=raw_vpe.dtype)
                for i, channel_idx in enumerate(class_channels):
                    class_vpe += raw_vpe[:, channel_idx, :] * class_weights[i]
                
                combined_vpe[:, class_id, :] = class_vpe
                
                print(f"    类别{class_id}: 使用{len(class_channels)}个通道进行平均")
                print(f"      通道索引: {class_channels}")
                print(f"      权重: {class_weights.tolist()}")
                print(f"      VPE范数: {torch.norm(class_vpe).item():.4f}")
            else:
                print(f"    类别{class_id}: 无有效通道，保持零值")
        
        print(f"[Debug] 类别级别VPE转换完成，输出形状: {combined_vpe.shape}")
        return combined_vpe
    
    def combine_vpe_overall_averaging(self, raw_vpe, class_count_mask):
        """整体平均方案（备用方案）
        
        当VPE维度不能按类别分割时使用
        """
        batch_size, num_prompts, embed_dim = raw_vpe.shape
        
        print(f"[Debug] 使用整体平均方案: {raw_vpe.shape} -> [batch, 1, {embed_dim}]")
        
        # 计算每个prompt的权重（基于它包含的有效类别数量）
        prompt_weights = []
        for prompt_idx, cls_array in enumerate(self.prompts_cls):
            if isinstance(cls_array, np.ndarray) and cls_array.size > 0:
                # 权重 = 该prompt包含的有效类别数量
                valid_classes = [c for c in np.unique(cls_array) if class_count_mask[c] > 0]
                weight = len(valid_classes)
                prompt_weights.append(weight)
                print(f"[Debug] Prompt {prompt_idx}: 包含{weight}个有效类别")
            else:
                prompt_weights.append(0.0)
                print(f"[Debug] Prompt {prompt_idx}: 无有效类别")
        
        # 转换为tensor并归一化
        prompt_weights = torch.tensor(prompt_weights, device=raw_vpe.device, dtype=raw_vpe.dtype)
        if prompt_weights.sum() > 0:
            prompt_weights = prompt_weights / prompt_weights.sum()
        else:
            prompt_weights = torch.ones(num_prompts, device=raw_vpe.device) / num_prompts
        
        print(f"[Debug] 归一化权重: {prompt_weights}")
        
        # 加权平均
        combined_vpe = torch.zeros(batch_size, 1, embed_dim, device=raw_vpe.device, dtype=raw_vpe.dtype)
        for prompt_idx in range(num_prompts):
            combined_vpe[:, 0, :] += raw_vpe[:, prompt_idx, :] * prompt_weights[prompt_idx]
        
        # 归一化
        combined_vpe = torch.nn.functional.normalize(combined_vpe, dim=-1, p=2)
        
        print(f"[Debug] 整体平均VPE结合完成，输出形状: {combined_vpe.shape}")
        return combined_vpe

    def pre_transform(self, im):
        letterbox = LetterBox(
            self.imgsz,
            auto=False,
            stride=int(self.model.stride[-1].item()),
        )

        cls = self.prompts["cls"]
        cls = [cls] if not isinstance(cls, list) else cls
        
        # 🚀 保存原始cls信息，用于后续VPE结合
        self.prompts_cls = cls
        
        # 🔧 新增：统一类别空间处理
        # 1. 找到所有prompt中的最大类别ID，确定统一的类别空间大小
        all_class_ids = set()
        for cls_array in cls:
            if isinstance(cls_array, np.ndarray) and cls_array.size > 0:
                all_class_ids.update(cls_array.tolist())
        
        if not all_class_ids:
            raise ValueError("No valid class IDs found in prompts")
        
        max_class_id = max(all_class_ids)
        unified_nc = max_class_id + 1  # 统一的类别数量
        
        print(f"[Debug] 统一类别空间: 发现类别ID {sorted(all_class_ids)}, 统一nc={unified_nc}")
        
        # 🚀 新增：创建类别计数掩码，用于后续VPE结合
        self.class_count_mask = self.create_class_count_mask(cls, unified_nc)
        self.unified_nc = unified_nc
            
        if "bboxes" in self.prompts:
            bboxes = self.prompts["bboxes"]
            bboxes = [bboxes] if not isinstance(bboxes, list) else bboxes
            
            # 🔧 修改：为每个prompt单独处理，保持独立
            labels = []
            self.individual_prompts = []  # 保存每个prompt的visual prompts
            self.prompt_unique_cls = []   # 🔧 新增：保存每个prompt的unique类别
            for i in range(len(im)):
                label = self.process_box_label_individual(im[i], bboxes[i], cls[i], letterbox)
                labels.append(label)
                self.individual_prompts.append(label["visuals"])
                self.prompt_unique_cls.append(label["unique_cls"])  # 🔧 保存unique类别
                
        elif "masks" in self.prompts:
            masks = self.prompts["masks"]
            masks = [masks] if not isinstance(masks, list) else masks
            
            # 🔧 修改：为每个prompt单独处理，保持独立
            labels = []
            self.individual_prompts = []  # 保存每个prompt的visual prompts
            self.prompt_unique_cls = []   # 🔧 新增：保存每个prompt的unique类别
            for i in range(len(im)):
                label = self.process_mask_label_individual(im[i], masks[i], cls[i], letterbox)
                labels.append(label)
                self.individual_prompts.append(label["visuals"])
                self.prompt_unique_cls.append(label["unique_cls"])  # 🔧 保存unique类别
        else:
            raise ValueError("Please provide valid bboxes or masks")

        # 🔧 关键修改：不合并prompts，保持独立处理
        print(f"[Debug] 保存{len(self.individual_prompts)}个独立的visual prompts")
        for i, (prompt, unique_cls) in enumerate(zip(self.individual_prompts, self.prompt_unique_cls)):
            print(f"  Prompt {i}: {prompt.shape}, unique类别: {unique_cls}")
        
        # 使用统一的类别数量
        self.model.model[-1].nc = unified_nc
        self.model.names = [f"object{i}" for i in range(unified_nc)]
        
        return [label["img"] for label in labels]
    
    def process_box_label_individual(self, img, bboxes, cls, letterbox):
        """处理单个prompt的bbox标签，使用unique类别数量"""
        # 🔧 关键修正：计算unique类别数量
        unique_cls = np.unique(cls)
        unique_nc = len(unique_cls)
        
        print(f"[Debug] Prompt处理: {len(cls)}个object, {len(unique_cls)}个unique类别")
        print(f"[Debug] Object类别: {cls}")
        print(f"[Debug] Unique类别: {unique_cls}")
        
        label = dict(
            img=img,
            instances=Instances(bboxes=bboxes.astype(np.float32), 
                                segments=np.zeros((0, 1000, 2), dtype=np.float32), 
                                bbox_format="xyxy", normalized=False),
            cls=torch.tensor(cls).unsqueeze(-1)
        )
        label = letterbox(label)
        instances = label.pop("instances")
        h, w = label["img"].shape[:2]
        instances.normalize(w, h)
        instances.convert_bbox(format="xywh")
        label["bboxes"] = torch.from_numpy(instances.bboxes)
        
        # 🔧 传递unique类别信息
        return self.load_vp_individual(label, unique_nc, unique_cls)
    
    def process_mask_label_individual(self, img, masks, cls, letterbox):
        """处理单个prompt的mask标签，使用unique类别数量"""
        # 🔧 关键修正：计算unique类别数量
        unique_cls = np.unique(cls)
        unique_nc = len(unique_cls)
        
        print(f"[Debug] Prompt处理: {len(cls)}个object, {len(unique_cls)}个unique类别")
        print(f"[Debug] Object类别: {cls}")
        print(f"[Debug] Unique类别: {unique_cls}")
        
        img = letterbox(image=img)
        masks = np.stack([letterbox(image=mask) for mask in masks])
        masks[masks == 114] = 0
        label = dict(
            img=img,
            masks=masks,
            cls=torch.tensor(cls).unsqueeze(-1)
        )
        
        # 🔧 传递unique类别信息
        return self.load_vp_individual(label, unique_nc, unique_cls)
        
    def load_vp_individual(self, label, unique_nc, unique_cls):
        """使用unique类别数量加载visual prompt"""
        print(f"[Debug] LoadVisualPrompt: unique_nc={unique_nc}, unique类别={unique_cls}")
        print(f"[Debug] 原始类别数组长度={len(label['cls'])}, 内容={label['cls'].flatten()}")
        
        label["img"] = label["img"].transpose(2, 0, 1)
        # 🔧 关键修改：使用unique类别数量
        load_vp = LoadVisualPrompt(nc=unique_nc, augment=False)
        label = load_vp(label)
        label["img"] = label["img"].transpose(1, 2, 0)
        
        print(f"[Debug] LoadVisualPrompt输出形状: {label['visuals'].shape}")
        
        # 🔧 保存unique类别信息，用于后续填充
        label["unique_cls"] = unique_cls
        return label

    def get_individual_vpe(self, im, prompt_visual):
        """获取单个prompt的VPE"""
        # 将单个prompt转换为batch格式
        prompt_batch = prompt_visual.unsqueeze(0).to(self.device)  # [1, nc, H, W]
        
        # 获取VPE
        vpe = self.model.get_visual_pe(im, visual=prompt_batch)  # [1, nc, 512]
        return vpe

    def pad_and_align_vpe(self, vpe, unique_cls, unified_nc):
        """Fixed version with proper class alignment"""
        batch_size, vpe_nc, embed_dim = vpe.shape
        
        # Create aligned VPE tensor
        aligned_vpe = torch.zeros(batch_size, unified_nc, embed_dim, 
                                device=vpe.device, dtype=vpe.dtype)
        
        # Handle case where we have more VPE channels than unique classes
        num_classes_to_map = min(vpe_nc, len(unique_cls))
        
        for vpe_idx in range(num_classes_to_map):
            if vpe_idx < len(unique_cls):
                class_id = unique_cls[vpe_idx]
                if 0 <= class_id < unified_nc:
                    aligned_vpe[:, class_id, :] = vpe[:, vpe_idx, :]
        
        return aligned_vpe

    def combine_vpe_by_class_averaging_correct(self, individual_vpes, class_count_mask):
        """正确的类别级别VPE加权平均
        
        Args:
            individual_vpes: List[torch.Tensor] 每个prompt的VPE，已对齐到统一类别空间
            class_count_mask: torch.Tensor [nc] 类别计数掩码
            
        Returns:
            torch.Tensor: [batch, nc, embed_dim] 加权平均后的VPE
        """
        if not individual_vpes:
            raise ValueError("No individual VPEs provided")
        
        batch_size, nc, embed_dim = individual_vpes[0].shape
        
        print(f"[Debug] 类别级别VPE加权平均:")
        print(f"  输入: {len(individual_vpes)}个VPE，每个形状: {individual_vpes[0].shape}")
        print(f"  类别计数: {class_count_mask}")
        
        # 创建结果VPE
        combined_vpe = torch.zeros(batch_size, nc, embed_dim, 
                                 device=individual_vpes[0].device, 
                                 dtype=individual_vpes[0].dtype)
        
        # 对每个类别进行加权平均
        for class_id in range(nc):
            class_vpes = []
            class_weights = []
            
            # 收集该类别在各个prompt中的VPE
            for prompt_idx, vpe in enumerate(individual_vpes):
                cls_array = self.prompts_cls[prompt_idx]
                
                # 检查该prompt是否包含这个类别
                if isinstance(cls_array, np.ndarray) and class_id in cls_array:
                    class_vpes.append(vpe[:, class_id, :])  # [batch, embed_dim]
                    class_weights.append(1.0)
                    print(f"    类别{class_id}: 来自Prompt {prompt_idx} ✓")
            
            # 进行加权平均
            if class_vpes:
                # 转换为tensor并归一化权重
                class_weights = torch.tensor(class_weights, 
                                           device=individual_vpes[0].device, 
                                           dtype=individual_vpes[0].dtype)
                class_weights = class_weights / class_weights.sum()
                
                # 加权平均
                class_vpe_avg = torch.zeros_like(class_vpes[0])
                for i, class_vpe in enumerate(class_vpes):
                    class_vpe_avg += class_vpe * class_weights[i]
                
                combined_vpe[:, class_id, :] = class_vpe_avg
                
                print(f"    类别{class_id}: 使用{len(class_vpes)}个VPE进行加权平均")
                print(f"      权重: {class_weights.tolist()}")
                print(f"      VPE范数: {torch.norm(class_vpe_avg).item():.4f}")
            else:
                print(f"    类别{class_id}: 无VPE，保持零值")
        
        print(f"[Debug] 类别级别VPE加权平均完成，输出形状: {combined_vpe.shape}")
        return combined_vpe

    def inference(self, im, *args, **kwargs):
        if self.return_vpe:
            print(f"[Debug] 开始单独处理每个prompt的VPE")
            
            # 🔧 关键修改：单独处理每个prompt
            individual_vpes = []
            
            for prompt_idx, (prompt_visual, unique_cls) in enumerate(zip(self.individual_prompts, self.prompt_unique_cls)):
                print(f"[Debug] 处理Prompt {prompt_idx}, 形状: {prompt_visual.shape}, unique类别: {unique_cls}")
                
                # 获取单个prompt的VPE
                vpe = self.get_individual_vpe(im, prompt_visual)  # [1, unique_nc, 512]
                print(f"[Debug] Prompt {prompt_idx} VPE形状: {vpe.shape}")
                
                # 填充到统一类别空间
                aligned_vpe = self.pad_and_align_vpe(vpe, unique_cls, self.unified_nc)
                print(f"[Debug] Prompt {prompt_idx} 对齐后VPE形状: {aligned_vpe.shape}")
                
                individual_vpes.append(aligned_vpe)
            
            # 🎯 按类别进行加权平均
            self.vpe = self.combine_vpe_by_class_averaging_correct(individual_vpes, self.class_count_mask)
            print(f"[Debug] 最终VPE形状: {self.vpe.shape}")
        
        # 注意：这里需要传入合并后的visual prompts用于实际推理
        # 临时合并prompts用于推理
        merged_prompts = torch.nn.utils.rnn.pad_sequence(self.individual_prompts, batch_first=True).to(self.device)
        return super().inference(im, vpe=merged_prompts, *args, **kwargs)

class YOLOEVPDetectPredictor(YOLOEVPPredictorMixin, DetectionPredictor):
    pass

class YOLOEVPSegPredictor(YOLOEVPPredictorMixin, SegmentationPredictor):
    pass