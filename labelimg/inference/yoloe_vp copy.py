import sys
import os
import torch

# 添加 yoloe 目錄到 Python 路徑
yoloe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yoloe'))
if yoloe_path not in sys.path:
    sys.path.insert(0, yoloe_path)  # 使用 insert(0, ...) 確保優先使用 yoloe 目錄中的包
    print(f"Added yoloe path: {yoloe_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")

# 修改導入路徑
from ultralytics import YOLOE
import numpy as np
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

# --- Placeholder for actual YOLOE model and predictor ---
class MockYOLOEModel:
    def predict(self, images_or_target, prompts=None, predictor=None, conf=None):
        print(f"MockYOLOEModel.predict called with: images_or_target={images_or_target}, prompts_provided={prompts is not None}, predictor_exists={predictor is not None}, conf={conf}")
        if predictor: # 第一步调用，带VP-Seg predictor
            print("MockYOLOEModel: VP-Seg predict call (通常用于设置上下文或初步预测).")
            # YOLOE API可能要求此调用返回特定内容或修改模型状态
            # 此处返回空列表，表示此步骤不直接输出最终检测框
            return [] 
        elif conf is not None: # 第二步调用，获取最终检测框
            print("MockYOLOEModel: Final predict call for target image.")
            # 模拟目标图像的检测结果
            # 返回格式: List[Tuple[x1,y1,x2,y2,score,class_id]]
            return [
                (50.0, 50.0, 150.0, 150.0, 0.9, 0),  # x1, y1, x2, y2, score, class_id
                (70.0, 70.0, 180.0, 180.0, 0.85, 1),
            ]
        return []

class MockYOLOEVPSegPredictor:
    def __init__(self):
        print("MockYOLOEVPSegPredictor initialized")
# --- End of Placeholder ---

class YOLOEWrapper:
    def __init__(self, model_path="/Users/patrick/Desktop/labeling/pretrain/yoloe-11l-seg.pt"):
        """初始化 YOLOE 模型
        
        Args:
            model_path: YOLOE 模型路径
        """
        self.model = YOLOE(model_path)
        

        

    def old_auto_label_with_vp(self, prompt_image_paths: list, visuals: dict, target_image_path: str, conf_thresh: float = 0.4):
        """
        prompt_image_paths: list[str]，多个prompt图片路径
        visuals: {'bboxes': [np.array], 'cls': [np.array]}，格式与官方demo一致
        target_image_path: str
        conf_thresh: float
        """
        try:
            # --- 输入检查 ---
            print("==== YOLOE auto_label_with_vp DEBUG ====")
            print("prompt_image_paths:", prompt_image_paths)
            print("len(prompt_image_paths):", len(prompt_image_paths))
            print("len(visuals['bboxes']):", len(visuals['bboxes']))
            print("len(visuals['cls']):", len(visuals['cls']))
            for i, (b, c) in enumerate(zip(visuals['bboxes'], visuals['cls'])):
                print(f"  prompt {i}: bboxes.shape={b.shape}, cls.shape={c.shape}, bboxes.dtype={b.dtype}, cls.dtype={c.dtype}")
            print("target_image_path:", target_image_path)
            print("========================================")

            # 长度一致性检查
            if not (len(prompt_image_paths) == len(visuals['bboxes']) == len(visuals['cls'])):
                raise ValueError(f"prompt_image_paths, visuals['bboxes'], visuals['cls'] 长度不一致: {len(prompt_image_paths)}, {len(visuals['bboxes'])}, {len(visuals['cls'])}")

            # shape 检查

            # 1. 先对prompt images做VPE推理
            self.model.predictor = None
            self.model.predict(
                prompt_image_paths,
                prompts=visuals,
                predictor=YOLOEVPSegPredictor,
                return_vpe=True
            )
            # 2. 直接从 self.model.predictor.vpe 取 embedding
            vpe_tensor = self.model.predictor.vpe  # [num_prompt, max_cls_per_prompt, dim]
            cls_list = visuals['cls']  # list of np.array，每個 prompt image 的 class id

            all_cls = np.concatenate(cls_list)
            num_classes = int(all_cls.max()) + 1 if all_cls.size > 0 else 1
            dim = vpe_tensor.shape[-1]
            embeddings = []

            for class_id in range(num_classes):
                emb_list = []
                for i, cls_arr in enumerate(cls_list):
                    for j, c in enumerate(cls_arr):
                        if c == class_id and j < vpe_tensor.shape[1]:
                            emb_list.append(vpe_tensor[i, j])
                if emb_list:
                    emb = torch.stack(emb_list, dim=0).mean(dim=0, keepdim=True)
                    emb = F.normalize(emb, dim=-1, p=2)
                    embeddings.append(emb)
                else:
                    embeddings.append(torch.zeros((1, dim), dtype=vpe_tensor.dtype, device=vpe_tensor.device))

            embeddings = torch.cat(embeddings, dim=0)  # [num_classes, dim]
            embeddings = embeddings.unsqueeze(0)       # [1, num_classes, dim]
            object_set = [f"object{i}" for i in range(num_classes)]
            self.model.set_classes(object_set, embeddings)
            
            # 4. 对target image做普通推理
            self.model.predictor = None 
          
            results = self.model.predict(
                target_image_path,
                save=False
            )
                    
            predictions = []
            if results and hasattr(results[0], 'boxes'):
                boxes = getattr(results[0], 'boxes', [])
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        score = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        if score >= conf_thresh:
                            predictions.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'class_id': class_id,
                                'confidence': score
                            })
            return predictions
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"YOLOE预测错误: {e}")
            # 额外打印输入，方便定位
            print("[ERROR DEBUG] prompt_image_paths:", prompt_image_paths)
            print("[ERROR DEBUG] visuals['bboxes']:", [b.shape for b in visuals['bboxes']])
            print("[ERROR DEBUG] visuals['cls']:", [c.shape for c in visuals['cls']])
            print("[ERROR DEBUG] target_image_path:", target_image_path)
            return []