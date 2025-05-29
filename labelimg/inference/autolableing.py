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
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor


class YOLOEWrapper:
    """YOLOE模型包装器，用于视觉提示自动标注"""
    
    def __init__(self, model_path: str = "pretrain/yoloe-v8l-seg.pt", class_names: List[str] = None):
        """初始化YOLOE包装器
        
        Args:
            model_path: YOLOE模型文件路径
            class_names: 类别名称列表
        """
        self.model_path = model_path
        self.model = None
        self.true_class_names = class_names or []
        self.vpe = None  # Visual Prompt Encoder
        
        # 延迟加载模型
        self._load_model()
        
    def _load_model(self):
        """加载YOLOE模型"""
        try:
            print(f"[Debug] 正在加载YOLOE模型: {self.model_path}")
            self.model = YOLOE(self.model_path)
            print(f"[Debug] YOLOE模型加载成功")
        except Exception as e:
            print(f"[Error] 加载YOLOE模型失败: {e}")
            raise
    
    def auto_label_with_vp(self, 
                          source_image, 
                          visuals: Dict[str, Any], 
                          target_image: str,
                          conf_threshold: float = 0.25,
                          min_confidence: float = 0.25,
                          progress_callback=None) -> List[Dict[str, Any]]:
        """使用视觉提示进行自动标注
        
        Args:
            source_image: 源图像路径（提示图像），可以是字符串或列表
            visuals: 视觉提示字典，包含bboxes和cls
            target_image: 目标图像路径
            conf_threshold: YOLO模型的置信度阈值
            min_confidence: 最终结果的最小置信度阈值
            progress_callback: 进度回调函数，接收 (current_step, total_steps, message) 参数
            
        Returns:
            List[Dict]: 预测结果列表，每个元素包含bbox、class_id、confidence
        """
        try:
            # 如果 source_image 是列表，取第一个元素
            if isinstance(source_image, list):
                source_image_path = source_image[0]
            else:
                source_image_path = source_image
                
            print(f"[Debug] 开始视觉提示自动标注")
            print(f"[Debug] 源图像: {source_image_path}")
            print(f"[Debug] 目标图像: {target_image}")
            print(f"[Debug] 视觉提示: {visuals}")
            print(f"[Debug] 置信度阈值: {conf_threshold}")
            
            # 报告进度：步骤1
            if progress_callback:
                progress_callback(1, 4, "Validating input files...")
            
            # 验证输入
            if not os.path.exists(source_image_path):
                raise FileNotFoundError(f"源图像不存在: {source_image_path}")
            if not os.path.exists(target_image):
                raise FileNotFoundError(f"目标图像不存在: {target_image}")
            
            # 报告进度：步骤2
            if progress_callback:
                progress_callback(2, 4, "Building visual prompt encoder...")
            
            # 使用新的VPE初始化方法
            vpe_success = self._init_vpe_with_prompts(source_image_path, visuals, conf_threshold)
            
            if not vpe_success:
                print(f"[Warning] VPE初始化失败，使用备用方法")
                # 备用方案：直接在目标图像上进行预测
                try:
                    # 报告进度：步骤3（跳过VPE设置）
                    if progress_callback:
                        progress_callback(3, 4, "Using fallback prediction method...")
                    
                    # 报告进度：步骤4
                    if progress_callback:
                        progress_callback(4, 4, "Executing target image prediction...")
                    
                    # 直接预测目标图像
                    print(f"[Debug] 备用方案: 直接预测目标图像")
                    target_results = self.model.predict(
                        target_image, 
                        save=False,
                        conf=conf_threshold
                    )
                    
                    # 解析预测结果
                    predictions = self._parse_predictions(target_results, min_confidence)
                    print(f"[Debug] 备用方案解析得到 {len(predictions)} 个预测结果")
                    
                    return predictions
                    
                except Exception as fallback_error:
                    print(f"[Error] 备用方案也失败了: {fallback_error}")
                    raise RuntimeError(f"VPE获取失败且备用方案也失败: {fallback_error}")
            
            print(f"[Debug] VPE初始化成功，继续正常流程")
            
            # 报告进度：步骤3
            if progress_callback:
                progress_callback(3, 4, "Setting classes and preparing prediction...")
            
            # 第二步：设置类别并清除predictor
            print(f"[Debug] 步骤2: 设置类别并清除predictor")
            if self.true_class_names:
                self.model.set_classes(self.true_class_names, self.vpe)
                print(f"[Debug] 已设置类别: {self.true_class_names}")
            else:
                # 如果没有提供类别名称，使用默认类别
                default_classes = [f"object{i}" for i in range(len(visuals.get('cls', [[]])[0]))]
                self.model.set_classes(default_classes, self.vpe)
                print(f"[Debug] 使用默认类别: {default_classes}")
            
            # 清除predictor以使用标准预测
            self.model.predictor = None
            
            # 报告进度：步骤4
            if progress_callback:
                progress_callback(4, 4, "Executing target image prediction...")
            
            # 第三步：在目标图像上进行预测
            print(f"[Debug] 步骤3: 在目标图像上进行预测")
            target_results = self.model.predict(
                target_image, 
                save=False,
                conf=conf_threshold
            )
            
            # 解析预测结果
            predictions = self._parse_predictions(target_results, min_confidence)
            print(f"[Debug] 解析得到 {len(predictions)} 个预测结果")
            
            return predictions
            
        except Exception as e:
            print(f"[Error] 视觉提示自动标注失败: {e}")
            raise
    
    def _parse_predictions(self, results, min_confidence: float) -> List[Dict[str, Any]]:
        """解析YOLOE预测结果
        
        Args:
            results: YOLOE预测结果
            min_confidence: 最终结果的最小置信度阈值
            
        Returns:
            List[Dict]: 解析后的预测结果
        """
        predictions = []
        
        try:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    # 获取边界框坐标 (xyxy格式)
                    if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                        xyxy = boxes.xyxy.cpu().numpy()
                    else:
                        continue
                    
                    # 获取置信度
                    if hasattr(boxes, 'conf') and boxes.conf is not None:
                        confidences = boxes.conf.cpu().numpy()
                    else:
                        confidences = np.ones(len(xyxy))
                    
                    # 获取类别ID
                    if hasattr(boxes, 'cls') and boxes.cls is not None:
                        class_ids = boxes.cls.cpu().numpy().astype(int)
                    else:
                        class_ids = np.zeros(len(xyxy), dtype=int)
                    
                    # 转换为预测结果格式
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        confidence = float(confidences[i])
                        
                        # 只保留置信度超过 0.4 的预测结果
                        if confidence < min_confidence:
                            print(f"[Debug] 跳过低置信度预测: confidence={confidence:.3f}")
                            continue
                        
                        # 转换为 [x, y, width, height] 格式
                        bbox = [
                            float(x1),
                            float(y1), 
                            float(x2 - x1),
                            float(y2 - y1)
                        ]
                        
                        prediction = {
                            'bbox': bbox,
                            'class_id': int(class_ids[i]),
                            'confidence': confidence
                        }
                        
                        predictions.append(prediction)
                        print(f"[Debug] 添加预测: {prediction}")
        
        except Exception as e:
            print(f"[Error] 解析预测结果时出错: {e}")
            
        return predictions
    
    def update_class_names(self, class_names: List[str]):
        """更新类别名称
        
        Args:
            class_names: 新的类别名称列表
        """
        self.true_class_names = class_names
        print(f"[Debug] 类别名称已更新: {class_names}")
    
    def reset_vpe(self):
        """重置VPE"""
        self.vpe = None
        if self.model:
            self.model.predictor = None
        print(f"[Debug] VPE已重置")
    
    def _init_vpe_with_prompts(self, source_image_path: str, visuals: Dict[str, Any], conf_threshold: float = 0.25):
        """专门用于初始化VPE的方法
        
        Args:
            source_image_path: 源图像路径
            visuals: 视觉提示
            conf_threshold: 置信度阈值
            
        Returns:
            bool: 是否成功初始化VPE
        """
        print(f"[Debug] 开始初始化VPE...")
        
        try:
            # 方法1: 使用YOLOEVPSegPredictor直接创建
            print(f"[Debug] 尝试方法1: 直接创建YOLOEVPSegPredictor")
            
            # 重置predictor
            self.model.predictor = None
            
            # 创建predictor实例
            predictor = YOLOEVPSegPredictor(overrides=self.model.overrides)
            predictor.setup_model(model=self.model.model)
            
            # 尝试处理prompts
            if hasattr(predictor, 'setup_source'):
                predictor.setup_source(source_image_path)
            
            # 设置prompts
            if hasattr(predictor, 'set_prompts'):
                predictor.set_prompts(visuals)
            elif hasattr(predictor, 'prompts'):
                predictor.prompts = visuals
            
            # 尝试获取或创建VPE
            if hasattr(predictor, 'vpe') and predictor.vpe is not None:
                self.vpe = predictor.vpe
                self.model.predictor = predictor
                print(f"[Debug] 方法1成功: 从predictor获取VPE")
                return True
            
            # 方法2: 尝试运行一次预测来触发VPE创建
            print(f"[Debug] 尝试方法2: 运行预测触发VPE创建")
            
            # 使用原始方法但添加更多检查
            results = self.model.predict(
                source_image_path,
                save=False,
                prompts=visuals,
                predictor=YOLOEVPSegPredictor,
                return_vpe=True,
                conf=conf_threshold,
                verbose=True  # 启用详细输出
            )
            
            # 检查多个可能的位置
            vpe_sources = [
                ('model.predictor.vpe', lambda: getattr(self.model.predictor, 'vpe', None) if hasattr(self.model, 'predictor') and self.model.predictor else None),
                ('model.vpe', lambda: getattr(self.model, 'vpe', None)),
                ('results[0].vpe', lambda: getattr(results[0], 'vpe', None) if results else None),
            ]
            
            for source_name, getter in vpe_sources:
                try:
                    vpe = getter()
                    if vpe is not None:
                        self.vpe = vpe
                        print(f"[Debug] 方法2成功: 从{source_name}获取VPE")
                        return True
                except Exception as e:
                    print(f"[Debug] 从{source_name}获取VPE失败: {e}")
            
            # 方法3: 手动创建VPE（如果有相关类）
            print(f"[Debug] 尝试方法3: 手动创建VPE")
            try:
                # 这里需要根据实际的VPE类来实现
                # 暂时跳过，因为不确定具体的VPE类结构
                pass
            except Exception as e:
                print(f"[Debug] 方法3失败: {e}")
            
            print(f"[Debug] 所有VPE初始化方法都失败了")
            return False
            
        except Exception as e:
            print(f"[Error] VPE初始化过程中出错: {e}")
            return False 