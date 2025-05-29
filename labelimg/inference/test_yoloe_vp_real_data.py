#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import glob

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from yoloe_vp import YOLOEWrapper
    print("✓ 成功导入 YOLOEWrapper")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def load_classes(classes_path):
    """加载类别名称"""
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

def parse_yolo_annotation(txt_path, img_width=640, img_height=480):
    """解析YOLO格式的标注文件
    
    Args:
        txt_path: 标注文件路径
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        bboxes: [[x1, y1, x2, y2], ...] 格式的边界框
        class_ids: [class_id, ...] 格式的类别ID
    """
    bboxes = []
    class_ids = []
    
    if not os.path.exists(txt_path):
        return np.array([]), np.array([])
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 5:
            continue
        
        class_id = int(parts[0])
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
        class_ids.append(class_id)
    
    return np.array(bboxes), np.array(class_ids)

def test_with_real_data():
    """使用set16_test的真实数据测试类别管理"""
    print("\n=== 使用真实数据测试类别管理稳定性 ===")
    
    # 数据路径
    data_dir = "/Users/patrick/Desktop/labeling/Bunnpris-data/set16_test"
    classes_path = os.path.join(data_dir, "classes.txt")
    
    # 加载类别名称
    class_names = load_classes(classes_path)
    print(f"加载类别: {class_names}")
    print(f"总类别数: {len(class_names)}")
    
    # 获取所有图像文件
    image_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 初始化wrapper
    wrapper = YOLOEWrapper()
    print(f"初始状态 - 类别数量: {wrapper.num_classes}, 类别映射: {wrapper.class_mapping}")
    
    # 选择前几个文件作为初始prompt
    initial_prompt_count = 3
    prompt_images = image_files[:initial_prompt_count]
    
    print(f"\n--- 使用前{initial_prompt_count}个图像作为初始prompt ---")
    
    # 收集初始prompts的数据
    initial_bboxes = []
    initial_cls = []
    valid_prompts = []
    
    for img_path in prompt_images:
        # 获取对应的标注文件
        txt_path = img_path.replace('.jpg', '.txt')
        bboxes, cls_ids = parse_yolo_annotation(txt_path)
        
        if len(bboxes) > 0 and len(cls_ids) > 0:
            initial_bboxes.append(bboxes)
            initial_cls.append(cls_ids)
            valid_prompts.append(img_path)
            print(f"  {os.path.basename(img_path)}: {len(cls_ids)} 个对象, 类别: {cls_ids}")
    
    if not valid_prompts:
        print("❌ 没有找到有效的初始prompt数据")
        return None
    
    # 使用auto_label_with_vp进行初始化
    print("\n--- 初始化VPE ---")
    visuals = {
        'bboxes': initial_bboxes,
        'cls': initial_cls
    }
    
    # 选择一个目标图像进行测试
    target_image = image_files[initial_prompt_count] if len(image_files) > initial_prompt_count else image_files[-1]
    
    try:
        predictions = wrapper.auto_label_with_vp(
            prompt_image_paths=valid_prompts,
            visuals=visuals,
            target_image_path=target_image,
            conf_thresh=0.3
        )
        
        print(f"\n--- 初始化结果 ---")
        print(f"类别数量: {wrapper.num_classes}")
        print(f"类别映射: {wrapper.class_mapping}")
        print(f"对象集合: {wrapper.initial_object_set}")
        print(f"预测结果数量: {len(predictions)}")
        
        # 验证类别一致性
        consistency_check = wrapper._validate_class_consistency()
        print(f"类别一致性检查: {'✓ 通过' if consistency_check else '✗ 失败'}")
        
        # 验证VPE张量形状
        if wrapper.final_refined_embeddings_tensor is not None:
            expected_shape = (1, wrapper.num_classes, wrapper.embedding_dim)
            actual_shape = wrapper.final_refined_embeddings_tensor.shape
            shape_correct = actual_shape == expected_shape
            print(f"VPE张量形状: {actual_shape}, 期望: {expected_shape}, {'✓ 正确' if shape_correct else '✗ 错误'}")
        else:
            print("VPE张量为None")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return wrapper

def test_incremental_updates(wrapper, data_dir):
    """测试增量更新"""
    print("\n=== 测试增量更新 ===")
    
    if wrapper is None:
        print("❌ wrapper为None，跳过增量更新测试")
        return
    
    # 获取更多图像进行增量测试
    image_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    
    # 选择几个新的图像进行增量更新测试
    test_images = image_files[5:8] if len(image_files) > 8 else image_files[-3:]
    
    for i, test_img in enumerate(test_images):
        print(f"\n--- 增量测试 {i+1}: {os.path.basename(test_img)} ---")
        
        try:
            predictions = wrapper.auto_label_with_vp(
                prompt_image_paths=[],  # 空的初始prompts，使用已有的VPE
                visuals={'bboxes': [], 'cls': []},
                target_image_path=test_img,
                conf_thresh=0.3
            )
            
            print(f"预测结果数量: {len(predictions)}")
            print(f"当前类别数量: {wrapper.num_classes}")
            print(f"当前类别映射: {wrapper.class_mapping}")
            
            # 显示预测结果
            for j, pred in enumerate(predictions):
                print(f"  预测 {j+1}: 类别ID={pred['class_id']}, 名称='{pred['class_name']}', 置信度={pred['confidence']:.3f}")
            
            # 验证类别一致性
            consistency_check = wrapper._validate_class_consistency()
            print(f"类别一致性检查: {'✓ 通过' if consistency_check else '✗ 失败'}")
            
        except Exception as e:
            print(f"❌ 增量测试失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主测试函数"""
    try:
        # 测试真实数据
        wrapper = test_with_real_data()
        
        if wrapper:
            # 测试增量更新
            data_dir = "/Users/patrick/Desktop/labeling/Bunnpris-data/set16_test"
            test_incremental_updates(wrapper, data_dir)
        
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 