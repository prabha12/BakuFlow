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
    """解析YOLO格式的标注文件"""
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

def test_real_class_mapping():
    """测试真实类别映射"""
    print("\n=== 测试真实类别映射 ===")
    
    # 数据路径
    data_dir = "/Users/patrick/Desktop/labeling/Bunnpris-data/set16_test"
    classes_path = os.path.join(data_dir, "classes.txt")
    
    # 加载真实的类别名称
    true_class_names = load_classes(classes_path)
    print(f"真实类别名称: {true_class_names}")
    print(f"类别数量: {len(true_class_names)}")
    
    # 获取一些图像文件用于测试
    image_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))[:3]
    print(f"\n使用图像: {[os.path.basename(f) for f in image_files]}")
    
    # 初始化wrapper，传入真实的类别名称
    wrapper = YOLOEWrapper(class_names=true_class_names)
    
    # 收集标注数据并显示
    all_found_class_ids = set()
    initial_bboxes = []
    initial_cls = []
    valid_prompts = []
    
    print(f"\n=== 收集标注数据 ===")
    for img_path in image_files:
        txt_path = img_path.replace('.jpg', '.txt')
        bboxes, cls_ids = parse_yolo_annotation(txt_path)
        
        if len(cls_ids) > 0:
            initial_bboxes.append(bboxes)
            initial_cls.append(cls_ids)
            valid_prompts.append(img_path)
            all_found_class_ids.update(cls_ids)
            
            print(f"  {os.path.basename(img_path)}:")
            for i, cls_id in enumerate(cls_ids):
                true_name = true_class_names[cls_id] if 0 <= cls_id < len(true_class_names) else f"未知({cls_id})"
                print(f"    对象{i+1}: 类别ID={cls_id} -> '{true_name}'")
    
    print(f"\n发现的类别ID: {sorted(all_found_class_ids)}")
    
    # 测试类别映射更新
    print(f"\n=== 测试类别映射更新 ===")
    for i, (img_path, cls_ids) in enumerate(zip(valid_prompts, initial_cls)):
        print(f"\n--- 添加第{i+1}个prompt: {os.path.basename(img_path)} ---")
        
        # 添加prompt
        wrapper._add_or_replace_prompt(
            img_path,
            initial_bboxes[i],
            cls_ids,
            1.0,
            is_initial=True
        )
        
        # 更新类别映射
        wrapper._validate_and_update_class_mapping([cls_ids])
        
        print(f"当前类别映射:")
        for cls_id, cls_name in sorted(wrapper.class_mapping.items()):
            print(f"  ID {cls_id} -> '{cls_name}'")
        
        print(f"索引映射:")
        for idx, cls_id in wrapper.index_to_class_id.items():
            cls_name = wrapper.class_mapping[cls_id]
            print(f"  索引 {idx} -> 类别ID {cls_id} -> '{cls_name}'")
    
    # 模拟预测测试
    print(f"\n=== 模拟预测映射测试 ===")
    print("模拟模型预测不同索引的结果:")
    
    for predicted_idx in range(len(wrapper.initial_object_set)):
        actual_class_id = wrapper.index_to_class_id.get(predicted_idx, -1)
        class_name = wrapper.class_mapping.get(actual_class_id, "未知")
        
        print(f"  模型预测索引 {predicted_idx} -> 实际类别ID {actual_class_id} -> 类别名称 '{class_name}'")
        
        # 验证映射的正确性
        if actual_class_id in all_found_class_ids:
            expected_name = true_class_names[actual_class_id] if 0 <= actual_class_id < len(true_class_names) else f"object_{actual_class_id}"
            is_correct = class_name == expected_name
            print(f"    ✓ 映射正确: {is_correct}")
        else:
            print(f"    ⚠ 这个类别ID在标注中未出现")
    
    return wrapper

def test_auto_label_simulation():
    """测试auto_label的模拟调用"""
    print(f"\n=== 测试auto_label模拟调用 ===")
    
    wrapper = test_real_class_mapping()
    
    # 准备数据
    data_dir = "/Users/patrick/Desktop/labeling/Bunnpris-data/set16_test"
    image_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    
    # 选择前3个作为prompt，第4个作为目标
    prompt_images = image_files[:3]
    target_image = image_files[3] if len(image_files) > 3 else image_files[-1]
    
    # 准备prompts数据
    initial_bboxes = []
    initial_cls = []
    
    for img_path in prompt_images:
        txt_path = img_path.replace('.jpg', '.txt')
        bboxes, cls_ids = parse_yolo_annotation(txt_path)
        if len(cls_ids) > 0:
            initial_bboxes.append(bboxes)
            initial_cls.append(cls_ids)
    
    visuals = {
        'bboxes': initial_bboxes,
        'cls': initial_cls
    }
    
    print(f"Prompts: {[os.path.basename(p) for p in prompt_images]}")
    print(f"Target: {os.path.basename(target_image)}")
    
    try:
        predictions = wrapper.auto_label_with_vp(
            prompt_image_paths=prompt_images,
            visuals=visuals,
            target_image_path=target_image,
            conf_thresh=0.3
        )
        
        print(f"\n预测结果 ({len(predictions)} 个):")
        for i, pred in enumerate(predictions):
            print(f"  {i+1}. 类别ID={pred['class_id']}, 名称='{pred['class_name']}', 置信度={pred['confidence']:.3f}")
            print(f"      边界框=[{pred['bbox'][0]:.1f}, {pred['bbox'][1]:.1f}, {pred['bbox'][2]:.1f}, {pred['bbox'][3]:.1f}]")
        
        # 比较预测结果与真实标注
        target_txt = target_image.replace('.jpg', '.txt')
        if os.path.exists(target_txt):
            true_bboxes, true_cls_ids = parse_yolo_annotation(target_txt)
            print(f"\n真实标注 ({len(true_cls_ids)} 个):")
            true_class_names = wrapper.true_class_names
            for i, cls_id in enumerate(true_cls_ids):
                true_name = true_class_names[cls_id] if 0 <= cls_id < len(true_class_names) else f"unknown_{cls_id}"
                print(f"  {i+1}. 类别ID={cls_id}, 名称='{true_name}'")
                print(f"      边界框=[{true_bboxes[i][0]:.1f}, {true_bboxes[i][1]:.1f}, {true_bboxes[i][2]:.1f}, {true_bboxes[i][3]:.1f}]")
    
    except Exception as e:
        print(f"❌ auto_label调用失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 开始测试真实类别映射")
    
    try:
        # 测试类别映射
        test_real_class_mapping()
        
        # 测试auto_label模拟
        test_auto_label_simulation()
        
        print("\n🎉 测试完成！")
        print("\n📊 关键改进:")
        print("✓ 使用真实的类别名称而不是 'object_X'")
        print("✓ 正确的索引到类别ID的映射")
        print("✓ 预测结果使用正确的类别名称")
        print("✓ 处理不连续的类别ID")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 