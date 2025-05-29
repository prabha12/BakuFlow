#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from yoloe_vp import YOLOEWrapper
    print("✓ 成功导入 YOLOEWrapper")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def test_class_management():
    """测试类别管理的稳定性"""
    print("\n=== 测试类别管理稳定性 ===")
    
    # 初始化wrapper
    wrapper = YOLOEWrapper()
    print(f"初始状态 - 类别数量: {wrapper.num_classes}, 类别映射: {wrapper.class_mapping}")
    
    # 模拟第一次标注：类别0和2
    print("\n--- 第一次标注 (类别 0, 2) ---")
    prompt_images = ["image1.jpg"]
    visuals = {
        'bboxes': [np.array([[10, 10, 50, 50], [60, 60, 100, 100]])],
        'cls': [np.array([0, 2])]
    }
    
    # 手动添加初始prompt
    wrapper._add_or_replace_prompt(
        "image1.jpg",
        visuals['bboxes'][0],
        visuals['cls'][0],
        1.0,
        is_initial=True
    )
    
    # 更新VPE
    success = wrapper._update_vpe_with_new_annotations()
    print(f"VPE更新成功: {success}")
    print(f"类别数量: {wrapper.num_classes}")
    print(f"类别映射: {wrapper.class_mapping}")
    print(f"对象集合: {wrapper.initial_object_set}")
    
    # 模拟第二次标注：添加类别1和3
    print("\n--- 第二次标注 (类别 1, 3) ---")
    wrapper._add_or_replace_prompt(
        "image2.jpg",
        np.array([[20, 20, 60, 60], [80, 80, 120, 120]]),
        np.array([1, 3]),
        0.9,
        is_initial=False
    )
    
    success = wrapper._update_vpe_with_new_annotations()
    print(f"VPE更新成功: {success}")
    print(f"类别数量: {wrapper.num_classes}")
    print(f"类别映射: {wrapper.class_mapping}")
    print(f"对象集合: {wrapper.initial_object_set}")
    
    # 模拟第三次标注：重复使用已有类别
    print("\n--- 第三次标注 (重复类别 0, 1) ---")
    wrapper._add_or_replace_prompt(
        "image3.jpg",
        np.array([[30, 30, 70, 70], [90, 90, 130, 130]]),
        np.array([0, 1]),
        0.95,
        is_initial=False
    )
    
    success = wrapper._update_vpe_with_new_annotations()
    print(f"VPE更新成功: {success}")
    print(f"类别数量: {wrapper.num_classes}")
    print(f"类别映射: {wrapper.class_mapping}")
    print(f"对象集合: {wrapper.initial_object_set}")
    
    # 验证类别一致性
    print("\n--- 类别一致性验证 ---")
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
    
    return wrapper

def test_prediction_mapping():
    """测试预测时的类别映射"""
    print("\n=== 测试预测类别映射 ===")
    
    wrapper = test_class_management()
    
    # 模拟auto_label_with_vp的调用（简化版本）
    print(f"\n当前类别映射: {wrapper.class_mapping}")
    print(f"当前对象集合: {wrapper.initial_object_set}")
    
    # 模拟预测结果的类别映射
    for predicted_index in range(len(wrapper.initial_object_set)):
        class_name = wrapper.initial_object_set[predicted_index]
        actual_class_id = wrapper.reverse_class_mapping.get(class_name, predicted_index)
        print(f"预测索引 {predicted_index} -> 类别名称 '{class_name}' -> 实际类别ID {actual_class_id}")

if __name__ == "__main__":
    try:
        test_class_management()
        test_prediction_mapping()
        print("\n🎉 所有测试完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 