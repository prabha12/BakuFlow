#!/usr/bin/env python3
import os
import argparse
from PyQt5.QtWidgets import QApplication
from labelimg.controller.label_controller import LabelController
from labelimg.gui.main_window import LabelingTool as MainWindow

def main():
    parser = argparse.ArgumentParser(description='批量自動標註工具')
    parser.add_argument('--prompt-dir', required=True, help='包含提示圖像的目錄')
    parser.add_argument('--target-dir', required=True, help='需要標註的目標圖像目錄')
    parser.add_argument('--class-file', required=True, help='類別定義文件路徑')
    args = parser.parse_args()

    # 初始化 Qt 應用
    app = QApplication([])
    
    # 創建主窗口
    main_window = MainWindow()
    
    # 加載類別定義
    if os.path.exists(args.class_file):
        # main_window.load_classes(args.class_file)
        main_window.loadClasses(args.class_file)
    else:
        print(f"錯誤: 類別文件不存在: {args.class_file}")
        return
        
    print("Loading classes done")
    
    # 獲取提示圖像
    prompt_images = []
    for filename in os.listdir(args.prompt_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(args.prompt_dir, filename)
            # 檢查是否有對應的標註文件
            base_name = os.path.splitext(filename)[0]
            for ext in ['.xml', '.txt', '.json']:
                annot_path = os.path.join(args.prompt_dir, base_name + ext)
                if os.path.exists(annot_path):
                    prompt_images.append(image_path)
                    break
                    
    if not prompt_images:
        print("錯誤: 未找到帶有標註的提示圖像")
        return
        
    # 創建控制器
    controller = LabelController(main_window)
    
    print("Label controller created and going to handle")
    # 執行批量自動標註
    try:
        controller.handle_batch_auto_label_with_vp(prompt_images, args.target_dir)
    except Exception as e:
        print(f"自動標註過程中出錯: {e}")
        
if __name__ == '__main__':
    main() 