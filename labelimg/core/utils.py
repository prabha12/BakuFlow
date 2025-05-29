import cv2

def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_length=10):
    """绘制虚线矩形"""
    x1, y1 = pt1
    x2, y2 = pt2
    # 顶部水平线
    for x in range(x1, x2, dash_length * 2):
        end_x = min(x + dash_length, x2)
        cv2.line(img, (x, y1), (end_x, y1), color, thickness)

    # 底部水平线
    for x in range(x1, x2, dash_length * 2):
        end_x = min(x + dash_length, x2)
        cv2.line(img, (x, y2), (end_x, y2), color, thickness)

    # 左侧垂直线
    for y in range(y1, y2, dash_length * 2):
        end_y = min(y + dash_length, y2)
        cv2.line(img, (x1, y), (x1, end_y), color, thickness)

    # 右侧垂直线
    for y in range(y1, y2, dash_length * 2):
        end_y = min(y + dash_length, y2)
        cv2.line(img, (x2, y), (x2, end_y), color, thickness) 