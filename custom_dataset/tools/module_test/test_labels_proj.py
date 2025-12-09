#!/usr/bin/env python3
"""
将3D bbox从LiDAR坐标系投影到相机图像平面并可视化

使用方法:
    python custom_dataset/tools/module_test/test_label_proj.py --frame-id 000000
    python custom_dataset/tools/module_test/test_label_proj.py --frame-id 000000 --output-dir output
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
from os import path as osp
from mmdet3d.core.bbox import LiDARInstance3DBoxes

# 添加项目根目录到路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from custom_dataset.tools.data_converter.custom_converter import (
    read_calib_file,
    parse_calib_to_camera_params,
    class_id
)


def get_3d_box_corners(center, size, yaw):
    """
    计算3D bbox的8个角点（在LiDAR坐标系中）
    
    Args:
        center: [x, y, z] 底部中心点
        size: [w, l, h] 宽度、长度、高度
        yaw: 绕z轴的旋转角度（弧度）
    
    Returns:
        corners: [8, 3] 8个角点的坐标
    """
    w, l, h = size
    x, y, z = center
    
    # 创建相对于中心点的8个角点（在bbox的局部坐标系中）
    # 使用origin=(0.5, 0.5, 0)，即底部中心
    corners = np.array([
        [-l/2, -w/2, 0],      # 0: 左下后
        [-l/2, -w/2, h],      # 1: 左下后上
        [-l/2, w/2, 0],       # 2: 右下后
        [-l/2, w/2, h],       # 3: 右下后上
        [l/2, -w/2, 0],       # 4: 左下前
        [l/2, -w/2, h],        # 5: 左下前上
        [l/2, w/2, 0],        # 6: 右下前
        [l/2, w/2, h],        # 7: 右下前上
    ], dtype=np.float32)
    
    # 绕z轴旋转
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 应用旋转
    corners = corners @ rotation_matrix.T
    
    # 平移到中心点
    corners = corners + np.array([x, y, z])
    
    return corners


def boxes_to_lidar_boxes(boxes, class_id_map):
    """
    将解析的boxes转换为LiDARInstance3DBoxes格式，统一尺寸/旋转定义
    """
    if len(boxes) == 0:
        return None, np.array([], dtype=np.int64), []

    bbox_array = []
    labels = []
    classes = []

    for box in boxes:
        center = box['center']  # [x, y, z] 底部中心
        size = box['size']      # [w, l, h]
        yaw = box['yaw']        # 弧度
        class_name = box['class']

        std_class_name = class_id_map.get(class_name, class_name)

        bbox = np.array([
            center[0],  # x
            center[1],  # y
            center[2],  # z (底部中心)
            size[0],    # w (x_size)
            size[1],    # l (y_size)
            size[2],    # h (z_size)
            yaw
        ], dtype=np.float32)

        bbox_array.append(bbox)

        if std_class_name not in classes:
            classes.append(std_class_name)
        labels.append(classes.index(std_class_name))

    bbox_array = np.array(bbox_array, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    bboxes_3d = LiDARInstance3DBoxes(bbox_array, box_dim=7)
    return bboxes_3d, labels, classes


def lidar_to_camera(points, Tr_velo_to_cam, R0_rect=None):
    """
    将点从LiDAR坐标系转换到相机坐标系
    根据KITTI格式：先Tr_velo_to_cam，然后R0_rect
    
    Args:
        points: [N, 3] LiDAR坐标系中的点（只使用x,y,z）
        Tr_velo_to_cam: [3, 4] 变换矩阵
        R0_rect: [3, 3] 校正矩阵（可选）
    
    Returns:
        points_cam: [N, 3] 相机坐标系中的点（校正后）
    """
    # 只使用x,y,z坐标
    if points.shape[1] > 3:
        points_3d = points[:, :3]
    else:
        points_3d = points
    
    # 扩展为齐次坐标
    points_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
    
    # Tr_velo_to_cam是3x4，需要扩展为4x4
    Tr_velo_to_cam_4x4 = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam_4x4[:3, :] = Tr_velo_to_cam
    
    # 第一步：通过Tr_velo_to_cam转换到相机坐标系（未校正）
    points_cam = points_homo @ Tr_velo_to_cam_4x4.T
    points_cam_3d = points_cam[:, :3]
    
    # 第二步：应用R0_rect校正（如果提供）
    if R0_rect is not None:
        # R0_rect需要扩展为4x4（齐次坐标）
        R0_rect_4x4 = np.eye(4, dtype=np.float32)
        R0_rect_4x4[:3, :3] = R0_rect
        # 应用校正
        points_cam_homo = np.concatenate([points_cam_3d, np.ones((points_cam_3d.shape[0], 1))], axis=1)
        points_cam_corrected = points_cam_homo @ R0_rect_4x4.T
        points_cam_3d = points_cam_corrected[:, :3]
    
    return points_cam_3d


def project_to_image(points_cam, P2):
    """
    将相机坐标系中的点投影到图像平面
    
    Args:
        points_cam: [N, 3] 相机坐标系中的点
        P2: [3, 4] 投影矩阵（相机内参）
    
    Returns:
        points_img: [N, 2] 图像坐标 (u, v)
    """
    # 扩展为齐次坐标
    points_homo = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)
    
    # 投影
    points_2d = points_homo @ P2.T
    
    # 归一化
    points_img = points_2d[:, :2] / (points_2d[:, 2:3] + 1e-8)
    
    return points_img


def draw_3d_box_on_image(image, corners_2d, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制3D bbox的投影
    
    Args:
        image: 图像数组
        corners_2d: [8, 2] 8个角点的2D坐标
        color: 颜色 (B, G, R)
        thickness: 线宽
    """
    # 定义12条边的连接关系（立方体的12条边）
    # 角点顺序遵循mmdet3d的corners输出：前4个为底面，后4个为顶面（顺时针/逆时针闭合）
    edges = [
        # 底面
        [0, 1], [1, 2], [2, 3], [3, 0],
        # 顶面
        [4, 5], [5, 6], [6, 7], [7, 4],
        # 立柱
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    
    # 绘制边
    for edge in edges:
        pt1 = tuple(corners_2d[edge[0]].astype(int))
        pt2 = tuple(corners_2d[edge[1]].astype(int))
        cv2.line(image, pt1, pt2, color, thickness)
    
    return image


def parse_label_file(label_path):
    """
    解析标签文件，提取bbox信息
    
    Args:
        label_path: 标签文件路径
    
    Returns:
        boxes: list of dict, 每个dict包含 'class', 'center', 'size', 'yaw'
    """
    boxes = []
    
    if not osp.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 8:
                continue
            
            # 解析类别名称
            class_name = None
            numeric_fields = None
            
            first_field = parts[0]
            if first_field in class_id:
                class_name = first_field
                numeric_fields = parts[1:8]
            elif first_field.lower() in class_id:
                class_name = first_field.lower()
                numeric_fields = parts[1:8]
            else:
                continue
            
            try:
                # 解析数值字段: x y z w l h angle
                values = [float(x) for x in numeric_fields]
                if len(values) != 7:
                    continue
                
                center = np.array(values[:3], dtype=np.float32)  # x, y, z
                size = np.array(values[3:6], dtype=np.float32)   # w, l, h
                yaw = values[6]  # angle
                
                boxes.append({
                    'class': class_name,
                    'center': center,
                    'size': size,
                    'yaw': yaw
                })
            except (ValueError, IndexError):
                continue
    
    return boxes


def visualize_bbox_projection(frame_id, data_root, output_dir=None):
    """
    可视化指定帧的3D bbox投影
    
    Args:
        frame_id: 帧ID（如 '000000'）
        data_root: 数据根目录
        output_dir: 输出目录（可选）
    """
    # 文件路径
    calib_path = osp.join(data_root, 'calib', f'{frame_id}.txt')
    label_path = osp.join(data_root, 'labels', f'{frame_id}.txt')
    
    # 尝试不同的图像扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_path = None
    for ext in image_extensions:
        test_path = osp.join(data_root, 'camera', 'cam_front', f'{frame_id}{ext}')
        if osp.exists(test_path):
            image_path = test_path
            break
    
    if image_path is None:
        print(f"Error: Image file not found for frame {frame_id}")
        return
    
    # 读取标定文件
    if not osp.exists(calib_path):
        print(f"Error: Calib file not found: {calib_path}")
        return
    
    calib_data = read_calib_file(calib_path)
    
    # 提取标定参数
    P2 = calib_data['P2'].reshape(3, 4)  # 投影矩阵
    cam_intrinsic = P2[:, :3]  # 内参矩阵
    
    # Tr_velo_to_cam: LiDAR到相机的变换
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
    
    # R0_rect: 校正矩阵
    R0_rect = None
    if 'R0_rect' in calib_data:
        R0_rect = calib_data['R0_rect'].reshape(3, 3)
    
    # 读取标签文件
    boxes = parse_label_file(label_path)
    print(f"Found {len(boxes)} boxes in frame {frame_id}")
    bboxes_3d, labels, classes = boxes_to_lidar_boxes(boxes, class_id)
    
    # 读取图像
    image = np.array(Image.open(image_path))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 3:
        # 假设是RGB，转换为BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 定义不同类别的颜色
    class_colors = {
        'car': (0, 255, 0),           # 绿色
        'truck': (255, 0, 0),         # 蓝色
        'bus': (0, 0, 255),           # 红色
        'trailer': (255, 255, 0),     # 青色
        'construction_vehicle': (255, 0, 255),  # 洋红色
        'bicycle': (0, 255, 255),     # 黄色
        'motorcycle': (128, 0, 128),  # 紫色
        'pedestrian': (255, 165, 0),  # 橙色
        'traffic_cone': (0, 128, 255), # 橙色
        'barrier': (128, 128, 128),   # 灰色
    }
    
    # 投影并绘制每个bbox
    valid_boxes = 0
    if bboxes_3d is None:
        print("No boxes to project")
        return

    for i in range(len(bboxes_3d)):
        class_idx = labels[i]
        class_name = classes[class_idx] if class_idx < len(classes) else 'unknown'

        # 使用mmdet3d的corners，确保与点云可视化保持一致
        corners_3d_lidar = bboxes_3d.corners[i].cpu().numpy().astype(np.float32)

        # 转换到相机坐标系
        corners_3d_cam = lidar_to_camera(corners_3d_lidar, Tr_velo_to_cam, R0_rect)
        
        # 过滤掉在相机后面的点（z < 0）
        valid_mask = corners_3d_cam[:, 2] > 0
        if not np.any(valid_mask):
            continue
        
        # 投影到图像平面
        corners_2d = project_to_image(corners_3d_cam, P2)
        
        # 检查是否在图像范围内
        h, w = image.shape[:2]
        in_image_mask = (corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < w) & \
                        (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < h)
        
        if not np.any(in_image_mask):
            continue
        
        # 选择颜色
        color = class_colors.get(class_name, (255, 255, 255))
        
        # 绘制3D框
        image = draw_3d_box_on_image(image, corners_2d, color=color, thickness=2)
        
        # 在底部中心点绘制类别标签
        center_lidar = bboxes_3d.tensor[i, :3].cpu().numpy()  # z为底部中心
        center_cam = lidar_to_camera(center_lidar[None, :], Tr_velo_to_cam, R0_rect)
        center_2d = project_to_image(center_cam, P2)[0]
        if 0 <= center_2d[0] < w and 0 <= center_2d[1] < h:
            cv2.putText(image, class_name, 
                       (int(center_2d[0]), int(center_2d[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        valid_boxes += 1
    
    print(f"Successfully projected {valid_boxes} boxes")
    
    # 保存或显示结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = osp.join(output_dir, f'{frame_id}_projection.jpg')
        cv2.imwrite(output_path, image)
        print(f"Saved visualization to: {output_path}")
    else:
        # 显示图像
        cv2.imshow(f'3D BBox Projection - Frame {frame_id}', image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Project 3D bboxes to camera image')
    parser.add_argument('--frame-id', type=str, default='000000',
                       help='Frame ID (e.g., 000000)')
    parser.add_argument('--data-root', type=str, 
                       default='data/custom_mini',
                       help='Root directory of the dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for saving visualization (if None, display)')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    data_root = osp.abspath(args.data_root)
    
    if not osp.exists(data_root):
        print(f"Error: Data root directory not found: {data_root}")
        return
    
    visualize_bbox_projection(args.frame_id, data_root, args.output_dir)


if __name__ == '__main__':
    main()

