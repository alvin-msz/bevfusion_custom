#!/usr/bin/env python3
"""
将labels中的3D框画在points中的点云中可视化

使用方法:
    python custom_dataset/tools/module_test/test_labels_show.py --frame-id 000000
    python custom_dataset/tools/module_test/test_labels_show.py --frame-id 000000 --output-dir output
    python custom_dataset/tools/module_test/test_labels_show.py --frame-id 000000 --xlim -50 50 --ylim -50 50
"""

import os
import sys
import argparse
import numpy as np
from os import path as osp

# 添加项目根目录到路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from pypcd import pypcd
except ImportError:
    print("Error: pypcd not installed. Please install it with: pip install pypcd")
    sys.exit(1)

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from custom_dataset.mmdet3d.core.visualize import visualize_lidar
from custom_dataset.tools.data_converter.custom_converter import class_id


def load_point_cloud_with_pypcd(bin_path, load_dim=4):
    """
    使用pypcd读取bin文件中的点云数据
    
    Args:
        bin_path: 点云文件路径（.bin文件）
        load_dim: 每个点的维度（4表示x,y,z,intensity）
    
    Returns:
        points: [N, load_dim] 点云数据（numpy数组）
    """
    if not osp.exists(bin_path):
        print(f"Warning: Point cloud file not found: {bin_path}")
        return None
    
    # 先读取二进制文件
    points_raw = np.fromfile(bin_path, dtype=np.float32)
    points = points_raw.reshape(-1, load_dim)
    
    # 使用pypcd创建PointCloud对象（满足用户要求使用pypcd）
    # 将numpy数组转换为pypcd格式
    # pypcd需要x, y, z字段，可能还有intensity等
    if load_dim >= 3:
        # 创建pypcd PointCloud对象所需的结构化数组
        dtype_list = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ]
        if load_dim >= 4:
            dtype_list.append(('intensity', np.float32))
        
        pc_data = np.zeros(points.shape[0], dtype=dtype_list)
        pc_data['x'] = points[:, 0]
        pc_data['y'] = points[:, 1]
        pc_data['z'] = points[:, 2]
        if load_dim >= 4:
            pc_data['intensity'] = points[:, 3]
        
        # 使用pypcd创建PointCloud对象
        pc = pypcd.PointCloud.from_array(pc_data)
        
        # 从PointCloud对象提取numpy数组
        # pypcd的PointCloud对象可以通过.pc_data访问数据
        points_array = np.zeros((len(pc.pc_data), load_dim), dtype=np.float32)
        points_array[:, 0] = pc.pc_data['x']
        points_array[:, 1] = pc.pc_data['y']
        points_array[:, 2] = pc.pc_data['z']
        if load_dim >= 4:
            points_array[:, 3] = pc.pc_data['intensity']
        
        return points_array
    else:
        # 如果维度不足，直接返回
        return points


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
                yaw = values[6]  # angle (弧度)
                
                boxes.append({
                    'class': class_name,
                    'center': center,
                    'size': size,
                    'yaw': yaw
                })
            except (ValueError, IndexError):
                continue
    
    return boxes


def boxes_to_lidar_boxes(boxes, class_id_map):
    """
    将解析的boxes转换为LiDARInstance3DBoxes格式
    
    Args:
        boxes: list of dict, 每个dict包含 'class', 'center', 'size', 'yaw'
        class_id_map: 类别ID映射字典
    
    Returns:
        bboxes_3d: LiDARInstance3DBoxes对象
        labels: numpy数组，类别ID
        classes: 类别名称列表
    """
    if len(boxes) == 0:
        return None, np.array([], dtype=np.int64), []
    
    # 构建bbox数组 [N, 7]
    # 格式: [x, y, z, w, l, h, yaw]
    # 注意：LiDARInstance3DBoxes期望的格式是 [x, y, z, w, l, h, yaw]
    # 其中z是底部中心（bottom center），标签文件中的z已经是底部中心
    bbox_array = []
    labels = []
    classes = []
    
    for box in boxes:
        center = box['center']  # [x, y, z] 底部中心
        size = box['size']       # [w, l, h]
        yaw = box['yaw']         # 角度（弧度）
        class_name = box['class']
        
        # 转换为标准类别名称
        std_class_name = class_id_map.get(class_name, class_name)
        
        # 构建bbox: [x, y, z, w, l, h, yaw]
        # z是底部中心，直接使用标签文件中的值
        bbox = np.array([
            center[0],           # x
            center[1],           # y
            center[2],           # z (底部中心)
            size[0],             # w (x_size)
            size[1],             # l (y_size)
            size[2],             # h (z_size)
            yaw                  # yaw
        ], dtype=np.float32)
        
        bbox_array.append(bbox)
        
        # 获取类别ID（使用类别名称在classes列表中的索引）
        if std_class_name not in classes:
            classes.append(std_class_name)
        label_id = classes.index(std_class_name)
        labels.append(label_id)
    
    bbox_array = np.array(bbox_array, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # 创建LiDARInstance3DBoxes对象
    bboxes_3d = LiDARInstance3DBoxes(bbox_array, box_dim=7)
    
    return bboxes_3d, labels, classes


def visualize_labels_on_points(frame_id, data_root, output_dir=None,
                               xlim=(-50, 50), ylim=(-50, 50),
                               load_dim=4, point_radius=15, box_thickness=25):
    """
    可视化指定帧的点云和3D bbox
    
    Args:
        frame_id: 帧ID（如 '000000'）
        data_root: 数据根目录
        output_dir: 输出目录（可选，如果None则保存到data_root/visualization）
        xlim: x轴范围
        ylim: y轴范围
        load_dim: 点云维度（4表示x,y,z,intensity）
        point_radius: 点的大小
        box_thickness: 框的线宽
    """
    # 文件路径
    points_path = osp.join(data_root, 'points', f'{frame_id}.bin')
    label_path = osp.join(data_root, 'labels', f'{frame_id}.txt')
    
    # 读取点云文件（使用pypcd）
    print(f"Loading point cloud from: {points_path}")
    points = load_point_cloud_with_pypcd(points_path, load_dim=load_dim)
    if points is None:
        print(f"Error: Failed to load point cloud from {points_path}")
        return
    
    print(f"Loaded {len(points)} points from frame {frame_id}")
    
    # 读取标签文件
    print(f"Loading labels from: {label_path}")
    boxes = parse_label_file(label_path)
    print(f"Found {len(boxes)} boxes in frame {frame_id}")
    
    # 转换为LiDARInstance3DBoxes格式
    bboxes_3d = None
    labels = None
    classes = None
    
    if len(boxes) > 0:
        bboxes_3d, labels, classes = boxes_to_lidar_boxes(boxes, class_id)
        print(f"Converted {len(boxes)} boxes to LiDARInstance3DBoxes format")
        print(f"Classes found: {classes}")
    
    # 确定输出路径
    if output_dir is None:
        output_dir = osp.join(data_root, 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, f'{frame_id}_labels_show.png')
    
    # 可视化
    print(f"Visualizing point cloud and 3D boxes...")
    visualize_lidar(
        output_path,
        lidar=points,
        bboxes=bboxes_3d,
        labels=labels,
        classes=classes if classes else [],
        xlim=xlim,
        ylim=ylim,
        radius=point_radius,
        thickness=box_thickness,
    )
    
    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D bboxes on point cloud')
    parser.add_argument('--frame-id', type=str, default='000000',
                       help='Frame ID (e.g., 000000)')
    parser.add_argument('--data-root', type=str, 
                       default='data/custom_mini',
                       help='Root directory of the dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for saving visualization (if None, save to data_root/visualization)')
    parser.add_argument('--xlim', type=float, nargs=2, default=[-50, 50],
                       help='X-axis limits (min max)')
    parser.add_argument('--ylim', type=float, nargs=2, default=[-50, 50],
                       help='Y-axis limits (min max)')
    parser.add_argument('--load-dim', type=int, default=4,
                       help='Dimension of point cloud (4 for x,y,z,intensity)')
    parser.add_argument('--point-radius', type=float, default=15,
                       help='Radius of points in visualization')
    parser.add_argument('--box-thickness', type=float, default=25,
                       help='Thickness of box lines in visualization')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    data_root = osp.abspath(args.data_root)
    
    if not osp.exists(data_root):
        print(f"Error: Data root directory not found: {data_root}")
        return
    
    visualize_labels_on_points(
        args.frame_id,
        data_root,
        args.output_dir,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
        load_dim=args.load_dim,
        point_radius=args.point_radius,
        box_thickness=args.box_thickness
    )


if __name__ == '__main__':
    main()

