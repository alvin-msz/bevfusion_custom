#!/usr/bin/env python3
"""
将点云从LiDAR坐标系投影到相机图像平面并可视化

使用方法:
    python custom_dataset/tools/module_test/test_points_proj.py --frame-id 000000
    python custom_dataset/tools/module_test/test_points_proj.py --frame-id 000000 --output-dir output
    python custom_dataset/tools/module_test/test_points_proj.py --frame-id 000000 --max-points 50000 --color-by depth
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
from os import path as osp

# 添加项目根目录到路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from custom_dataset.tools.data_converter.custom_converter import (
    read_calib_file,
)


def load_point_cloud(lidar_path, load_dim=4):
    """
    从二进制文件加载点云
    
    Args:
        lidar_path: 点云文件路径
        load_dim: 每个点的维度（4表示x,y,z,intensity）
    
    Returns:
        points: [N, load_dim] 点云数据
    """
    if not osp.exists(lidar_path):
        print(f"Warning: Point cloud file not found: {lidar_path}")
        return None
    
    # 读取二进制文件
    points = np.fromfile(lidar_path, dtype=np.float32)
    points = points.reshape(-1, load_dim)
    
    return points


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


def depth_to_color(depth, min_depth=None, max_depth=None, colormap=cv2.COLORMAP_JET):
    """
    将深度值转换为颜色
    
    Args:
        depth: [N] 深度值数组
        min_depth: 最小深度（如果None则自动计算）
        max_depth: 最大深度（如果None则自动计算）
        colormap: OpenCV颜色映射
    
    Returns:
        colors: [N, 3] BGR颜色数组
    """
    if min_depth is None:
        min_depth = np.min(depth)
    if max_depth is None:
        max_depth = np.max(depth)
    
    # 归一化到0-255
    depth_normalized = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # 应用颜色映射
    depth_colored = cv2.applyColorMap(depth_uint8, colormap)
    
    # 转换为 [N, 3] 格式
    colors = depth_colored[:, 0, :]
    
    return colors


def intensity_to_color(intensity, min_intensity=None, max_intensity=None, colormap=cv2.COLORMAP_JET):
    """
    将强度值转换为颜色（使用jet渐变，强度低的用冷色）
    
    Args:
        intensity: [N] 强度值数组
        min_intensity: 最小强度（如果None则自动计算）
        max_intensity: 最大强度（如果None则自动计算）
        colormap: OpenCV颜色映射（默认JET，低值冷色，高值暖色）
    
    Returns:
        colors: [N, 3] BGR颜色数组
    """
    if min_intensity is None:
        min_intensity = np.min(intensity)
    if max_intensity is None:
        max_intensity = np.max(intensity)
    
    # 归一化到0-255
    intensity_normalized = (intensity - min_intensity) / (max_intensity - min_intensity + 1e-8)
    intensity_normalized = np.clip(intensity_normalized, 0, 1)
    intensity_uint8 = (intensity_normalized * 255).astype(np.uint8)
    
    # 应用颜色映射（jet：低值=冷色/蓝色，高值=暖色/红色）
    intensity_colored = cv2.applyColorMap(intensity_uint8, colormap)
    
    # 转换为 [N, 3] 格式
    colors = intensity_colored[:, 0, :]
    
    return colors


def visualize_point_cloud_projection(frame_id, data_root, output_dir=None, 
                                     max_points=50000, color_by='depth', 
                                     point_size=1, load_dim=4):
    """
    可视化指定帧的点云投影
    
    Args:
        frame_id: 帧ID（如 '000000'）
        data_root: 数据根目录
        output_dir: 输出目录（可选）
        max_points: 最大投影点数（用于性能优化）
        color_by: 颜色编码方式 ('depth', 'intensity', 'fixed')
        point_size: 点的大小（像素）
        load_dim: 点云维度（4表示x,y,z,intensity）
    """
    # 文件路径
    calib_path = osp.join(data_root, 'calib', f'{frame_id}.txt')
    points_path = osp.join(data_root, 'points', f'{frame_id}.bin')
    
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
    
    # Tr_velo_to_cam: LiDAR到相机的变换
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
    
    # R0_rect: 校正矩阵
    R0_rect = None
    if 'R0_rect' in calib_data:
        R0_rect = calib_data['R0_rect'].reshape(3, 3)
    
    # 读取点云文件
    points = load_point_cloud(points_path, load_dim=load_dim)
    if points is None:
        print(f"Error: Failed to load point cloud from {points_path}")
        return
    
    print(f"Loaded {len(points)} points from frame {frame_id}")
    
    # 如果点太多，随机采样
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        print(f"Sampled {max_points} points for visualization")
    
    # 转换到相机坐标系
    points_cam = lidar_to_camera(points, Tr_velo_to_cam, R0_rect)
    
    # 过滤掉在相机后面的点（z < 0）
    valid_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_mask]
    points = points[valid_mask]
    
    if len(points_cam) == 0:
        print("Warning: No valid points after filtering (all points are behind camera)")
        return
    
    print(f"Valid points (in front of camera): {len(points_cam)}")
    
    # 投影到图像平面
    points_2d = project_to_image(points_cam, P2)
    
    # 读取图像
    image = np.array(Image.open(image_path))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 3:
        # 假设是RGB，转换为BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    h, w = image.shape[:2]
    
    # 过滤在图像范围内的点
    in_image_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    points_2d = points_2d[in_image_mask]
    points_cam = points_cam[in_image_mask]
    points = points[in_image_mask]
    
    print(f"Points within image bounds: {len(points_2d)}")
    
    if len(points_2d) == 0:
        print("Warning: No points within image bounds")
        return
    
    # 根据选择的颜色编码方式设置颜色
    if color_by == 'depth':
        # 使用深度（z坐标）进行颜色编码
        depth = points_cam[:, 2]
        colors = depth_to_color(depth, colormap=cv2.COLORMAP_JET)
    elif color_by == 'intensity' and points.shape[1] >= 4:
        # 使用强度进行颜色编码
        intensity = points[:, 3]
        colors = intensity_to_color(intensity)
    else:
        # 固定颜色（绿色）
        colors = np.array([[0, 255, 0]] * len(points_2d), dtype=np.uint8)
    
    # 在图像上绘制点
    for i, (pt, color) in enumerate(zip(points_2d, colors)):
        pt_int = pt.astype(int)
        if point_size == 1:
            image[pt_int[1], pt_int[0]] = color
        else:
            cv2.circle(image, tuple(pt_int), point_size, tuple(color.tolist()), -1)
    
    print(f"Successfully projected {len(points_2d)} points")
    
    # 保存或显示结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = osp.join(output_dir, f'{frame_id}_points_projection.jpg')
        cv2.imwrite(output_path, image)
        print(f"Saved visualization to: {output_path}")
    else:
        # 显示图像
        cv2.imshow(f'Point Cloud Projection - Frame {frame_id}', image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Project point cloud to camera image')
    parser.add_argument('--frame-id', type=str, default='000000',
                       help='Frame ID (e.g., 000000)')
    parser.add_argument('--data-root', type=str, 
                       default='data/custom_mini',
                       help='Root directory of the dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for saving visualization (if None, display)')
    parser.add_argument('--max-points', type=int, default=50000,
                       help='Maximum number of points to project (for performance)')
    parser.add_argument('--color-by', type=str, default='depth',
                       choices=['depth', 'intensity', 'fixed'],
                       help='Color encoding method: depth, intensity, or fixed color')
    parser.add_argument('--point-size', type=int, default=5,
                       help='Size of points in pixels')
    parser.add_argument('--load-dim', type=int, default=4,
                       help='Dimension of point cloud (4 for x,y,z,intensity)')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    data_root = osp.abspath(args.data_root)
    
    if not osp.exists(data_root):
        print(f"Error: Data root directory not found: {data_root}")
        return
    
    visualize_point_cloud_projection(
        args.frame_id, 
        data_root, 
        args.output_dir,
        max_points=args.max_points,
        color_by=args.color_by,
        point_size=args.point_size,
        load_dim=args.load_dim
    )


if __name__ == '__main__':
    main()

