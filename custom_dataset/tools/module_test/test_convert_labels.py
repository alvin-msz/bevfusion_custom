import os
import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入mmdet3d的标准转换函数
from mmdet3d.core.bbox.box_np_ops import box_camera_to_lidar

def parse_kitti_label(label_path, calib_path=None):
    """
    解析KITTI格式的标签文件，转换为LiDAR坐标系下的3D Bbox
    
    Args:
        label_path: 标签文件路径
        calib_path: 标定文件路径（可选，如果提供则使用变换矩阵转换）
        
    Returns:
        bboxes: List of [type, x, y, z, w, l, h, angle]
    """
    bboxes = []
    
    # 如果提供了标定文件，读取变换矩阵
    Tr_velo_to_cam = None
    R0_rect = None
    if calib_path and os.path.exists(calib_path):
        from custom_dataset.tools.data_converter.custom_converter import read_calib_file
        calib_data = read_calib_file(calib_path)
        Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
        if 'R0_rect' in calib_data:
            R0_rect = calib_data['R0_rect'].reshape(3, 3)
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 16:
            continue
        
        # 解析字段
        obj_type = parts[0]
        
        # 3D尺寸 (camera坐标系: height, width, length)
        h_cam = float(parts[8])   # height
        w_cam = float(parts[9])   # width
        l_cam = float(parts[10])  # length
        
        # 3D位置 (camera坐标系: x, y, z)
        x_cam = float(parts[11])  # 横向
        y_cam = float(parts[12])  # 纵向(高度)
        z_cam = float(parts[13])  # 深度(距离)
        
        # 旋转角 (绕camera的Y轴)
        ry_cam = float(parts[14])
        
        # ===== 坐标系转换: Camera -> LiDAR =====
        if Tr_velo_to_cam is not None and R0_rect is not None:
            # 使用mmdet3d的标准转换函数
            # 输入格式：[x, y, z, l, h, w, r] (相机坐标系)
            # 输出格式：[x, y, z, w, l, h, r] (LiDAR坐标系)
            
            # 构建变换矩阵
            velo2cam_4x4 = np.eye(4, dtype=np.float32)
            velo2cam_4x4[:3, :] = Tr_velo_to_cam
            
            r_rect_4x4 = np.eye(4, dtype=np.float32)
            r_rect_4x4[:3, :3] = R0_rect
            
            # 构建相机坐标系下的bbox：[x, y, z, l, h, w, r]
            # 注意：mmdet3d期望的格式是 [l, h, w]，不是 [h, w, l]
            box_cam = np.array([[x_cam, y_cam, z_cam, l_cam, h_cam, w_cam, ry_cam]], dtype=np.float32)
            
            # 使用标准转换函数
            box_lidar = box_camera_to_lidar(box_cam, r_rect_4x4, velo2cam_4x4)
            
            # 提取结果：[x, y, z, w, l, h, r]
            x_lidar = box_lidar[0, 0]
            y_lidar = box_lidar[0, 1]
            z_lidar = box_lidar[0, 2]
            w_lidar = box_lidar[0, 3]
            l_lidar = box_lidar[0, 4]
            h_lidar = box_lidar[0, 5]
            angle_lidar = box_lidar[0, 6]
        else:
            # 如果没有标定文件，使用简单的坐标轴映射（可能不够精确）
            # Camera坐标系: x右, y下, z前
            # LiDAR坐标系: x前, y左, z上
            
            # 位置转换
            x_lidar = z_cam          # camera的z(前) -> lidar的x(前)
            y_lidar = -x_cam         # camera的x(右) -> lidar的-y(左)
            z_lidar = -y_cam         # camera的y(下，底部中心) -> lidar的z(上，底部中心)
            
            # 尺寸转换：根据mmdet3d标准，[l, h, w] -> [w, l, h]
            w_lidar = l_cam          # 相机的length -> LiDAR的width
            l_lidar = h_cam          # 相机的height -> LiDAR的length
            h_lidar = w_cam          # 相机的width -> LiDAR的height
            
            # 角度转换（简化处理）
            angle_lidar = -ry_cam - np.pi / 2.0
        
        # 归一化角度到 [-pi, pi]
        angle_lidar = np.arctan2(np.sin(angle_lidar), np.cos(angle_lidar))
        
        bbox = [
            obj_type,
            x_lidar,
            y_lidar,
            z_lidar,
            w_lidar,
            l_lidar,
            h_lidar,
            angle_lidar
        ]
        
        bboxes.append(bbox)
    
    return bboxes


def process_label_directory(label_dir, output_dir=None, calib_dir=None):
    """
    批量处理标签目录
    
    Args:
        label_dir: 输入标签目录
        output_dir: 输出目录(可选)，如果不指定则只返回结果
        calib_dir: 标定文件目录(可选)，如果提供则使用变换矩阵转换
        
    Returns:
        results: Dict {filename: bboxes}
    """
    label_dir = Path(label_dir)
    results = {}
    
    # 获取所有txt文件
    label_files = sorted(label_dir.glob('*.txt'))
    
    print(f"找到 {len(label_files)} 个标签文件")
    
    for label_file in label_files:
        filename = label_file.stem
        
        # 尝试读取对应的标定文件
        calib_path = None
        if calib_dir:
            calib_file = Path(calib_dir) / f"{filename}.txt"
            if calib_file.exists():
                calib_path = str(calib_file)
        
        bboxes = parse_kitti_label(str(label_file), calib_path=calib_path)
        results[filename] = bboxes
        
        print(f"\n处理: {label_file.name}")
        print(f"检测到 {len(bboxes)} 个物体:")
        
        for bbox in bboxes:
            obj_type, x, y, z, w, l, h, angle = bbox
            print(f"  {obj_type}: center=({x:.2f}, {y:.2f}, {z:.2f}), "
                  f"size=(w:{w:.2f}, l:{l:.2f}, h:{h:.2f}), angle={angle:.2f}rad")
    
    # 如果指定了输出目录，保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, bboxes in results.items():
            output_file = output_dir / f"{filename}.txt"
            with open(output_file, 'w') as f:
                for bbox in bboxes:
                    obj_type, x, y, z, w, l, h, angle = bbox
                    f.write(f"{obj_type} {x:.6f} {y:.6f} {z:.6f} "
                           f"{w:.6f} {l:.6f} {h:.6f} {angle:.6f}\n")
            print(f"保存到: {output_file}")
    
    return results


def visualize_bbox_info(bboxes):
    """
    可视化显示bbox信息（用于调试）
    """
    print("\n" + "="*80)
    print("3D Bounding Box 信息汇总")
    print("="*80)
    print(f"{'类型':<20} {'中心坐标(x,y,z)':<30} {'尺寸(w,l,h)':<25} {'角度(rad)':<10}")
    print("-"*80)
    
    for bbox in bboxes:
        obj_type, x, y, z, w, l, h, angle = bbox
        center_str = f"({x:6.2f}, {y:6.2f}, {z:6.2f})"
        size_str = f"({w:4.2f}, {l:4.2f}, {h:4.2f})"
        print(f"{obj_type:<20} {center_str:<30} {size_str:<25} {angle:6.2f}")


# 使用示例
if __name__ == "__main__":
    # 设置输入输出路径
    input_label_dir = "data/custom_mini/labels_cam_cs"  # 修改为你的标签目录
    output_label_dir = "data/custom_mini/labels"  # 可选：保存转换后的标签
    calib_dir = "data/custom_mini/calib"  # 标定文件目录
    # 处理标签
    results = process_label_directory(
        label_dir=input_label_dir,
        output_dir=output_label_dir,  # 如果不需要保存文件，设为None
        calib_dir=calib_dir
    )
    
    # 示例：查看第一个文件的详细信息
    if results:
        first_file = list(results.keys())[0]
        print(f"\n\n详细信息 - {first_file}:")
        visualize_bbox_info(results[first_file])
    
    print(f"\n总共处理了 {len(results)} 个文件")
