# tools/data_convert/custom_converter.py
import os
from os import path as osp
import mmcv
import numpy as np
from pyquaternion import Quaternion

# NuScenes标准类别映射：将标注标签映射到NuScenes标准类别名称
# 根据您的实际标注标签，修改左侧的键值
# 支持大小写不敏感的匹配
class_id = {
    "Car": "car",
    "car": "car",
    "Truck": "truck",
    "truck": "truck",
    "Trailer": "trailer",
    "trailer": "trailer",
    "Bus": "bus",
    "bus": "bus",
    "ConstructionVehicle": "construction_vehicle",
    "construction_vehicle": "construction_vehicle",
    "Bicycle": "bicycle",
    "bicycle": "bicycle",
    "Motorcycle": "motorcycle",
    "motorcycle": "motorcycle",
    "Pedestrian": "pedestrian",
    "pedestrian": "pedestrian",
    "TrafficCone": "traffic_cone",
    "traffic_cone": "traffic_cone",
    "trafficcone": "traffic_cone",
    "Barrier": "barrier",
    "barrier": "barrier"
}                                   

def _read_imageset_file(path):
    try:
        result = []
        with open(path, 'r') as f:
            for line in f:
                # 去除行尾的换行符
                clean_line = line.rstrip('\n')
                result.append(clean_line)
        return result
    except FileNotFoundError:
        print(f"文件 {path} 未找到，请检查路径。")
        return []
    except ValueError:
        print(f"文件 {path} 中的内容存在问题，请检查文件内容。")
        return []

def get_train_val_scenes(root_path):
    """
    划分训练集和测试集
    如果ImageSets文件不存在，会创建空的文件
    """
    imageset_folder = osp.join(root_path, 'ImageSets')
    train_txt_path = osp.join(imageset_folder, 'train.txt')
    val_txt_path = osp.join(imageset_folder, 'val.txt')
    
    # 如果ImageSets目录不存在，创建它
    if not osp.exists(imageset_folder):
        os.makedirs(imageset_folder, exist_ok=True)
        print(f"Warning: ImageSets directory not found, created: {imageset_folder}")
    
    # 如果train.txt不存在，创建空文件
    if not osp.exists(train_txt_path):
        with open(train_txt_path, 'w') as f:
            pass
        print(f"Warning: {train_txt_path} not found, created empty file. Please add frame IDs to it.")
    
    # 如果val.txt不存在，创建空文件
    if not osp.exists(val_txt_path):
        with open(val_txt_path, 'w') as f:
            pass
        print(f"Warning: {val_txt_path} not found, created empty file. Please add frame IDs to it.")
    
    train_img_ids = _read_imageset_file(train_txt_path)
    val_img_ids = _read_imageset_file(val_txt_path)
    # test_img_ids = _read_imageset_file(str(imageset_folder + '/test.txt'))    
    return train_img_ids, val_img_ids


def read_calib_file(filepath):
    """
    读取KITTI格式的标定文件
    
    Args:
        filepath: 标定文件路径
        
    Returns:
        dict: 包含标定参数的字典
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(':', 1)
            # 移除空格并转换为numpy数组
            data[key] = np.array([float(x) for x in value.strip().split()])
    return data


def parse_calib_to_camera_params(calib_data):
    """
    从KITTI标定数据中提取相机内参和外参
    
    Args:
        calib_data: read_calib_file返回的字典
        
    Returns:
        cam_intrinsic: 相机内参矩阵 (3x3)
        cam_extrinsic_r: 相机外参旋转四元数 (w, x, y, z)
        cam_extrinsic_t: 相机外参平移向量 (x, y, z)
    """
    # 从P2矩阵提取相机内参（前3x3部分）
    # P2格式: [fx 0 cx tx; 0 fy cy ty; 0 0 1 tz]
    if 'P2' in calib_data:
        P2 = calib_data['P2'].reshape(3, 4)
        cam_intrinsic = P2[:, :3]  # 提取前3x3作为内参矩阵
    else:
        # 如果P2不存在，使用默认值
        print("Warning: P2 not found in calib file, using default intrinsic")
        cam_intrinsic = np.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]], dtype=np.float32)
    
    # 从Tr_velo_to_cam提取激光雷达到相机的变换
    # Tr_velo_to_cam格式: [R11 R12 R13 tx; R21 R22 R23 ty; R31 R32 R33 tz]
    if 'Tr_velo_to_cam' in calib_data:
        Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
        rotation_matrix = Tr_velo_to_cam[:, :3]  # 提取旋转矩阵
        translation = Tr_velo_to_cam[:, 3]  # 提取平移向量
    else:
        # 如果Tr_velo_to_cam不存在，使用单位矩阵和零平移
        print("Warning: Tr_velo_to_cam not found in calib file, using identity")
        rotation_matrix = np.eye(3)
        translation = np.zeros(3)
    
    # 应用R0_rect校正（如果存在）
    if 'R0_rect' in calib_data:
        R0_rect = calib_data['R0_rect'].reshape(3, 3)
        # 组合旋转矩阵: R0_rect * rotation_matrix
        rotation_matrix = R0_rect @ rotation_matrix
    
    # 将旋转矩阵转换为四元数 (w, x, y, z)
    # 使用pyquaternion直接从旋转矩阵创建四元数
    quaternion = Quaternion(matrix=rotation_matrix)
    # Quaternion对象可以直接转换为数组，格式为 (w, x, y, z)
    cam_extrinsic_r = np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
    
    cam_extrinsic_t = translation
    
    return cam_intrinsic, cam_extrinsic_r, cam_extrinsic_t


def create_custom_infos(
    root_path, info_prefix
):
    train_scenes, val_scenes= get_train_val_scenes(root_path)
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(root_path, train_scenes, val_scenes)

    metadata = dict(version="custom")

    print(
        "train sample: {}, val sample: {}".format(
            len(train_nusc_infos), len(val_nusc_infos)
        )
    )

    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
    mmcv.dump(data, info_path)
    data["infos"] = val_nusc_infos
    info_val_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
    mmcv.dump(data, info_val_path)


def _fill_trainval_infos(root_path, train_scenes, val_scenes, test=False):
    """
    填充训练和验证集信息
    支持从每帧的标定文件中读取标定参数
    """
    # 默认标定参数（当标定文件不存在时使用）
    default_cam_intrinsic = np.array([625.30933437, 0.0, 961.13004221,
                                      0.0, 623.64759937, 546.09541553,
                                      0, 0, 1]).reshape((3, 3))
    default_cam_front_extrinsic_r = np.array([0.703439089347274, -0.7103943323318405, 
                                              -0.020487775159276703, -0.009674256462361884])
    default_cam_front_extrinsic_t = np.array([-0.0, 5.1654, 0.891921])

    train_kitti_infos = []
    val_kitti_infos = []

    available_scene_names = train_scenes + val_scenes
    for sid, scenes_id in enumerate(available_scene_names):  
        frame_id = scenes_id
        lidar_path = osp.abspath(osp.join(root_path, "points", str(scenes_id) + ".bin"))
        label_path = osp.abspath(osp.join(root_path, "labels", str(scenes_id) + ".txt"))
        
        # 尝试从标定文件中读取标定参数
        # 优先查找 calib 目录，如果不存在则查找 training/calib 目录
        calib_path = osp.join(root_path, "calib", str(scenes_id) + ".txt")
        if not osp.exists(calib_path):
            calib_path = osp.join(root_path, "training", "calib", str(scenes_id) + ".txt")
        
        # 读取标定参数
        if osp.exists(calib_path):
            try:
                calib_data = read_calib_file(calib_path)
                cam_intrinsic, cam_extrinsic_r, cam_extrinsic_t = parse_calib_to_camera_params(calib_data)
            except Exception as e:
                print(f"Warning: Failed to parse calib file {calib_path}: {e}")
                print(f"Using default calibration parameters for frame {scenes_id}")
                cam_intrinsic = default_cam_intrinsic
                cam_extrinsic_r = default_cam_front_extrinsic_r
                cam_extrinsic_t = default_cam_front_extrinsic_t
        else:
            # 如果标定文件不存在，使用默认值
            print(f"Warning: Calib file not found for frame {scenes_id}, using default calibration parameters")
            cam_intrinsic = default_cam_intrinsic
            cam_extrinsic_r = default_cam_front_extrinsic_r
            cam_extrinsic_t = default_cam_front_extrinsic_t
        
        # dataset infos
        lidar2ego_rotation_matrix = np.eye(3).astype(np.float32)
        lidar2ego_translation = np.zeros(3).T
        info = {
            "frame_id": frame_id,
            'lidar_path': lidar_path,
            'token': '',
            'sweeps': [],
            'cams': dict(),
            'radars': dict(), 
            'lidar2ego_translation': lidar2ego_translation,
            'lidar2ego_rotation': lidar2ego_rotation_matrix,
            'timestamp': scenes_id,
        }

        # 只使用前视相机
        camera_types = [
            "cam_front",
        ]        

        for cam in camera_types:
            cam_path = osp.abspath(osp.join(root_path, "camera", cam, str(scenes_id) + ".jpg"))
            cam_info = {
                'data_path': cam_path,
                'type': cam,
                'sensor2ego_translation': cam_extrinsic_t,
                'sensor2ego_rotation': cam_extrinsic_r,
                'sensor2lidar_translation': cam_extrinsic_t,
                'sensor2lidar_rotation': Quaternion(cam_extrinsic_r).rotation_matrix,
                'cam_intrinsic': cam_intrinsic,
            }
            info["cams"].update({cam: cam_info})            

        gt_boxes = []
        gt_names = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            line_list = line.split()
            
            if len(line_list) < 8:
                print(f"Warning: Line has too few fields in {label_path}, skipping: {line}")
                continue
            
            # 根据实际标签文件格式：class_name x y z ... w l h angle ...
            # 检查第一个字段是否是类别名称（支持大小写不敏感）
            class_name = None
            numeric_fields = None
            
            # 首先尝试第一个字段（最常见的情况）
            first_field = line_list[0]
            if first_field in class_id:
                # 格式：class_name x y z w l h angle [其他参数...]
                class_name = first_field
                # 取第2-8个字段作为 x y z w l h angle (7个数值)
                numeric_fields = line_list[1:8]
            # 尝试第一个字段的小写版本
            elif first_field.lower() in class_id:
                class_name = first_field.lower()
                numeric_fields = line_list[1:8]
            # 尝试第一个字段的首字母大写版本
            elif first_field.capitalize() in class_id:
                class_name = first_field.capitalize()
                numeric_fields = line_list[1:8]
            # 尝试最后一个字段（标准KITTI格式）
            elif len(line_list) >= 8 and line_list[-1] in class_id:
                class_name = line_list[-1]
                numeric_fields = line_list[:7]
            elif len(line_list) >= 8 and line_list[-1].lower() in class_id:
                class_name = line_list[-1].lower()
                numeric_fields = line_list[:7]
            else:
                # 尝试查找类别名称（遍历所有字段，支持大小写不敏感）
                for i, field in enumerate(line_list):
                    if field in class_id:
                        class_name = field
                        if i == 0:
                            numeric_fields = line_list[1:8]
                        elif i == len(line_list) - 1:
                            numeric_fields = line_list[:7]
                        break
                    elif field.lower() in class_id:
                        class_name = field.lower()
                        if i == 0:
                            numeric_fields = line_list[1:8]
                        elif i == len(line_list) - 1:
                            numeric_fields = line_list[:7]
                        break
                
                if class_name is None:
                    print(f"Warning: Could not find class name in {label_path}, skipping line: {line}")
                    print(f"  First field: '{line_list[0]}', Available classes: {list(class_id.keys())[:5]}...")
                    continue
            
            # 尝试将数值字段转换为float
            try:
                box_values = [float(x) for x in numeric_fields]
                if len(box_values) != 7:
                    print(f"Warning: Expected 7 box values, got {len(box_values)} in {label_path}, skipping line: {line}")
                    continue
                gt_boxes.append(np.array(box_values, dtype=np.float32))
                gt_names.append(class_id[class_name])
            except (ValueError, IndexError, TypeError) as e:
                print(f"Warning: Could not parse box values in {label_path}, skipping line: {line}, error: {e}")
                continue
        
        # 处理空标注框的情况
        if len(gt_boxes) == 0:
            # 如果为空，创建形状为 (0, 7) 的数组，而不是 (0,)
            info["gt_boxes"] = np.zeros((0, 7), dtype=np.float32)
            info["gt_names"] = np.array([], dtype=object)
            info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
            info['valid_flag'] = np.array([], dtype=bool)
        else:
            info["gt_boxes"] = np.array(gt_boxes)
            info["gt_names"] = np.array(gt_names)
            info['gt_velocity'] = np.array([0, 0] * len(gt_names)).reshape(-1, 2)  # 没有速度，只是为了跟nuscences对齐
            info['valid_flag'] = np.array(True * len(gt_names)).reshape(-1)
        info["lidar_path"] = lidar_path
        
        if scenes_id in train_scenes:
            train_kitti_infos.append(info)
        if scenes_id in val_scenes:
            val_kitti_infos.append(info)
    return train_kitti_infos, val_kitti_infos