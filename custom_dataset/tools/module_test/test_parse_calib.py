import sys
sys.path.insert(0, 'custom_dataset/tools')
from data_converter.custom_converter import read_calib_file, parse_calib_to_camera_params
import numpy as np

# 测试读取标定文件
calib_path = 'data/nuscenes-kitti-format/mini_val/calib/000000.txt'
calib_data = read_calib_file(calib_path)
print('Calib data keys:', list(calib_data.keys()))
print('P2 shape:', calib_data['P2'].shape if 'P2' in calib_data else 'Not found')
print('Tr_velo_to_cam shape:', calib_data['Tr_velo_to_cam'].shape if 'Tr_velo_to_cam' in calib_data else 'Not found')

# 测试解析函数
cam_intrinsic, cam_extrinsic_r, cam_extrinsic_t = parse_calib_to_camera_params(calib_data)
print('\\nCamera intrinsic shape:', cam_intrinsic.shape)
print('Camera intrinsic:\\n', cam_intrinsic)
print('\\nCamera extrinsic rotation (quaternion):', cam_extrinsic_r)
print('Camera extrinsic translation:', cam_extrinsic_t)
