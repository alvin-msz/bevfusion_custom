import pickle
from os import path as osp
import mmcv
import numpy as np
from mmcv import track_iter_progress
# 延迟导入 box_np_ops，避免触发需要编译扩展的依赖
# from mmdet3d.core.bbox import box_np_ops as box_np_ops
# 延迟导入 build_dataset，避免触发需要编译扩展的依赖
# from mmdet3d.datasets import build_dataset
import torch
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入 custom_dataset
# 获取当前文件的目录，然后向上两级到达项目根目录
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义数据集以注册到数据集注册表
# 注意：如果从 create_data.py 运行，路径已经在 create_data.py 中设置
# 如果直接导入此模块，路径会在上面设置
try:
    from custom_dataset.mmdet3d.datasets.custom_dataset import MyCustomDataset
except ImportError:
    # 如果导入失败，说明路径可能还没设置，将在函数内部延迟导入
    pass

def create_groundtruth_database( 
        dataset_class_name,
        root_path,        
        info_prefix, 
        info_path=None,    
        used_classes=None,
        database_save_path=None,
        db_info_save_path=None,
        with_mask=False):
    
    # 延迟导入 build_dataset 和 box_np_ops，避免在模块导入时触发需要编译扩展的依赖
    # 注意：MyCustomDataset 的导入会在 build_dataset 时自动处理，因为它是通过字符串名称注册的
    from mmdet3d.datasets import build_dataset
    from mmdet3d.core.bbox import box_np_ops as box_np_ops
    
    # 尝试导入自定义数据集以注册到数据集注册表
    # 如果导入失败（由于编译扩展问题），build_dataset 仍然可以通过字符串名称找到它
    try:
        from custom_dataset.mmdet3d.datasets.custom_dataset import MyCustomDataset
    except ImportError as e:
        # 如果导入失败，尝试添加路径并再次导入
        try:
            current_dir = osp.dirname(osp.abspath(__file__))
            project_root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from custom_dataset.mmdet3d.datasets.custom_dataset import MyCustomDataset
        except ImportError:
            # 如果仍然失败，给出警告但继续执行
            # build_dataset 可能仍然可以通过字符串名称找到数据集类
            print(f"Warning: Could not import MyCustomDataset: {e}")
            print("Will try to use dataset by string name 'MyCustomDataset'")

    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=root_path, ann_file=info_path
    )
    dataset_cfg.update(
        use_valid_flag=True,
        pipeline=[
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=4,    # change by why
                use_dim=4,
            ),
            dict(
                type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
            ),
        ],
    )

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(root_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(root_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)
    
    all_db_infos = dict()
    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example["ann_info"]
        # print("annos:  ",annos)
        image_idx = example["sample_idx"]
        # print("image_idx:  ",image_idx)
        points = example["points"].tensor.numpy()
        # print("pointsshape:  ",points.shape)
        # break        
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
        # print("gt_boxes_3d:  ",gt_boxes_3d)
        names = annos["gt_names"]
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)

        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        # print("num_obj:  ",num_obj)
        
        # 跳过没有标注框的样本
        if num_obj == 0:
            continue
        
        # 检查 gt_boxes_3d 的形状是否正确（应该是 [N, 7]）
        if gt_boxes_3d.shape[1] != 7:
            print(f"Warning: Invalid gt_boxes_3d shape {gt_boxes_3d.shape} for sample {image_idx}, expected (N, 7), skipping.")
            continue
        
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if with_mask:
                    db_info.update({"box2d_camera": gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)

