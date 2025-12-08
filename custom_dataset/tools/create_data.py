# projects/tools/create_data.py
import argparse
import os
import sys
from os import path as osp

# 添加项目根目录到 Python 路径，以便导入 custom_dataset
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import data_converter.custom_converter as custom_converter
from data_converter.create_gt_database import create_groundtruth_database

def custom_data_prep(root_path, info_prefix, dataset_name, out_dir, skip_gt_database=False):
    # 创建 info 文件
    print("Creating info files...")
    custom_converter.create_custom_infos(root_path, info_prefix)
    print("Info files created successfully!")
    
    # 创建GT database是可选的，因为它需要编译扩展
    if skip_gt_database:
        print("Skipping GT database creation as requested.")
        return
    
    # 尝试创建 GT database
    print("\nCreating GT database...")
    try:
        create_groundtruth_database(dataset_name,
                                    root_path,
                                    info_prefix,
                                    f"{out_dir}/{info_prefix}_infos_train.pkl")
        print("GT database created successfully!")
    except ImportError as e:
        error_msg = str(e)
        print(f"\nWarning: Could not create GT database due to import error: {error_msg}")
        print("\nThis usually means the compiled extensions are not available.")
        print("To fix this, you need to compile the mmdet3d extensions:")
        print("  1. Make sure you are in the project root directory")
        print("  2. Run: python setup.py develop")
        print("     or: python setup.py build_ext --inplace")
        print("\nIf you want to skip GT database creation, use --skip-gt-database flag.")
        print("Note: Info files have been created successfully.")
        raise  # 重新抛出异常，让用户知道需要修复
    except Exception as e:
        print(f"\nWarning: Could not create GT database: {e}")
        print("Info files have been created successfully.")
        raise  # 重新抛出异常


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="MyCustomDataset", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="/",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="/",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="custom")
parser.add_argument("--skip-gt-database", action="store_true", 
                    help="Skip GT database creation (useful when compiled extensions are not available)")


args = parser.parse_args()

if __name__ == "__main__":

    if args.dataset == "custom":

        custom_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            dataset_name="MyCustomDataset",
            out_dir=args.out_dir,
            skip_gt_database=args.skip_gt_database
        )
# python custom_dataset/tools/create_data.py custom --root-path data/20240617-720 --out-dir data/20240617-720 --extra-tag custom 
        