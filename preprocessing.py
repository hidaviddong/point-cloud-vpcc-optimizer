# 1、匹配文件夹内的点云文件、切块

from collections import defaultdict
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import shutil

def chunk_point_cloud_fixed_size(points, block_size=100, cube_size=1024, overlap=1, device='cuda'):
    """
    将点云数据切分为固定大小的块，支持可选偏移，并确保块的数量和大小一致。
    """
    points = torch.tensor(points, device=device, dtype=torch.float32)
    coords = points

    stride = block_size - overlap
    x_range = torch.arange(0, cube_size, stride, device=device)
    y_range = torch.arange(0, cube_size, stride, device=device)
    z_range = torch.arange(0, cube_size, stride, device=device)

    blocks = []

    for x in x_range:
        for y in y_range:
            for z in z_range:
                mask = (
                        (coords[:, 0] >= x) & (coords[:, 0] < x + block_size) &
                        (coords[:, 1] >= y) & (coords[:, 1] < y + block_size) &
                        (coords[:, 2] >= z) & (coords[:, 2] < z + block_size)
                )

                block_coords = coords[mask]
                if len(block_coords) >= 0:# 后续还有匹配，不能舍去
                    blocks.append((block_coords.cpu().numpy(), (x.item(), y.item(), z.item())))

                # 清理未使用的张量
                del block_coords
                torch.cuda.empty_cache()

    print(f"去除空的block后，总切块数: {len(blocks)}")
    return blocks

class PointCloudDataset(Dataset):
    def __init__(self, folder_A, folder_B, block_folder, block_size, cube_size):
        # path
        self.folder_A = folder_A
        self.folder_B = folder_B
        self.block_folder = block_folder

        # size
        self.block_size = block_size
        self.cube_size = cube_size


        self.file_pairs = self._get_file_pairs()
        self._preprocess_data()

    def merge_blocks(self, block_folder, output_folder):
        """
        根据文件名中的特定部分合并块并保存为多个完整的PLY文件。
        """
        blocks = defaultdict(list)

        # 遍历块文件夹中的所有块文件
        for block_file in sorted(os.listdir(block_folder)):
            if block_file.endswith('.ply'):
                # 提取文件名中的特定部分
                parts = block_file.split('_')
                if len(parts) > 3:
                    key = '_'.join(parts[:3])
                else:
                    key = '_'.join(parts[:2])
                block_path = os.path.join(block_folder, block_file)
                blocks[key].append(block_path)

        # 合并每个特定部分的块
        for key, block_files in blocks.items():
            all_points = []

            for block_file in block_files:
                block_points = self._load_ply(block_file)
                all_points.append(block_points)

            # 合并所有块
            all_points = np.vstack(all_points)

            # 确定输出文件名
            if 'vox10' in key:
                output_file = os.path.join(output_folder, f"{key}_origin.ply")
            elif 'S26C03R03_rec' in key:
                output_file = os.path.join(output_folder, f"{key}.ply")

            # 保存为一个完整的PLY文件
            self._save_ply(all_points, output_file)
            print(f"合并后的点云保存为: {output_file}")

    def _get_file_pairs(self):
        # 获取folder_A和folder_B中的文件对
        files_A = sorted([f for f in os.listdir(self.folder_A) if f.endswith('.ply')])
        files_B = sorted([f for f in os.listdir(self.folder_B) if f.endswith('.ply')])
        return list(zip(files_A, files_B))

    def _preprocess_data(self):
        adjusted_A_folder = os.path.join(self.block_folder, 'block_origin')
        adjusted_B_folder = os.path.join(self.block_folder, 'block_compress')
        if os.path.exists(adjusted_A_folder):
            shutil.rmtree(adjusted_A_folder)
        if os.path.exists(adjusted_B_folder):
            shutil.rmtree(adjusted_B_folder)
        os.makedirs(adjusted_A_folder, exist_ok=True)
        os.makedirs(adjusted_B_folder, exist_ok=True)

        # 遍历文件对并进行kd-tree匹配
        for file_A, file_B in self.file_pairs:
            print('开始处理：', self.folder_A, file_A, file_B)
            points_A = self._load_ply(os.path.join(self.folder_A, file_A))
            points_B = self._load_ply(os.path.join(self.folder_B, file_B))
            print(f"origin nums: {points_A.shape[0]}",f" compress nums: {points_B.shape[0]}")

            chunks_A = chunk_point_cloud_fixed_size(points_A, self.block_size, self.cube_size)
            chunks_B = chunk_point_cloud_fixed_size(points_B, self.block_size, self.cube_size)

            adjusted_chunks_A = []
            adjusted_chunks_B = []
            for (chunk_A, index_A), (chunk_B, index_B) in zip(chunks_A, chunks_B):
                if index_A == index_B and len(chunk_B) > 0 and len(chunk_A) > 0:
                    adjusted_chunks_A.append(chunk_A)
                    adjusted_chunks_B.append(chunk_B)
            # print('-------------------------------开始打印每块的点数------------------------')
            nums_a,nums_b = 0,0
            for i in range(len(adjusted_chunks_A)):
                # print(f'第{i}块: ', adjusted_chunks_A[i].shape, adjusted_chunks_B[i].shape)
                nums_a+=int(len(adjusted_chunks_A[i]))
                nums_b+=int(len(adjusted_chunks_B[i]))
                file_A = file_A.replace('.ply', '')
                file_B = file_B.replace('.ply', '')
                self._save_ply(adjusted_chunks_A[i], os.path.join(adjusted_A_folder, f"{file_A}_block_{i}.ply"))
                self._save_ply(adjusted_chunks_B[i], os.path.join(adjusted_B_folder, f"{file_B}_block_{i}.ply"))
            print('切块后总点数：',nums_a,nums_b)
    def _load_ply(self, file_path):
        # 只要坐标
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        return points

    def _save_ply(self, points, file_path):
        # 使用open3d保存点云数据为ply格式，不包含颜色
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        color = np.array([[0, 0, 0]] * len(points))
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pass



def main():
    base_folder = 'data30/soldier'

    folder_A = f'{base_folder}/original'
    folder_B = f'{base_folder}/compress'
    block_folder = f'{base_folder}/block160'

    PointCloudDataset(folder_A=folder_A,
                    folder_B=folder_B,
                    block_folder=block_folder,
                    block_size=160,
                    cube_size=1024
                                )


if __name__ == '__main__':
    main()

# 2、对每一块，去找 new_original 





