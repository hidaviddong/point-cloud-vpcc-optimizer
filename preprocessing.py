import os
import re
import torch
from config import COMPRESS_DIR, COMPRESS_BLOCK_DIR, ORIGIN_DIR, ORIGIN_BLOCK_DIR
from utils import load_ply, save_ply, get_file_pairs

def chunk_point_cloud_fixed_size(points, block_size=100, cube_size=1024, overlap=1, device='cuda'):
    """将点云数据切分为固定大小的块"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():  # For Apple Silicon
        device = 'mps'
    else:
        device = 'cpu'
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
                if len(block_coords) >= 0:
                    blocks.append((block_coords.cpu().numpy(), (x.item(), y.item(), z.item())))
                del block_coords
                torch.cuda.empty_cache()
    
    print(f"总切块数: {len(blocks)}")
    return blocks

def process_point_cloud_pair(file_A, file_B, block_size, cube_size):
    """处理一对点云文件"""
    print('处理文件对：', file_A, file_B)
    
    # 加载点云
    points_A = load_ply(os.path.join(ORIGIN_DIR, file_A))
    points_B = load_ply(os.path.join(COMPRESS_DIR, file_B))
    print(f"原始点云数量: {points_A.shape[0]}, 压缩点云数量: {points_B.shape[0]}")

    # 切分点云
    chunks_A = chunk_point_cloud_fixed_size(points_A, block_size, cube_size)
    chunks_B = chunk_point_cloud_fixed_size(points_B, block_size, cube_size)

    # 保存匹配的块
    nums_a, nums_b = 0, 0
    for i, ((chunk_A, index_A), (chunk_B, index_B)) in enumerate(zip(chunks_A, chunks_B)):
        if index_A == index_B and len(chunk_B) > 0 and len(chunk_A) > 0:
            nums_a += len(chunk_A)
            nums_b += len(chunk_B)
            
            # 保存块
            save_ply(chunk_A, os.path.join(ORIGIN_BLOCK_DIR, 
                    f"{file_A.replace('.ply', '')}_block_{i}.ply"))
            save_ply(chunk_B, os.path.join(COMPRESS_BLOCK_DIR, 
                    f"{file_B.replace('.ply', '')}_block_{i}.ply"))
    
    print(f"切块后总点数：原始 {nums_a}, 压缩 {nums_b}")

def process_all_point_clouds(block_size=160, cube_size=1024):
    """处理所有点云文件"""
    # 确保输出目录存在
    os.makedirs(ORIGIN_BLOCK_DIR, exist_ok=True)
    os.makedirs(COMPRESS_BLOCK_DIR, exist_ok=True)
    
    # 获取文件对并处理
    file_pairs = get_file_pairs(ORIGIN_DIR, COMPRESS_DIR)
    for file_A, file_B in file_pairs:
        process_point_cloud_pair(file_A, file_B, block_size, cube_size)

def main():
    process_all_point_clouds(block_size=160, cube_size=1024)

if __name__ == '__main__':
    main()
