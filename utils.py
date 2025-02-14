import os
import re
import open3d as o3d
import numpy as np
import torch

def get_file_pairs(file_A, file_B):
    """匹配文件名中四位数字相同的文件对"""
    files_A = [f for f in os.listdir(file_A) if f.endswith('.ply')]
    files_B = [f for f in os.listdir(file_B) if f.endswith('.ply')]
    print(f"找到 {len(files_A)} 个PLY文件和 {len(files_B)} 个PLY文件")
    pattern = r'_(\d{4})\.ply$'
    
    # 为压缩文件创建数字到文件名的映射
    number_to_fileB = {
        re.search(pattern, file_B).group(1): file_B 
        for file_B in files_B 
        if re.search(pattern, file_B)
    }

    # 匹配文件对
    pairs = [
        (file_A, number_to_fileB[number])
        for file_A in files_A
        if (match := re.search(pattern, file_A)) 
        and (number := match.group(1)) in number_to_fileB
    ]
    
    pairs.sort()
    print(f"找到 {len(pairs)} 对匹配的文件")
    for pair in pairs:
        print(f"匹配: {pair[0]} <-> {pair[1]}")
        
    return pairs


def load_ply(file_path):
    """加载PLY文件，返回点云坐标"""
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def save_ply(points, file_path):
    """保存点云坐标为PLY文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(points), 3)))
    o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

def get_matching_paths(file_a, origin_dir, compress_dir):
    """获取匹配的文件完整路径
    
    Args:
        file_a: compress 中的原始文件名
        origin_dir: 原始文件目录
        compress_dir: 压缩文件目录
        
    Returns:
        (uncompressed_path, reconstructed_path)
    """
    # 获取file_a的特征数字
    four_digit = re.search(r'rec_(\d{4})', file_a).group(1)
    block_num = re.search(r'block_(\d+)\.ply', file_a).group(1)
    
    # 在origin_dir中查找匹配文件
    for file_b in os.listdir(origin_dir):
        if (re.search(rf'vox10_({four_digit})', file_b) 
            and re.search(rf'block_({block_num})\.ply', file_b)):
            return (
                os.path.join(compress_dir, file_a),
                os.path.join(origin_dir, file_b)
            )
            
    raise FileNotFoundError(f"No matching file found for {file_a}")

def extract_points(output):
    """pc_error工具提取匹配的点对"""
    point_pairs = {}
    for line in output.splitlines():
        if 'Point A[' in line:
            try:
                a_index = int(line.split('A[')[1].split(']')[0])
                parts = line.split(' -> ')
                if len(parts) == 2:
                    a_part = parts[0].split('(')[1].split(')')[0]
                    ax, ay, az = map(float, a_part.split(','))
                    b_part = parts[1].split('(')[1].split(')')[0]
                    bx, by, bz = map(float, b_part.split(','))
                    point_pairs[a_index] = ((ax, ay, az), (bx, by, bz))
            except Exception:
                continue
    return point_pairs