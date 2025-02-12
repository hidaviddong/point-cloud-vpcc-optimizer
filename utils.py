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