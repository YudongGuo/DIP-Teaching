import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points
from natsort import natsorted

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_points3D_text(path):
    """Read points3D.txt file"""
    points3D = {}
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            data = line.split()
            point_id = int(data[0])
            xyz = np.array([float(x) for x in data[1:4]])
            rgb = np.array([int(x) for x in data[4:7]])
            error = float(data[7])
            points3D[point_id] = {
                'xyz': xyz,
                'rgb': rgb,
                'error': error
            }
    return points3D

def read_images_text(path):
    """Read images.txt file and return images sorted by name"""
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # First collect all images
    for i in range(0, len(lines), 2):
        line = lines[i]
        if line[0] == '#':
            continue
        data = line.split()
        image_id = int(data[0])
        qvec = np.array([float(x) for x in data[1:5]])
        tvec = np.array([float(x) for x in data[5:8]])
        camera_id = int(data[8])
        name = data[9]
        
        R = qvec2rotmat(qvec)
        
        images[image_id] = {
            'R': R,
            't': tvec.reshape(3,1),
            'camera_id': camera_id,
            'name': name
        }
    
    # Sort images by name and create new ordered dictionary
    sorted_images = dict(natsorted(images.items(), key=lambda x: x[1]['name']))
    
    return sorted_images

def read_cameras_text(path):
    """Read cameras.txt file"""
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            data = line.split()
            camera_id = int(data[0])
            model = data[1]
            width = int(data[2])
            height = int(data[3])
            params = np.array([float(x) for x in data[4:]])
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def get_intrinsic_matrix(camera, downsample_factor=1):
    """Get intrinsic matrix from camera parameters"""
    if camera['model'] == 'PINHOLE':
        fx, fy, cx, cy = camera['params']
        fx, fy, cx, cy = fx / downsample_factor, fy / downsample_factor, cx / downsample_factor, cy / downsample_factor
        K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])
        return K
    else:
        raise ValueError(f"Camera model {camera['model']} not supported yet")




class ColmapDataset(Dataset):
    def __init__(self, data_path, downsample_factor=8, maximum_pts_num = 3000):
        """
        Dataset class for COLMAP data
        """
        sparse_path = os.path.join(data_path, "sparse", "0_text")
        images_dir = os.path.join(data_path, "images")

        self.downsample_factor = downsample_factor
        
        # Load COLMAP data
        self.cameras = read_cameras_text(os.path.join(sparse_path, "cameras.txt"))
        self.images = read_images_text(os.path.join(sparse_path, "images.txt"))
        points3D = read_points3D_text(os.path.join(sparse_path, "points3D.txt"))

        
        # Convert points3D to torch.tensor
        self.points3D_xyz = torch.as_tensor(np.array([p['xyz'] for p in points3D.values()])).float()
        self.points3D_rgb = torch.as_tensor(np.array([p['rgb'] for p in points3D.values()])).float()

        # ### sample 3D points to a specific number
        # indices = sample_farthest_points(self.points3D_xyz.unsqueeze(0), K = maximum_pts_num)[1]
        # self.points3D_xyz = self.points3D_xyz[indices.reshape(-1)]
        # self.points3D_rgb = self.points3D_rgb[indices.reshape(-1)]
        
        # Get image paths and convert camera parameters
        self.image_paths = []
        self.camera_data = []
        
        for image_id, image_data in self.images.items():
            image_path = os.path.join(images_dir, image_data['name'])
            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                camera = self.cameras[image_data['camera_id']]
                K = get_intrinsic_matrix(camera, downsample_factor)
                self.camera_data.append({
                    'K': K,
                    'R': image_data['R'],
                    't': image_data['t']
                })
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (0,0), fx=1./self.downsample_factor, fy=1./self.downsample_factor)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.FloatTensor(image) / 255.0
        
        # Get camera parameters
        camera_data = self.camera_data[idx]
        K = torch.FloatTensor(camera_data['K'])
        R = torch.FloatTensor(camera_data['R'])
        t = torch.FloatTensor(camera_data['t'])
        
        return {
            'image': image,
            'K': K,
            'R': R,
            't': t,
            'image_path': image_path
        }
