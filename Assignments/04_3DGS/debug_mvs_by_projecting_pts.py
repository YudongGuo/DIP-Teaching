import numpy as np
import cv2
import os
import argparse

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
    """Read images.txt file"""
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    
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
    return images

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

def get_intrinsic_matrix(camera):
    """Get intrinsic matrix from camera parameters"""
    if camera['model'] == 'PINHOLE':
        fx, fy, cx, cy = camera['params']
        K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])
        return K
    else:
        raise ValueError(f"Camera model {camera['model']} not supported yet")

def project_points(points3D, R, t, K):
    """Project 3D points to image plane"""
    # Convert points to camera coordinates
    points3D_cam = (R @ points3D.T + t).T
    
    # Get points in front of camera
    mask = points3D_cam[:, 2] > 0
    
    # Project to image plane
    points3D_cam = points3D_cam[mask]
    points2D = points3D_cam[:, :2] / points3D_cam[:, 2:]
    points2D = (K[:2, :2] @ points2D.T).T + K[:2, 2]
    
    return points2D, mask

def main():
    parser = argparse.ArgumentParser(description='Debug COLMAP output by projecting 3D points onto images')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the input directory containing images in data_dir/images')
    args = parser.parse_args()
    # Set paths
    dataset_path = args.data_dir  # Change this to your dataset path
    sparse_path = os.path.join(dataset_path, "sparse", "0_text")
    images_dir = os.path.join(dataset_path, "images")
    output_dir = os.path.join(dataset_path, "projections")  # Directory for output images
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COLMAP data
    print("Loading COLMAP data...")
    cameras = read_cameras_text(os.path.join(sparse_path, "cameras.txt"))
    images = read_images_text(os.path.join(sparse_path, "images.txt"))
    points3D = read_points3D_text(os.path.join(sparse_path, "points3D.txt"))
    
    # Convert points3D to numpy arrays for efficient processing
    points3D_xyz = np.array([p['xyz'] for p in points3D.values()])
    points3D_rgb = np.array([p['rgb'] for p in points3D.values()])
    
    # Process each image
    print("Processing images...")
    for image_id, image_data in images.items():
        # Get image path
        image_name = image_data['name']
        image_path = os.path.join(images_dir, image_name)
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found")
            continue
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image {image_name}")
            continue
        img_ori = np.array(img)
        img[:] = 0 
        # Get camera parameters
        camera = cameras[image_data['camera_id']]
        K = get_intrinsic_matrix(camera)
        R = image_data['R']
        t = image_data['t']
        
        # Project points
        points2D, mask = project_points(points3D_xyz, R, t, K)
        
        # Draw points on image
        points2D = points2D.astype(int)
        colors = points3D_rgb[mask]
        
        for pt, color in zip(points2D, colors):
            # Check if point is within image bounds
            if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                cv2.circle(img, (pt[0], pt[1]), 2, color[::-1].tolist(), -1)  # BGR to RGB
        
        # Save result
        output_path = os.path.join(output_dir, image_name)
        com_img = np.concatenate((img_ori, img), axis=1)
        com_img = cv2.resize(com_img, (0,0), fx=0.125, fy=0.125)
        cv2.imwrite(output_path, com_img)
        print(f"Processed {image_name}")
    
    print("Done! Check the 'projections' folder for results.")

if __name__ == "__main__":
    main()