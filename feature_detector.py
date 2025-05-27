import os
import cv2
import numpy as np
import h5py
from image_processing import ImageLoader, FeatureDetector, get_path_and_intrinsic

def extract_and_save_features():
    """Extract features for all KITTI sequences and save to HDF5 files"""
    
    # KITTI sequences available
    sequences = [0, 1, 3, 5, 9]  # Add more if you have them
    methods = ['SIFT']  # Add other methods if needed: ['SIFT', 'ORB', 'SURF']
    rates = [10]  # Add other rates if needed: [1, 5, 10]
    
    for dataset_num in sequences:
        print(f"\n=== Processing sequence {dataset_num:02d} ===")
        
        try:
            # Load dataset
            path_images, path_ground_truth, K = get_path_and_intrinsic(dataset_num)
            print(f"Images path: {path_images}")
            
            for rate in rates:
                print(f"\nProcessing with rate {rate}")
                
                # Load images
                loader = ImageLoader(path_images, path_ground_truth, desired_rate=rate)
                images, ground_truth_positions = loader.load_images()
                print(f"Loaded {len(images)} images")
                
                for method in methods:
                    print(f"Extracting {method} features...")
                    
                    # Create filename
                    filename = f"Sequence {dataset_num:02d} features {method}.h5"
                    
                    # Check if file already exists
                    if os.path.exists(filename):
                        print(f"File {filename} already exists, skipping...")
                        continue
                    
                    # Extract features
                    detector = FeatureDetector(method)
                    features = detector.detect_all_features(images)
                    
                    # Save to HDF5
                    save_features_to_hdf5(features, filename, method, rate, dataset_num, K)
                    print(f"Saved features to {filename}")
                    
        except Exception as e:
            print(f"Error processing sequence {dataset_num}: {e}")
            continue

def save_features_to_hdf5(features, filename, method, rate, dataset_num, K):
    """Save features to HDF5 file"""
    
    with h5py.File(filename, 'w') as f:
        # Store metadata
        f.attrs['method'] = method
        f.attrs['rate'] = rate
        f.attrs['dataset_num'] = dataset_num
        f.attrs['num_images'] = len(features)
        f.attrs['camera_matrix'] = K
        
        # Store features for each image
        for i, (keypoints, descriptors) in enumerate(features):
            img_group = f.create_group(f'image_{i:06d}')
            
            if keypoints is not None and len(keypoints) > 0:
                # Convert keypoints to numpy array (x, y, angle, response, octave, class_id)
                kp_array = np.array([[kp.pt[0], kp.pt[1], kp.angle, kp.response, 
                                    kp.octave, kp.class_id] for kp in keypoints])
                img_group.create_dataset('keypoints', data=kp_array)
                
                if descriptors is not None:
                    img_group.create_dataset('descriptors', data=descriptors)
                else:
                    # Create empty dataset if no descriptors
                    img_group.create_dataset('descriptors', data=np.array([]))
            else:
                # Create empty datasets for images with no features
                img_group.create_dataset('keypoints', data=np.array([]).reshape(0, 6))
                img_group.create_dataset('descriptors', data=np.array([]))

def load_features_from_hdf5(filename):
    """Load features from HDF5 file in the same format as FeatureDetector.detect_all_features()"""
    
    features = []
    
    with h5py.File(filename, 'r') as f:
        num_images = f.attrs['num_images']
        
        for i in range(num_images):
            img_key = f'image_{i:06d}'
            if img_key in f:
                kp_data = f[img_key]['keypoints'][:]
                desc_data = f[img_key]['descriptors'][:]
                
                if len(kp_data) > 0:
                    # Convert back to cv2.KeyPoint objects
                    keypoints = []
                    for kp_row in kp_data:
                        kp = cv2.KeyPoint(x=kp_row[0], y=kp_row[1], size=1, 
                                        angle=kp_row[2], response=kp_row[3], 
                                        octave=int(kp_row[4]), class_id=int(kp_row[5]))
                        keypoints.append(kp)
                    
                    descriptors = desc_data if len(desc_data) > 0 else None
                else:
                    keypoints = []
                    descriptors = None
                    
                features.append((keypoints, descriptors))
            else:
                # Handle missing image data
                features.append(([], None))
    
    return features

if __name__ == "__main__":
    print("Starting offline feature extraction...")
    extract_and_save_features()
    print("Feature extraction completed!")