import numpy as np
from scipy import ndimage
import nibabel as nib
import mclahe as mc
import skimage

def read_nii_file(path):
    """ Read nii file (image, label) from path using nibabel library
    """
    scan = nib.load(path)
    affine = scan.affine
    header = scan.header
    volume = scan.get_fdata()
    return volume, affine, header

def clip(volume, min=-300, max=500):
    volume = np.clip(volume, min, max)
    return volume

def normalize(volume):
    min, max = np.min(volume), np.max(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resample(volume, pixdim, new_spacing=[1.5, 1.5, 1.5], label=False):
    origin_space = volume.shape
    new_space = [
        origin_space[0] * pixdim[0] / new_spacing[0],
        origin_space[1] * pixdim[1] / new_spacing[1],
        origin_space[2] * pixdim[2] / new_spacing[2]]

    label = not label
    order = 1 if label else 0
    data_resampled = skimage.transform.resize(volume, new_space, order=order, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=label, anti_aliasing_sigma=None)
    return data_resampled

def resample_back(volume, original_shape, is_label=False):
    is_label = not is_label
    order = 3 if is_label else 0  # order=0, nearest
    data_resampled = skimage.transform.resize(volume, original_shape, order=order, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=is_label, anti_aliasing_sigma=None)
    return data_resampled

def resize_volume(volume, desired_size, is_label=False):
    """Resize across z-axis"""
    current_width, current_height, current_depth = volume.shape
    # Compute depth factor
    depth = current_depth / desired_size[-1]
    width = current_width / desired_size[0]
    height = current_height / desired_size[1]
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    if is_label:
        volume = ndimage.zoom(volume, (width_factor, height_factor, depth_factor), order=0, mode='nearest')
    else:
        volume = ndimage.zoom(volume, (width_factor, height_factor, depth_factor), order=1, mode='nearest')
    return volume

def resize_image_with_crop_or_pad(image, img_size=(128, 128, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """
    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    rank = len(img_size)    # Get the image dimensionality

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs), slicer

def detect_by_mask(volume, expand=0):
    lower_bound = 0
    seg = np.where(volume == 1)
    bbx = 0, 0, 0, 0, 0, 0
    if len(seg) != 0 and len(seg[0]) != 0 and len(seg[1]) != 0 and len(seg[-1]) != 0:
        x_min = max(int(np.min(seg[0])) - expand, lower_bound)
        x_max = min(int(np.max(seg[0])) + expand, volume.shape[0] - 1)
        y_min = max(int(np.min(seg[1])) - expand, lower_bound)
        y_max = min(int(np.max(seg[1])) + expand, volume.shape[1] - 1)
        z_min = max(int(np.min(seg[-1])) - expand, lower_bound)
        z_max = min(int(np.max(seg[-1])) + expand, volume.shape[-1] - 1)
        bbx = x_min, x_max, y_min, y_max, z_min, z_max
    return bbx

def find_uniform_bbx(volume, bbx, size, origin_space, new_space, transform=False):
    lower_bound = 0
    # upper_bound = (volume.shape[0] - 1, volume.shape[1] - 1, volume.shape[2] - 1)
    upper_bound = (volume.shape[0], volume.shape[1], volume.shape[2])
    x_min, x_max, y_min, y_max, z_min, z_max = bbx
    
    bbx_center = (x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2
    if transform:
        bbx_center = list(map(lambda x, y, z: x * y / z, bbx_center, origin_space, new_space))

    bbx_center = list(bbx_center)
    for i in range(3):
        bbx_center[i] += max(0, size[i]//2-bbx_center[i]) + min(0, upper_bound[i] - bbx_center[i] - size[i]//2)
    
    real_x_min = max(int(bbx_center[0] - size[0]//2), lower_bound)
    real_x_max = min(int(bbx_center[0] + size[0]//2), upper_bound[0])
    
    real_y_min = max(int(bbx_center[1] - size[1]//2), lower_bound)
    real_y_max = min(int(bbx_center[1] + size[1]//2), upper_bound[1])
    
    real_z_min = max(int(bbx_center[-1] - size[-1]//2), lower_bound)
    real_z_max = min(int(bbx_center[-1] + size[-1]//2), upper_bound[2])
    real_bbx = real_x_min, real_x_max, real_y_min, real_y_max, real_z_min, real_z_max
    return real_bbx

def crop_volume(img, label, bbx):
    img = img[bbx[0]:bbx[1], bbx[2]:bbx[3], bbx[4]:bbx[5]]
    label = label[bbx[0]:bbx[1], bbx[2]:bbx[3], bbx[4]:bbx[5]]
    return img, label

def clahe_3d(voxel, clip_limit=0.0, kernel_size=(8, 8, 8)):
    voxel = mc.mclahe(voxel, kernel_size=kernel_size,
                      n_bins=128,
                      clip_limit=clip_limit,
                      adaptive_hist_range=False,)
    return (voxel*255.).astype(np.uint8).clip(0, 255)