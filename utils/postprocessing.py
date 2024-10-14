import numpy as np
import itertools
import denseCRF3D
from skimage import filters
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion

def apply_gaussian_smoothing(volume, sigma=1.0, iterations=1):
    """ Apply gaussian smoothing to the volume
    """
    smoothed_image = gaussian_filter(volume, sigma=sigma)

    threshold = filters.threshold_otsu(smoothed_image)
    binarized_image = (smoothed_image > threshold).astype(np.float32)    

    binarized_image = binary_dilation(binarized_image, iterations=iterations)
    binarized_image = binary_erosion(binarized_image, iterations=iterations)
    
    combined_image = volume + binarized_image
    combined_image = np.clip(combined_image, 0, 1)
    return combined_image

def filter_noise(img, neighbor_distance=2):
    """ Filter the small block of voxels from img. Using Union-Find algorithm.

    Args:
        img: original img
        neighbor_distance: the distance to find the group.

    Returns:
        new img after filtering
    """
    x, y, z = img.shape
    record = np.zeros(img.shape)
    delta = set(itertools.permutations(list(range(-neighbor_distance,neighbor_distance+1)) * 3,3))

    index = 0
    max_area = -1
    remind = []
    max_index = -1
    area = np.count_nonzero(img) // 3
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if record[i,j,k] != 0 or img[i,j,k] == 0:
                    continue

                q = [[i, j, k]]
                index += 1
                record[i, j, k] = index

                count = 1
                while q:
                    a, b, c = q.pop(0)
                    for da, db, dc in delta:
                        da += a
                        db += b
                        dc += c
                        if da < 0 or db < 0 or dc < 0 or da == x or db == y or dc == z or \
                            record[da, db, dc] != 0 or img[da, db, dc] == 0:
                            continue
                        record[da, db, dc] = index
                        count += 1
                        q.append([da, db, dc])

                if count > max_area:
                    max_index = index
                    max_area = count
                if count >= area:
                    remind.append(index)
                    
    # record[record != max_index] = 0
    # record[record == max_index] = 1
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if record[i, j, k] in remind or record[i, j, k] == max_index:
                    record[i, j, k] = 1
                else:
                    record[i, j, k] = 0
    
    img *= record
    return img

def densecrf3d(I, P, param):
    """
    input parameters:
        I: a numpy array of shape [D, H, W, C], where C is the channel number
           type of I should be np.uint8, and the values are in [0, 255]
        P: a probability map of shape [D, H, W, L], where L is the number of classes
           type of P should be np.float32
        param: a tuple giving parameters of CRF. see the following two examples for details.
    """
    return denseCRF3D.densecrf3d(I, P, param)

def denseCRF3d(img, prob):
    """
    input parameters:
        img: a numpy array of shape [D, H, W]
        prob: a probability map of shape [D, H, W]
    """
    img = np.asarray([img], np.float32)
    img = np.transpose(img, [1, 2, 3, 0])
    img = img / img.max()* 255
    img = np.asarray(img, np.uint8)

    # probability map for each class
    prob = 0.5 + (prob - 0.5) * 0.8
    prob = np.asarray([1.0 - prob, prob], np.float32)
    prob = np.transpose(prob, [1, 2, 3, 0])

    dense_crf_param = {}
    dense_crf_param['MaxIterations'] = 2.0
    dense_crf_param['PosW'] = 2.0
    dense_crf_param['PosRStd'] = 5
    dense_crf_param['PosCStd'] = 5
    dense_crf_param['PosZStd'] = 5
    dense_crf_param['BilateralW'] = 3.0
    dense_crf_param['BilateralRStd'] = 5.0
    dense_crf_param['BilateralCStd'] = 5.0
    dense_crf_param['BilateralZStd'] = 5.0
    dense_crf_param['ModalityNum'] = 1
    dense_crf_param['BilateralModsStds'] = (5.0,)
    return densecrf3d(img, prob, dense_crf_param)