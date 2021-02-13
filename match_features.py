import numpy as np
from PIL import Image
import random, math

from transform_images import compute_homography


# computes sum of squared differences between two arrays of img descriptors
# descriptors: flattened BW pixel arrays (from 8x8 patches spaced by 5)
def compute_ssd(descriptors1, descriptors2):
    return np.sum(np.square(descriptors1 - descriptors2), axis=1)



# returns initial feature matches between 
# params: descriptors from each image with their respective coordinates
# descriptors: flattened BW pixel arrays (from 8x8 patches spaced by 5)
def get_initial_matches(descriptors1, descriptors2, coords1, coords2):

    # (y,x) in img1 to (y,x) in img2
    matches = dict()
    NN_ratio_threshold = 0.06

    for i in range(len(descriptors1)):
        descriptor, coords = descriptors1[i], coords1[i]
        i, j = coords

        # errors for first and second nearest neighbors (using ssd)
        ssd_arr = compute_ssd(descriptor, descriptors2)
        NN1 = np.amin(ssd_arr)
        NN1_index = np.where(ssd_arr == NN1)[0][0]
        NN2 = np.partition(ssd_arr,-2)[1]
        
        # only keep match if ratio between first and second nearest
        # neighbors less than initialized threshold
        if NN1 / NN2 < NN_ratio_threshold:
            matches[(i,j)] = coords2[NN1_index]
    
    return matches



# iteratively computes homographies for random subsets of 
# feature matches to find homography with largest set of inliers
# matches: (y,x) in img1 to (y,x) in img2
def do_ransac(matches):

    img1_allpts = list(matches.keys())
    max_inliers = dict()
    max_H = []
    dist_threshold = 5.0

    for i in range(5000):
        # random sample of four feature pairs and corresponding homography
        img1_pts = random.sample(img1_allpts, 4)
        img2_pts = [ matches[pt] for pt in img1_pts ]
        H = compute_homography(img1_pts, img2_pts, 4)

        # inliers based on distance threshold 
        inliers = dict()
        for (y1,x1), (y2,x2) in matches.items():
            p = np.array([x1, y1, 1])
            p1 = H @ p
            p1 = p1 / p1[2]
            x1_Hp, y1_Hp = p1[0], p1[1]
            dist = math.sqrt((x1_Hp - x2)**2 + (y1_Hp - y2)**2)

            # only add as inlier if warped pixel location
            # < 5 pixels from correct location in img2
            if dist < dist_threshold:
                inliers[(y1,x1)] = (y2,x2)

        # update inliers set and homography with max-sized inliers set
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            max_H = H
    
    return max_H, max_inliers





































