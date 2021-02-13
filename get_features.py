import numpy as np
import heapq
from PIL import Image, ImageOps

from harris import get_harris_corners



# creates 8x8 normalized pixel patches for interest pts
def make_feature_descriptors(img_arr, interest_pts):
    size = 8
    spacing = 5
    margin = int((size * spacing) / 2)
    patches = []
    patch_centers = []
    for y,x in interest_pts:
        if margin <= y and y < len(img_arr) - margin and margin <= x and x < len(img_arr[0]) - margin:
            # create 8x8 patch with spacing=5
            y_arr = np.arange(y - margin, y + margin + 1, spacing)
            x_arr = np.arange(x - margin, x + margin + 1, spacing)
            ys, xs = np.meshgrid(x_arr, y_arr)
            patch = np.stack((xs,ys), axis=2)

            coords = np.array([ (i,j) for row in patch for [i,j] in row ])
            pixels = np.array([ img_arr[i][j] for (i,j) in coords])

            # normalize patch
            mean, std = np.average(pixels), np.std(pixels)
            pixels = (pixels - mean) / std

            patches.append(pixels)
            patch_centers.append((y,x))

    return np.array(patches), np.array(patch_centers)



def get_feature_descriptors(img):

    img_gray = ImageOps.grayscale(img)
    img_gray_arr = np.array(img_gray)

    # h values of all pixels and coordinates of harris corners
    h_arr, coords = get_harris_corners(img_gray_arr)

    # harris corners suppressed to 500 corners with anms
    anms_corners = get_anms_corners(h_arr, coords)

    # makes 8x8 normalized pixel patches 
    return make_feature_descriptors(img_gray_arr, anms_corners)




# suppress interest points using adaptive non-maximal suppression
# h_arr: 2d array of each pixel's h values 
# coords: : 2 x N coordinates (ys, xs)
def get_anms_corners(h_arr, coords):

    # h values to interest coordinates
    h_coords = dict()
    for i, j in coords.T:
        h = h_arr[i,j]
        h_coords[h] = (i,j)
    coords = np.array(list(h_coords.values()))  # ?????
    h_vals = np.array(list(h_coords.keys()))    # ????? idk if identically indexed
    c = 0.9
    ch_vals = c * h_vals
    num_pts = 500
    pq = []
    anms_corners = []
    dist_pts = dict()

    # finding ri values (ri: minimum suppression radius)
    # (seen here as min_dist)
    for h in h_vals:
        coords1 = h_coords[h]
        condition_arr = ch_vals > h

        # for pts with stronger corner strength by ratio c,
        # determines min distance from those pts to coords1
        if np.count_nonzero(condition_arr) > 0:
            coords2_arr = coords[condition_arr, :]
            dists = np.sqrt(np.square(coords2_arr - coords1).sum(axis=1))
            min_dist = np.amin(dists)
            heapq.heappush(pq, -min_dist)
            if -min_dist in dist_pts:
                dist_pts[-min_dist].append(coords1)
            else:
                dist_pts[-min_dist] = [ coords1 ]

    # gets num_pts coordinates with largest ri values, using priority queue
    while len(anms_corners) < num_pts and len(dist_pts) > 0:
        ri = heapq.heappop(pq)
        pts = dist_pts[ri]
        if len(anms_corners) + len(pts) < num_pts:
            anms_corners.extend(pts)
        else:
            anms_corners.extend(pts[:num_pts - len(anms_corners)])
    
    return anms_corners





















