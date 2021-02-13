import numpy as np
from PIL import Image



# computes homography matrix to map img2 onto img1
# (to warp img2 into img1 perspective)
# takes in points as (y,x)
def compute_homography(img1_pts, img2_pts, num_correspondence_pts):

    A = np.zeros((num_correspondence_pts * 2, 9))

    # fill A with correspondence pt pairs
    for i in range(num_correspondence_pts):
        x, y, u, v = img1_pts[i][1], img1_pts[i][0], img2_pts[i][1], img2_pts[i][0]
        A[2*i,:] = np.array([x, y, 1.0, 
                            0.0, 0.0, 0.0,
                            -u * x, -u * y, -u])
        A[2*i + 1,:] = np.array([0.0, 0.0, 0.0,
                                x, y, 1.0,
                                -v * x, -v * y, -v])

    # setting h to solution of Ah = 0 
    # (aka last column of V in SVD)
    V = np.linalg.svd(A)[2]
    h = np.reshape(V[8], (3,3))
    h = h / V[-1,-1]
    return h


# to contain width, height, and horizonal/vertical offset 
# of the two images being combined
def get_mosaic_info(img1_arr, img2_arr, img1_pts, img2_pts):
    info = dict()
    info["x_offset"] = int(np.average(img1_pts[:,1:]-img2_pts[:,1:]))
    info["y_offset"] = int(np.average(img1_pts[:,:1]-img2_pts[:,:1]))
    info["width"] = len(img1_arr[0]) + abs(info["x_offset"])
    info["height"] = len(img1_arr) + abs(info["y_offset"])
    return info


# inverse warps img1 to img2's perspective using homography
def warp_image(img1_arr, H, mosaic_info):
    
    x_offset = mosaic_info["x_offset"]
    y_offset = mosaic_info["y_offset"]
    rows = mosaic_info["height"] 
    cols = mosaic_info["width"] 
    warped_img = np.full((rows, cols, 3), 0, dtype=np.uint8)

    for i in range(-y_offset, rows):
        for j in range(-x_offset, cols):
            j -= x_offset
            p1 = np.array([j,i,1])
            p = np.linalg.inv(H) @ p1
            px, py = int(p[0] / p[2]), int(p[1] / p[2])
            wy, wx = i + y_offset, j + x_offset
            if (0 <= px and px < len(img1_arr[0]) and 0 <= py and py < len(img1_arr)) and \
                (0 <= wx and wx < cols and 0 <= wy and wy < rows):
                warped_img[wy,wx] = img1_arr[py,px]
    
    return warped_img


# applies alpha blending to pix1 and pix2
def blend(alpha, pix1, pix2):
    return pix1 * alpha + (1 - alpha) * pix2


# combines img2 onto warped img1's canvas
def make_mosaic(img1_arr, img2_arr, matches, H):

    img1_pts = np.array([ [i,j] for (i,j) in matches.keys() ])
    img2_pts = np.array([ [ matches[pt][0],matches[pt][1] ] for pt in list(matches.keys()) ])

    mosaic_info = get_mosaic_info(img1_arr, img2_arr, img1_pts, img2_pts)
    warped_img1 = warp_image(img1_arr, H, mosaic_info)
    x_offset = mosaic_info["x_offset"]
    y_offset = mosaic_info["y_offset"]
    imgw_width, imgw_height = len(warped_img1[0]), len(warped_img1)
    img2_width, img2_height = len(img2_arr[0]), len(img2_arr)
    alpha = 1.0

    for j in range(min(imgw_width, img2_width)):
        for i in range(min(imgw_height, img2_height)):
            x1, y1 = j + abs(x_offset), i + abs(y_offset)
            pix1 = warped_img1[y1, x1]
            pix2 = img2_arr[i, j]

            # replaces all black pixels with img2 pixel
            if np.all((pix1 == 0)):
                warped_img1[y1, x1] = pix2
            # if img1 pixel there, blends the two
            else:
                warped_img1[y1, x1] = blend(alpha, pix1, pix2)
        if j >= x_offset or i >= y_offset:
            alpha -= 0.0005

    mosaic = Image.fromarray(warped_img1, 'RGB')
    mosaic.show()
    mosaic.save("mosaic.png")















