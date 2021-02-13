import numpy as np
import sys
from PIL import Image, ImageDraw

from get_features import get_feature_descriptors
from match_features import get_initial_matches, do_ransac
from transform_images import make_mosaic


def stitch_images():
    # images, sizes, arrays for both images
    img1_name, img2_name = sys.argv[1], sys.argv[2]
    img1, img2 = Image.open(img1_name), Image.open(img2_name)
    width1, height1 = img1.size
    width2, height2 = img2.size
    img1_arr, img2_arr = np.array(img1), np.array(img2)

    # image to draw matching features
    features_img = Image.new('RGB', (width1 + width2, max(height1, height2)))
    features_img.paste(img1, (0,0))
    features_img.paste(img2, (width2,0))
    
    # feature descriptors and center coordinates
    descriptors1, coords1 = get_feature_descriptors(img1)
    descriptors2, coords2 = get_feature_descriptors(img2)

    # feature matches after ransac
    initial_matches = get_initial_matches(descriptors1, descriptors2, coords1, coords2)
    H, inlier_matches = do_ransac(initial_matches)

    # marks matching features and connects corresponding features
    create_matches_img(features_img, inlier_matches, width1)

    # creates and displays final mosaic with both pictures
    make_mosaic(img1_arr, img2_arr, initial_matches, H)




# marks matching features and connects corresponding features between two images
def create_matches_img(img, matches, x_offset):

    draw = ImageDraw.Draw(img)
    for (i1,j1), (i2,j2) in matches.items():
        draw.rectangle([j1-20, i1-20, j1+20, i1+20], outline="blue", width=3)
        draw.rectangle([j2-20 + x_offset, i2-20, j2+20 + x_offset, i2+20], outline="blue", width=3)
        draw.line([(j1,i1), (j2 + x_offset,i2)], fill="blue", width=3)
    img.show()
    img.save("feature_matches.png")
    















stitch_images()

















