import numpy as np
import cv2
import random
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        matches = bf.match(queryDescriptors = des2, trainDescriptors = des1)
        matches = sorted(matches, key = lambda x:x.distance)
        points1 = np.array([kp1[m.trainIdx].pt for m in matches])
        points2 = np.array([kp2[m.queryIdx].pt for m in matches])

        # TODO: 2. apply RANSAC to choose best H
        best_H = Ransac(points1, points2)

        # TODO: 3. chain the homographies
        cur_best_H = np.dot(last_best_H, best_H)
        last_best_H = np.copy(cur_best_H)
        
        # TODO: 4. apply warping
        h, w, _ = im1.shape
        out = warping(im2, dst[:, :w*(idx+2)], cur_best_H, 0, h, 0, w*(idx+2))

    return out


def transform(anchor_pts, H):
    N = anchor_pts.shape[0]
    
    anchor = np.concatenate((anchor_pts, np.ones((N, 1))), axis = 1) 
    trans = anchor.dot(H.T)
    trans /= (np.expand_dims(trans[:, -1], axis = -1) + 1e-12)
    
    return trans[:, :-1]


def Ransac(points1, points2, n_iters = 5000, threshold = 10):
    # pts2  H->  pts1
    N = points1.shape[0]
    n_samples = 4
    index = np.arange(N)

    max_inliers = np.zeros(N, dtype = bool)
    n_inliers = 0
    for _ in range(n_iters):
        np.random.shuffle(index)
        samples_index = index[:n_samples]
        randpts_1 = points1[samples_index]
        randpts_2 = points2[samples_index]

        H = solve_homography(randpts_2, randpts_1)

        trans_pts = transform(points2, H)
        error = ((trans_pts - points1) ** 2).sum(axis = 1) ** 0.5
        inliers = error < threshold
        
        if np.sum(inliers) > n_inliers: 
            n_inliers = np.sum(inliers)
            max_inliers = inliers
            
    return solve_homography(points2[max_inliers], points1[max_inliers])


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    np.random.seed(320)
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)