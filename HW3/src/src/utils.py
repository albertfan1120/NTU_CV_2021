import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.empty((2*N, 9))
    for i, (u, v) in enumerate(zip(u, v)):
        u_x, u_y = u
        v_x, v_y = v
        A[2*i]   = [u_x, u_y,  1,   0,   0,  0, -u_x*v_x, -u_y*v_x, -v_x]
        A[2*i+1] = [  0,   0,  0, u_x, u_y,  1, -u_x*v_y, -u_y*v_y, -v_y]
        
    # TODO: 2.solve H with A
    U, S, Vh =  np.linalg.svd(A)
    H = Vh[-1, :].reshape((3, 3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x, y = np.meshgrid(range(xmin, xmax), range(ymin, ymax))
    
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    anchor_pts = np.hstack((np.vstack((x.flatten(), y.flatten())).T, np.ones((x.size, 1))))

    if direction == 'b':
        # # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels
        target_pts = anchor_pts.dot(H_inv.T)
        target_pts = np.round((target_pts / target_pts[:, np.newaxis, -1])[:, :-1])

        # TODO: 4.calculate the mask of the transformed coordinate 
        mask = ((0 <= target_pts[:, 0]) * (target_pts[:, 0] < w_src)) * \
               ((0 <= target_pts[:, 1]) * (target_pts[:, 1] < h_src))

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        target_pts = target_pts[mask].astype('int')
        anchor_pts = (anchor_pts[:, :-1][mask]).astype('int')

        # TODO: 6. assign to destination image with proper masking
        dst[anchor_pts[:, 1], anchor_pts[:, 0]] = src[target_pts[:, 1], target_pts[:, 0]]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels
        target_pts = anchor_pts.dot(H.T)
        target_pts = np.round((target_pts / target_pts[:, np.newaxis, -1])[:, :-1])
        
        # TODO: 4.calculate the mask of the transformed coordinate
        mask = ((0 <= target_pts[:, 0]) * (target_pts[:, 0] < w_dst)) * \
               ((0 <= target_pts[:, 1]) * (target_pts[:, 1] < h_dst))

        # TODO: 5.filter the valid coordinates using previous obtained mask
        target_pts = target_pts[mask].astype('int')
        anchor_pts = (anchor_pts[:, :-1][mask]).astype('int')
        
        # TODO: 6. assign to destination image using advanced array indicing
        dst[target_pts[:, 1], target_pts[:, 0]] = src[anchor_pts[:, 1], anchor_pts[:, 0]]

    return dst