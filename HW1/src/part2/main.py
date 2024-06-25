import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def get_cost(img1, img2):
    error = np.sum(np.abs(img1.astype('int32') - img2.astype('int32')))
    
    return error


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ### TODO ###
    # Report the cost for each filtered image
    file = open(args.setting_path)
    _ = file.readline()
    
    RGB_list = []
    for _ in range(5):
        RGB_list.append(file.readline().strip().split(','))
    
    sigma = file.readline().strip().split(',')
    sigma_s, sigma_r = int(sigma[1]), float(sigma[3])
    
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    #print(img_gray.shape, img_gray.dtype)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
    cost = get_cost(bf_out, jbf_out)
    print("cv2.COLOR_BGR2GRAY cost = {}".format(cost))
    
    
    highest_cost, lowest_cost = 0, 2000000
    highest_jbf, lowest_jbf = None, None
    highest_gray, lowest_gray = None, None
    for R, G, B in RGB_list:
        img_gray = float(R)*img_rgb[:,:,0] + float(G)*img_rgb[:,:,1] + float(B)*img_rgb[:,:,2]
        img_gray = img_gray.astype(np.uint8)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
        cost = get_cost(bf_out, jbf_out)
        print("R*{}+G*{}+B*{} cost = {}".format(R, G, B, cost))
        
        if cost > highest_cost:
            highest_jbf, highest_gray = jbf_out, img_gray
            highest_cost = cost
        if cost < lowest_cost:
            lowest_jbf, lowest_gray = jbf_out, img_gray
            lowest_cost = cost
    
    cv2.imwrite("highest_jbf.png", cv2.cvtColor(highest_jbf, cv2.COLOR_RGB2BGR))
    cv2.imwrite("highest_gray.png", highest_gray)
    cv2.imwrite("lowest_jbf.png", cv2.cvtColor(lowest_jbf, cv2.COLOR_RGB2BGR))
    cv2.imwrite("lowest_gray.png", lowest_gray)
    


if __name__ == '__main__':
    main()






# import numpy as np
# import cv2
# import argparse
# import os
# from JBF import Joint_bilateral_filter


# def main():
#     parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
#     parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
#     parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
#     args = parser.parse_args()

#     img = cv2.imread(args.image_path)
#     img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # print(img_rgb)

#     ### TODO ###
#     f = open(args.setting_path, 'r')
#     data = []
#     for line in f:
#         line = line.strip()
#         line = line.split(',')
#         line = [float(item) for item in line]
#         data.append(line)

#     RGB = np.array(data[0:-2])
#     sigma_s = int(data[-2][0])
#     sigma_r = data[-1][0]

#     for i in range(6):
#         if i:
#             img_gray = RGB[i-1, 0] * img_rgb[:, :, 0] + RGB[i-1, 1] * img_rgb[:, :, 1] + RGB[i-1, 2] * img_rgb[:, :, 2]
#         JBF = Joint_bilateral_filter(sigma_s, sigma_r)
#         bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
#         jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
#         cost = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
#         print(cost)

#     f.close()
    


# if __name__ == '__main__':
#     main()