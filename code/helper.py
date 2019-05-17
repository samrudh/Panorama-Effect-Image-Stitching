import cv2
import numpy as np
import random
from tqdm import tqdm

### DLT
def create_matrix_M(real_points , image_points, total_points):
    M = []  #np.empty([2*total_points, 12])
    for index in range(total_points):

        X,Y = real_points[index]
        x, y= image_points[index]

        # print("XYZ are :", X, Y, Z)
        # print("xyz are :", x, y, z)

        M.append([X,Y, 1, 0, 0,0,  -(x*X), -(x*Y), -x])
        M.append([ 0, 0, 0, X, Y, 1,  -(y * X), -(y * Y), -y])


    M = np.asarray(M)
    return M

def dlt_calibrate(real_world_points, image_points, total_points):
    a = create_matrix_M(real_world_points, image_points, total_points)
    U, S, V = np.linalg.svd(a)

    P_vec = V[-1,:]
    ## Normalize
    P_vec = P_vec / V[-1,-1]

    # print("p vec after norm ", P_vec)
    P = np.reshape(P_vec, [3,3])
    # print("P is ", P)
    return P

def calculate_image_point(p_matrix, world_point, image_path, write = False, show= False, title="image"):
    x = np.dot(p_matrix, world_point.T)

    u, v, w = list(x)
    xx =  int(u/w)
    yy = int(v/w)

    return  xx, yy

### RANSAC
def ransac_calibrate(real_points , image_points, total_points, image_path, iterations):
    index_list = list(range(total_points))
    iterations = min(total_points - 1, iterations)
    errors = list(np.zeros(iterations))
    combinations = []
    p_estimations=[]
    
    for i in range(iterations):
        selected = random.sample(index_list,4)
        combinations.append(selected)
#        print("selected",selected)
        real_selected =[]
        image_selected =[]

        for x in selected:
            real_selected.append(real_points[x])
            image_selected.append(image_points[x])


        p_estimated = dlt_calibrate(real_selected, image_selected, 4)


        not_selected = list(set(index_list) - set(selected))
        error = 0
        for num in tqdm(not_selected):

            # get points from the estimation
            test_point = list(real_points[num])
            test_point = [int(x) for x in test_point]
            test_point = test_point + [1]

            try:
                xest, yest = calculate_image_point(p_estimated, np.array(test_point), image_path)
            except ValueError:
                
                continue
            
            error = error + np.square(abs(np.array(image_points[num])-np.asarray([xest,yest])))
#            print("estimated  :",np.array([xest, yest])  )
#            print("actual :",image_points[0])
#            print("error :",error)
        errors.append(np.mean(error))
        p_estimations.append(p_estimated)

    p_final = p_estimations[errors.index(min(errors))]
    return p_final ,errors, p_estimations


