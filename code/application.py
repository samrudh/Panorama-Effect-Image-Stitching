# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:53:36 2019

@author: SAMRUDHA KELKAR
"""

"""
Read images into opencv matrix and return dimensions 
"""
def createPanorama(input_img_list):
    images = []
    dims = []
    for index, path in enumerate(input_img_list):
        print (path)
        images.append(cv2.imread(path))
        dims.append(images[index].shape)
    return images, dims
        

"""
Resize all images to same size.
"""
def reSizeImages(images, dims):
    ht , _, _= min(dims,  key = lambda val: val[0])
    _, wt, _= min(dims,  key = lambda val: val[1])
    _, _, ch = min(dims,  key = lambda val: val[2])
    print("resizing to :", ht, wt)
    images= [cv2.resize(x, (ht, wt))  for x in images]
    blank  = np.zeros((wt,ht,ch), np.uint8)
    return images, blank
    
def getMatches(image1, image2, features, ransac_iterations = 150, dist_threshold = 0.7):
    
    points1, des1 = features.detectAndCompute(image1,  None)
    points2, des2 = features.detectAndCompute(image2,  None)
    
    imp = []

    ## Define flann based matcher
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(des1,des2,k=2)
    
    # important features
    imp = []
    for i, (one, two) in enumerate(matches):
        if one.distance < dist_threshold*two.distance:
            imp.append((one.trainIdx, one.queryIdx))
    
    #imp = imp[:min(len(points1), len(points2))]      
    matched1 = np.float32([points1[i].pt for (__, i) in imp])
    matched2 = np.float32([points2[i].pt for (i,__) in imp])
    
    
    print("len of matched features :", matched1.shape[0])

    ## Compute Homography
#    H, s = cv2.findHomography(matched2, matched1 , cv2.RANSAC, 5.0)
    H, err, Ps = helper.ransac_calibrate(matched2, matched1 , matched1.shape[0],"", ransac_iterations)

    # compute corresponding points for common points
    ht, wt, _ = image2.shape
    hmax = max(image1.shape[0], image1.shape[0])
    wmax = max(image1.shape[1], image1.shape[1])
    out = cv2.warpPerspective(image2, H, (4*wmax, 4*hmax))
    
    ht1, wt1, _ = image1.shape

    i1_mask = np.zeros(out.shape, np.uint8)
    i1_mask[:ht1, :wt1] = image1
    
    final_out = np.zeros(out.shape, np.uint8)

    h = final_out.shape[0]
    w = final_out.shape[1]
    ## get maximum of 2 images
    for ch in tqdm(range(3)):
        for x in range(0, h):
            for y in range(0, w):
                final_out[x, y, ch] = max(out[x,y, ch], i1_mask[x,y, ch])


    final_out = getNonZeroImage(final_out)
    
    return final_out, out , des1[0]#, i1_mask


def getNonZeroImage(image):
    
    ## keep nonzero pixels
    out_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = cv2.findNonZero(out_gray)
    pixels = pixels.reshape([pixels.shape[0],2])

    extreme_min_x = min(pixels[:,0])
    extreme_max_x = max(pixels[:,0]) 
    extreme_min_y = min(pixels[:,1])
    extreme_max_y = max(pixels[:,1])
    
    ## clip with extremes
    print("ys :",extreme_min_y, extreme_max_y)
    print("xs :",extreme_min_x, extreme_max_x)
    
    image = image[extreme_min_y:extreme_max_y, extreme_min_x:extreme_max_x,:]
    return image

def showAndWrite(output, temp, path = "out.png"):
    output = getNonZeroImage(output)
        
    ##show
    plt.subplot(141),plt.imshow(im1),plt.title('im1')
    plt.subplot(142),plt.imshow(im2),plt.title('im2')
    plt.subplot(143),plt.imshow(output),plt.title('pano')
    plt.subplot(144),plt.imshow(temp),plt.title('temp')
#
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window",out)

    cv2.imwrite(path,output)
    cv2.imwrite("temp.png",temp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
                    
    return

if __name__ =="__main__":
    image_list = constants.image_list
    images , dims = createPanorama(image_list)

    ## Define feature type
    feature_typye = cv2.xfeatures2d.SIFT_create()

    out = images[0]
    dist_thresh = [0.9, 0.7, 0.5, 0.5, 0.5, 0.5]
    for index in range(len(images) - 1): 
        print('index :', index)
        im2 = images[index+1]
        im1 = out
        out  ,temp, z = getMatches(im1, im2,feature_typye, ransac_iterations = 1000, dist_threshold=dist_thresh[index])
        outshape = out.shape

    out = getNonZeroImage(out)
    showAndWrite(out, temp, path ="taj_f_"+str(index) + ".png")
