import numpy as np
import cv2 as cv
import os

'''
Pre-processing of the input image following the strategy described in : https://github.com/QVPR/VPRSNN
as we want to use this method for VPR.
'''
def get_patches2D(image, patch_size):

    if patch_size[0] % 2 == 0: 
        nrows = image.shape[0] - patch_size[0] + 2
        ncols = image.shape[1] - patch_size[1] + 2
    else:
        nrows = image.shape[0] - patch_size[0] + 1
        ncols = image.shape[1] - patch_size[1] + 1
    return np.lib.stride_tricks.as_strided(image, patch_size + (nrows, ncols), image.strides + image.strides).reshape(patch_size[0]*patch_size[1],-1)

def patch_normalise_pad(image, patch_size):

    patch_size = (patch_size, patch_size)
    patch_half_size = [int((p-1)/2) for p in patch_size ]

    image_pad = np.pad(np.float64(image), patch_half_size, 'constant', constant_values=np.nan)

    nrows = image.shape[0]
    ncols = image.shape[1]
    patches = get_patches2D(image_pad, patch_size)
    mus = np.nanmean(patches, 0)
    stds = np.nanstd(patches, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = (image - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)

    out[np.isnan(out)] = 0.0
    out[out < -1.0] = -1.0
    out[out > 1.0] = 1.0
    return out

def loadImg(imgPath):
    # Read and convert image from BGR to RGB
    img = cv.imread(imgPath)[:,:,::-1]
    return img

def processSingleImage(imgPath, imWidth, imHeight, numPatches):
    img = loadImg(imgPath)
    img = cv.resize(img, (imWidth, imHeight))
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    img_norm = patch_normalise_pad(img, numPatches)

    # Scale elements to be between 0 and 255
    img = np.uint8(255. * (1 + img_norm) / 2.0)
    # Scale elements to be between 0 and 1 for Pytorch implementation
    img = img / 255.

    return img



def processImage(img, imWidth, imHeight, numPatches):

    img = cv.resize(img, (imWidth, imHeight))
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    img_norm = patch_normalise_pad(img, numPatches)

    # Scale elements to be between 0 and 255
    img = np.uint8(255. * (1 + img_norm) / 2.0)
    # Scale elements to be between 0 and 1 for Pytorch implementation
    img = img / 255.

    return img

def get_filtered_name_paths(filtered_names_path):

    assert os.path.isfile(filtered_names_path), "The file path {} is not a valid".format(filtered_names_path)

    with open(filtered_names_path) as f:
        content = f.read().splitlines()
        
    filtered_index = [int(''.join(x for x in char if x.isdigit())) for char in content] 

    return filtered_index

def get_train_test_datapath(org_data_path):

    if 'nordland' in org_data_path[0]:
        train_data_path = [org_data_path[0] + '/spring/', org_data_path[0] + '/fall/']  
        test_data_path =  [org_data_path[0] + '/summer/']

    elif '/data/ORC/' in org_data_path:
        train_data_path = [org_data_path[0] + 'Sun/', org_data_path[0] + 'Rain/']
        test_data_path =  [org_data_path[0] + 'Dusk/']
    
    elif '/data/SPEDTEST/' in org_data_path:
        train_data_path = [org_data_path[0] + "ref/"]
        test_data_path = [org_data_path[0] + "query/"]

    elif '/data/Synthia-NightToFall/' in org_data_path:
        train_data_path = [org_data_path[0] + "ref_modified/"]
        test_data_path = [org_data_path[0] + "query_modified/"]

    elif '/data/St-Lucia/' in org_data_path:
        train_data_path = [org_data_path[0] + "ref/"]
        test_data_path = [org_data_path[0] + "query/"]

    return train_data_path, test_data_path

def processImageDataset(path, type, imWidth, imHeight, num_patches=7, num_labels=100, skip=0, offset_after_skip=0):

    print("Computing features for image path: {} ...\n".format(path))

    imgLists = []
    for p in path: 
        imgList = np.sort(os.listdir(p))
        imgList = [os.path.join(p,f) for f in imgList]    
        imgLists.append(imgList)

    if "nordland" in path[0]:
        dirPath = os.path.abspath(os.getcwd())
        filtered_names_path = "{}/dataset_imagenames/nordland_imageNames.txt".format(dirPath)
        filtered_index = get_filtered_name_paths(filtered_names_path)

    frames = []
    paths_used = [] 
    labels = []

    for imgList in imgLists: 

        nordland = True if "nordland" in imgList[0] else False 

        j = 0
        ii = 0  # keep count of number of images
        kk = 0  # keep track of image indices, considering offset after skip

        for i, imPath in enumerate(imgList):
            
            if (ii == num_labels):
                break 

            if ".jpg" not in imPath and ".png" not in imPath:
                continue 
            
            if nordland: 
                if (i not in filtered_index):
                    continue

                if j % skip != 0:
                    j += 1
                    continue
                j += 1
            
            if not nordland and (skip != 0 and i % skip != 0):  
                continue
            
            if (offset_after_skip > 0 and kk < offset_after_skip):
                kk += 1
                continue
            
            frame = loadImg(imPath)

            frame = processImage(frame, imWidth, imHeight, num_patches)  
            frames.append(frame)

            labels.append(kk)

            if nordland: 
                idx = np.where(np.array(filtered_index) == i)[0][0]
                path_id = filtered_index[idx]
            else:
                path_id = i

            paths_used.append(path_id)

            ii += 1
            kk += 1 
    
    print("indices used in {}:\n{}".format(type, paths_used))
    print("labels used in {}:\n{}".format(type, labels))

    y = np.array([ [labels[i]] for i in range(len(labels)) ])
    data = {'x': np.array(frames), 'y': y, 'rows': imWidth, 'cols': imHeight}

    return data