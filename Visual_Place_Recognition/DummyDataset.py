import os
import cv2
import numpy as np
from torch.utils.data import Dataset

def dummy_image_processing(img):
    # Open CV is kinda shit ngl
    img = np.array(img)
    img = img / 255.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 1 or 0:
                if img[i][j] > 0.1:
                    img[i][j] = 1
                else:
                    img[i][j] = 0
    return img

class DummyDataset(Dataset):
    def __init__(self, dataset_path) -> None:
        super().__init__()
        self.data = []
        for file_path in sorted(os.listdir(dataset_path)):
            if os.path.isfile(os.path.join(dataset_path, file_path)):
                img = cv2.imread(dataset_path+file_path, cv2.IMREAD_GRAYSCALE) # Read image
                img = dummy_image_processing(img) # Pre process image for torch standard 
                self.data.append(img)
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Test 
if __name__=="__main__":
    dt = DummyDataset("../Dummy_Inputs/dummy_inputs/")
    for i in range(len(dt)):
        cv2.imshow("Image",dt.__getitem__(i))
        cv2.waitKey(0)
