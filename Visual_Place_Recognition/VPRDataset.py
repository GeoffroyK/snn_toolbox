from torch.utils.data import Dataset, DataLoader
from utils.image_processing import *

class VPRDataset(Dataset):
    def __init__(self, dataset_path, type) -> None:
        super().__init__()
        self.data = processImageDataset(dataset_path, type, 28, 28, 7, 5, 8)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data.get('x')[index], self.data.get('y')[index]


# Test 
if __name__=="__main__":

    org_data_path = ['/media/geoffroy/T7/VPRSNN/data/{}/'.format('nordland')]  
    train_data_path, test_data_path = get_train_test_datapath(org_data_path)
    dt = VPRDataset(train_data_path, 'train')


    tmp = processImageDataset(train_data_path, type, 28, 28, 7, 5, 8)
    print(tmp.keys())
    import cv2

    for i in range(len(tmp.get('x'))):
        cv2.imshow('test',tmp.get('x')[i])
        cv2.waitKey(0)
    print(i)