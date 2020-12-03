
import glob
from collections import defaultdict
import os
import numpy as np
import random

from numpy.testing import assert_array_almost_equal

from PIL import Image
from skimage import io

from utils.AID_LB_Dependent_P import P_05 as AID_P_05, P_01 as AID_P_01, P_03 as AID_P_03, P_07 as AID_P_07
from utils.NWPU45_Dependent_P import P_05 as NWPU_P_05, P_01 as NWPU_P_01, P_03 as NWPU_P_03, P_07 as NWPU_P_07

def default_loader(path):
    return Image.open(path).convert('RGB')


def eurosat_loader(path):
    return io.imread(path)


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return np.eye(n_classes)[x]


class DataGeneratorSplitting:
    """
    generate train and val dataset based on the following data structure:
    Data structure:
    └── SeaLake
        ├── SeaLake_1000.jpg
        ├── SeaLake_1001.jpg
        ├── SeaLake_1002.jpg
        ├── SeaLake_1003.jpg
        ├── SeaLake_1004.jpg
        ├── SeaLake_1005.jpg
        ├── SeaLake_1006.jpg
        ├── SeaLake_1007.jpg
        ├── SeaLake_1008.jpg
        ├── SeaLake_1009.jpg
        ├── SeaLake_100.jpg
        ├── SeaLake_1010.jpg
        ├── SeaLake_1011.jpg
    """

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()
        
        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.CreateIdx2fileDict()


    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)

            # train_subdirImgPth = subdirImgPth[:int(0.2*len(subdirImgPth))]
            # val_subdirImgPth = subdirImgPth[int(0.2*len(subdirImgPth)):int(0.3*len(subdirImgPth))]
            # test_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):]
            
            train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
            val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1
        
        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        elif self.phase == 'val':
            idx = self.valDataIndex[index]
        else:
            idx = self.testDataIndex[index]
        
        return self.__data_generation(idx)

            
    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]

        if self.phase == 'train':
            imgPth, imgLb = self.train_idx2fileDict[idx]
        elif self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        if self.dataset in ['eurosat', 'UCMerced']:
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)
        oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec.astype(np.float32)}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        # return {'img': img, 'label': imgLb}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)

def multiclass_noisify(y, P, random_state):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def noisify_multiclass_symmetric(y_train, noise_rate, random_state=42, nb_classes=10):
    
    """mistakes:
        flip in the symmetric way
    """

    P = np.ones((nb_classes, nb_classes))
    n = noise_rate
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # P[0, 0] = 1. - n
        # for i in range(1, nb_classes-1):
        #     P[i, i] = 1. - n
        # P[nb_classes-1, nb_classes-1] = 1. - n
        for i in range(nb_classes):
            P[i, i] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                            random_state=random_state)

        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0

        print(f'Actual noise ratio {actual_noise}')

        y_train = y_train_noisy
    
    return y_train, actual_noise

def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise
    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0

        print(f'Actual noise ratio {actual_noise}')
        y_train = y_train_noisy
    
    return y_train, actual_noise

def noisify_pairflip_P(y_train, P, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    # P = np.eye(nb_classes)
    # n = noise
    # if n > 0.0:
    #     # 0 -> 1
    #     P[0, 0], P[0, 1] = 1. - n, n
    #     for i in range(1, nb_classes-1):
    #         P[i, i], P[i, i + 1] = 1. - n, n
    #     P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

    y_train_noisy = multiclass_noisify(y_train, P=P,
                                        random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0

    print(f'Actual noise ratio {actual_noise}')
    y_train = y_train_noisy
    
    return y_train, actual_noise



class DataGeneratorNoisy:

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train', noise_rate=0, noise_type='symmetric'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        
        self.noise_rate = noise_rate

        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.noise_type = noise_type

        self.CreateIdx2fileDict()
        self.Noisy_train_y()

    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)

            train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
            val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1
        
        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

    def Noisy_train_y(self):

        y_train = []
        for _, label in self.train_idx2fileDict.values():
            y_train.append(label)
        y_train = np.asarray(y_train)

        if self.noise_type == 'symmetric':
            train_noisy_labels, self.actual_noise_rate = noisify_multiclass_symmetric(y_train, self.noise_rate, random_state=42, nb_classes=len(self.sceneList))
        if self.noise_type == 'pairflip':
            # train_noisy_labels, self.actual_noise_rate = noisify_pairflip(y_train, self.noise_rate, random_state=42, nb_classes=len(self.sceneList))
            if self.dataset == 'AID':
                if self.noise_rate == 0.1:
                    AID_P = AID_P_01
                elif self.noise_rate == 0.3:
                    AID_P = AID_P_03
                elif self.noise_rate == 0.5:
                    AID_P = AID_P_05
                elif self.noise_rate == 0.7:
                    AID_P = AID_P_07
                train_noisy_labels, self.actual_noise_rate = noisify_pairflip_P(y_train, AID_P, random_state=42, nb_classes=len(self.sceneList))
            elif self.dataset == 'NWPU-RESISC45':
                if self.noise_rate == 0.1:
                    NWPU_P = NWPU_P_01
                elif self.noise_rate == 0.3:
                    NWPU_P = NWPU_P_03
                elif self.noise_rate == 0.5:
                    NWPU_P = NWPU_P_05
                elif self.noise_rate == 0.7:
                    NWPU_P = NWPU_P_07
                train_noisy_labels, self.actual_noise_rate = noisify_pairflip_P(y_train, NWPU_P, random_state=42, nb_classes=len(self.sceneList))
            
        for idx, (imgPth, _) in enumerate(self.train_idx2fileDict.values()):
            self.train_idx2fileDict[idx] = (imgPth, train_noisy_labels[idx])

    def __getitem__(self, index):
        
        return self.__data_generation(index)
    
    def __data_generation(self, idx):
        
        if self.phase == 'train':
            imgPth, imgLb = self.train_idx2fileDict[idx]
        elif self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        if self.dataset in ['eurosat', 'UCMerced']:
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)
        oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        # return {'img': img, 'label': imgLb}

    def __len__(self):
        
        if self.phase == 'train':
            return self.train_numImgs
        elif self.phase == 'val':
            return self.val_numImgs
        else:
            return self.test_numImgs


class DataGeneratorNoisyTrip:

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train', noise_rate=0, noise_type='symmetric'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        
        self.noise_rate = noise_rate

        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.train_label2idx = defaultdict()
        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.noise_type = noise_type

        self.CreateIdx2fileDict()
        self.Noisy_train_y()

    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)

            train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
            val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1
        
        self.labels_list = list(range(len(self.sceneList)))

        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

    def Noisy_train_y(self):

        y_train = []
        for _, label in self.train_idx2fileDict.values():
            y_train.append(label)
        y_train = np.asarray(y_train)

        if self.noise_type == 'symmetric':
            train_noisy_labels, self.actual_noise_rate = noisify_multiclass_symmetric(y_train, self.noise_rate, random_state=42, nb_classes=len(self.sceneList))
        if self.noise_type == 'pairflip':
            if self.dataset == 'AID':
                if self.noise_rate == 0.1:
                    AID_P = AID_P_01
                elif self.noise_rate == 0.3:
                    AID_P = AID_P_03
                elif self.noise_rate == 0.5:
                    AID_P = AID_P_05
                elif self.noise_rate == 0.7:
                    AID_P = AID_P_07
                train_noisy_labels, self.actual_noise_rate = noisify_pairflip_P(y_train, AID_P, random_state=42, nb_classes=len(self.sceneList))
            elif self.dataset == 'NWPU-RESISC45':
                if self.noise_rate == 0.1:
                    NWPU_P = NWPU_P_01
                elif self.noise_rate == 0.3:
                    NWPU_P = NWPU_P_03
                elif self.noise_rate == 0.5:
                    NWPU_P = NWPU_P_05
                elif self.noise_rate == 0.7:
                    NWPU_P = NWPU_P_07
                train_noisy_labels, self.actual_noise_rate = noisify_pairflip_P(y_train, NWPU_P, random_state=42, nb_classes=len(self.sceneList))
            
        for idx, (imgPth, _) in enumerate(self.train_idx2fileDict.values()):
            self.train_idx2fileDict[idx] = (imgPth, train_noisy_labels[idx])

            if train_noisy_labels[idx] in self.train_label2idx:
                self.train_label2idx[train_noisy_labels[idx]].append(idx)
            else:
                self.train_label2idx[train_noisy_labels[idx]] = [idx]
        
    def __getitem__(self, index):

        if self.phase == 'train':
            _, imgLb = self.train_idx2fileDict[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.train_label2idx[imgLb])
            
            negative_label = np.random.choice(list(set(self.labels_list) - set([imgLb])))
            negative_index = np.random.choice(self.train_label2idx[negative_label])
            return self.__data_generation_triplet(index, positive_index, negative_index)
        else:
            return self.__data_generation(index)

    def __data_generation(self, idx):

        if self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]
        
        if self.dataset in ['eurosat', 'UCMerced']:
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        return {'img': img, 'label': imgLb}
    
    def __data_generation_triplet(self, idx, pos_idx, neg_idx):

        anc_imgPth, anc_label = self.train_idx2fileDict[idx]
        pos_imgPth, _ = self.train_idx2fileDict[pos_idx]
        neg_imgPth, _ = self.train_idx2fileDict[neg_idx]

        anc_img = default_loader(anc_imgPth)
        pos_img = default_loader(pos_imgPth)
        neg_img = default_loader(neg_imgPth)

        if self.imgTransform is not None:
            anc_img = self.imgTransform(anc_img)
            pos_img = self.imgTransform(pos_img)
            neg_img = self.imgTransform(neg_img)
        
        return {'anc':anc_img, 'pos':pos_img, 'neg':neg_img, 'anc_label':anc_label}

    def __len__(self):
        
        if self.phase == 'train':
            return self.train_numImgs
        elif self.phase == 'val':
            return self.val_numImgs
        else:
            return self.test_numImgs




if __name__ == "__main__":


    train_gen = DataGeneratorNoisy(data='/home/kang/Documents/Data', dataset='AID', noise_rate=0.5, noise_type='pairflip')
    


