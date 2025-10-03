import numpy as np
import pickle


def randset(numsample=6700):
    set = {
        'ecg':np.random.random((numsample,10,35)),
        'gsr':np.random.random((numsample,9,39)),
        'vision':np.random.random((numsample,75,35)),
        'classification_labels':np.random.randint(0,5,numsample),
        'ecg_lengths': np.random.randint(10, 20, numsample).tolist(),
        'gsr_lengths': np.random.randint(5, 10, numsample).tolist(),
        'vision_lengths':np.random.randint(15, 76, numsample).tolist(),
    }
    # 生成随机字符的2D数组
    random_chars = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'),
                                    size=(numsample, 10))
    # 使用numpy.char.add进行字符串连接
    id = []
    for i in range(numsample):
        id.append(''.join(random_chars[i]))
    set['id'] = id
    return set


if __name__ == '__main__':
    np.random.seed(0)
    train_set = randset(numsample=6700)
    valid_set = randset(numsample=500)
    test_set = randset(numsample=1500)
    print(train_set['id'][0:30])
    data_set = {
        'train':train_set,
        'test':test_set,
        'valid':valid_set
    }
    with open('./processed/unaligned.pkl', 'wb') as f:
        pickle.dump(data_set, f)
        print('unaligned.pkl saved')
