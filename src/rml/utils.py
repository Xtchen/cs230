import os
import shutil


def split_data():
    path = '/home/ec2-user/data/cropped/'
    des_path = '/home/ec2-user/data/cropped{}/'
    count = 0

    for dir in os.listdir(path):
        count += 1
        target = count % 5

        shutil.move(os.path.join(path, dir), os.path.join(path, des_path.join(target)))


if __name__ == '__main__':
    split_data()
