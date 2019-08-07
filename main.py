import argparse
import collector
import load_img
import re
import numpy as np
import fingerprint
import random
import torch
import torch.nn as nn
from model import IdModel
import os

CAM_ID = {'iP6': 9, 'iP4s': 8, 'GalaxyS4': 7, 'GalaxyN3': 6, 'MotoNex6': 5,
          'MotoMax': 4, 'HTC-1-M7': 3, 'MotoX': 2, 'Nex7': 1, 'LG5x': 0}


# From: http://blog.topspeedsnail.com/archives/1469
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def eliminate_nan_inf(arr):
    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 0
    return arr


def extract_camera_name(path: str) -> str:
    name = re.match(r'/?(\S*/)*(\S*.\S*)', path).group(2)
    camera_name = re.match(r'\((\S*)\)\S*$', name).group(1)
    return camera_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-f', '--format', default='jpg')
    parser.add_argument('-d', '--dir', default='data/')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-g', '--group', default=8)
    parser.add_argument('-b', '--batch', default=16)
    parser.add_argument('-e', '--epoch', default=100)
    args = parser.parse_args()
    properties = {'input_dir': args.dir, 'format': args.format, 'group': args.group,
                  'batch_size': args.batch, 'epoch': args.epoch}

    # Specify device
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'

    # Load imgaes
    print('Loading images...')
    imgs = load_img.load_img_from_dir(properties['input_dir'], properties['format'])
    img_collector = collector.ImageCollector()
    for filename in imgs:
        camera = extract_camera_name(filename)
        img_collector[camera].append(filename)
    cameras = list(img_collector.imgs.keys())
    print('Images loaded!')

    # Get fingerprint
    print('Generating fingerprints...')
    fps = []
    labels = []
    count = 0
    for camera in cameras:
        imgs = img_collector.imgs[camera]
        for group in chunks(imgs, properties['group']):
            fp = fingerprint.get_fingerprint(imgs, camera)
            fp = eliminate_nan_inf(fp)
            fps.append(fp)
            labels.append(camera)
        count += 1
        print(f"==> {count/len(cameras)}%")
    print('Finished!')

    # Train NN
    print("Training NN...")
    shuffled_inds = list(range(len(fps)))
    print(np.array(fps).shape)
    fps = torch.Tensor(np.transpose(np.array(fps), (0, 3, 1, 2))).to(device)  # Convert to arrays
    labels = np.array(labels)  # Convert to arrays
    random.shuffle(shuffled_inds)

    model = IdModel(cam_num=len(cameras)).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.002, betas=(0.5, 0.999))
    l2_loss_func = nn.MSELoss(reduce=True, size_average=False).to(device)

    for epoch in range(properties['epoch']):
        ind_generator = chunks(shuffled_inds, properties['batch_size'])
        for inds in ind_generator:
            train_fps = fps[inds] / len(inds)
            train_labels = labels[inds]

            real_label = torch.zeros(size=(len(inds), len(cameras))).to(device)
            result = model.forward(train_fps).to(device)
            for i in range(len(inds)):
                real_label[i][CAM_ID[train_labels[i]]] = 1
            loss = l2_loss_func(real_label, result)
            loss.backward()
            optimizer.step()

        print(f'====> epoch: {epoch}/{properties["epoch"]}, loss on train set: {loss}')

        if epoch % 5 == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")

            model_out_path = f"checkpoint/Model_epoch_{epoch}.pth"
            torch.save(model, model_out_path)
            print("Checkpoint saved to checkpoint/")
    print("Finished!")
