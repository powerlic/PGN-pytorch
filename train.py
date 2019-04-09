import argparse
import os
import torch
from data import TrainImageFolder
from torch import nn
from torchvision import transforms
from model import PGN
from PIL import Image
import numpy as np
from scipy import misc


def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))
#last layer x10 undo

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    train_set = TrainImageFolder(args.train_dir)
    data_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle=True,
                                          num_workers = args.num_workers)
    model = nn.DataParallel(PGN()).cuda()
    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    params = list(model.parameters())


    total_step = len(data_loader)
    for epoch in range(134,args.num_epochs):
        lr_=lr_poly(args.learning_rate, epoch*total_step, args.num_epochs*total_step, 0.9)
        optimizer = torch.optim.SGD(params, lr=lr_,momentum=args.momentum,
                                   weight_decay=args.weight_decay)
        for i, (images, parse) in enumerate(data_loader):
            images=images.cuda()
            parse=parse.long().cuda()
            parsing_out1, parsing_out2, edge_out1_final, edge_out_res5, edge_out_res4, edge_out_res3, edge_out2_final = model(images)
            #parsing_out1=model(images)
            loss=criterion(parsing_out1,parse).mean()
            model.zero_grad()

            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch, args.num_epochs, i, total_step, loss.item()))
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))

def predict():
    model = nn.DataParallel(PGN()).cuda()
    #model.load_state_dict(torch.load('models/model-134-2539.ckpt'))
    data_dir='LIP/testing_images'
    dirs = os.listdir(data_dir)
    for file in dirs:
        image=Image.open(data_dir+'/'+file).convert('RGB')
        a,b=image.size[0],image.size[1]
        image = torch.Tensor(np.array(image).astype(np.float32).transpose((2, 0, 1))).unsqueeze(0).cuda()
        #pre_image=Image.fromarray(model(image).cpu().detach().numpy()[0])
        c=Image.fromarray(np.argmax(model(image).cpu().detach().numpy()[0],axis=0).astype(np.uint8))
        c=c.resize((a,b),Image.NEAREST)
        #print(np.array(c))

        # save_image=pre_image.resize((a,b),Image.NEAREST)
        c.save('LIP/test_save/'+file[:-4]+'.png',quality=95,subsampling=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'models/', help = 'path for saving trained models')
    parser.add_argument('--train_dir', type = str, default = 'LIP', help = 'directory for resized images')
    parser.add_argument('--val_dir', type = str, default = 'LIP', help = 'directory for resized images')
    parser.add_argument('--log_step', type = int, default = 1, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 2539, help = 'step size for saving trained models')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')

    # Model parameters
    parser.add_argument('--num_epochs', type = int, default = 200)
    parser.add_argument('--batch_size', type = int, default = 12)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--learning_rate', type = float, default = 1e-5)
    args = parser.parse_args()
    print(args)
    main(args)
    #predict()