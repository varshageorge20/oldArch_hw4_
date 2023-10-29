import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import inspect

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = Detector().to(device)

change = dense_transforms.Compose([
    dense_transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    dense_transforms.RandomHorizontalFlip(0.5),
    dense_transforms.ToTensor(),
    dense_transforms.ToHeatmap()
])

def train(args):
    
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    global_step = 0 #keeps track of the # of training steps
    torch.save(model.state_dict(), (path.join(path.dirname(path.abspath(__file__)), 'det.th')))
    
    if args.rate_ln:
        print('hello')
        #loads the model's parameters from saved checkpoint file called det.th.
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))
        #evaluating expression; constructs dictionary which maps class names of dense_tranforms.py to corresponding class objects
        #change = eval(args.change, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})   
        
        #loading data that may undergo transformations
        dt = load_detection_data('dense_data/train', transform=change)

        #MSE loss will be used during training. reduction=none means it won't be reduced to a scalar value
        ls = torch.nn.MSELoss(reduction='none')

        #binary cross-entropy (BCE) loss function with logits
        ld = torch.nn.BCEWithLogitsLoss(reduction='none')

        #optimizer=Adam, The model's parameters are updated using this optimizer during training
        imp = torch.optim.Adam(model.parameters(), lr=args.rate_gain, weight_decay=1e-4)     
        
        for epoch in range(args.count):
            model.train()
            print("i'm training")
            
            #img = image, dg=ground truth detection maps, sg=other data. these batches are provided by data loader. dt=training data
            for img, dg, sg in dt:
                
                #checking for logging
                if train_logger is not None and global_step % 100 == 0:
                    log(train_logger, img, dg, det, global_step)

                if train_logger is not None:
                    train_logger.add_scalar('ld', dlv, global_step)
                    train_logger.add_scalar('ls', slv, global_step)
                    train_logger.add_scalar('loss', vl, global_step)
                
                #calling to device to connect with gpu at runtime
                img, dg, sg = img.to(device), dg.to(device), sg.to(device)
                
                #This line calculates size_u by finding the maximum value along dimension 
                # 1 of the dg tensor, which is likely a ground truth detection map. 
                # keepdim=True ensures that the result retains the same number of dimensions 
                # as the original tensor. The variable _ is used to ignore the second return 
                # value, possibly because it's not needed in subsequent calculations.
                size_u, _ = dg.max(dim=1, keepdim=True)

                #Here, the neural network model model takes the img tensor as input and returns 
                # two tensors: det and size. These tensors likely contain the model's predictions, 
                # where det might be related to detection results, and size could be related to 
                # size information.
                det, size = model(img)

                #This line computes slv, which appears to be a loss term. It calculates the mean 
                # squared error (MSE) loss between size and sg (other data), multiplied 
                # element-wise by size_u, and then takes the mean of this result. The final 
                # value is divided by the mean of size_u.
                slv = (size_u * ls(size, sg)).mean() / size_u.mean()

                #This line calculates the dp tensor. It computes the sigmoid of a tensor formed 
                # by element-wise multiplication of det and the result of subtracting twice the 
                # dg tensor from 1. This could be part of the loss calculation and is often used 
                # in binary classification tasks.
                dp = torch.sigmoid(det * (1-2*dg))
                
                #z is calculated as the mean of the product of two terms. The first term is the 
                # result of applying the binary cross-entropy with logits loss (ld) to det and dg, 
                # and the second term is the dp tensor calculated in the previous line. This line 
                # seems to compute a specific loss value.
                z=(ld(det, dg)*dp).mean()

                #This line calculates the mean of the dp tensor. o is likely used in the loss 
                # computation and might represent some kind of average prediction confidence.
                o=dp.mean()

                #Here, dlv is computed as the division of z by o. This could be a normalization 
                # or scaling operation for the calculated loss components.
                dlv = z/ o

                #The final loss value vl is computed as a combination of dlv and slv, where slv 
                # is scaled by the value of args.size_weight. This line represents the overall 
                # loss function that the model is trying to minimize during training.
                vl = dlv + slv * args.size_weight

                imp.zero_grad()
                vl.backward()
                imp.step()
                global_step = global_step + 1
            save_model(model)

    #raise NotImplementedError('train')


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    # Put custom arguments here
        #putting all the custom arguments here with their default values
    #parser.add_argument('-k', '--change', default='Compose([ColorJitter(0.8, 0.8, 0.8, 0.2), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    parser.add_argument('-lr','--rate_gain', type=float, default=1e-4)
    parser.add_argument('-u', '--wt', type=float, default=0.02)
    parser.add_argument('-n', '--count', type=int, default=180)
    parser.add_argument('-ln', '--rate_ln', action='store_true')
    parser.add_argument('-sw', '--size_weight', default=0.5)

    args = parser.parse_args()
    train(args)
