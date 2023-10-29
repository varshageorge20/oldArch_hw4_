import torch
import torch.nn.functional as F


#2d heatmap, 
# max_pool_ks=size of max pooling window to determine local maxima, 
# min_score: min score for a point to be considered a peak, 
# max_det: max # of peaks to be returned
def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """

    #using the max_pool2d function as given in hint
    #ret is the pooled heatmap. 
    # heatmap[None, None] adds 2 dimensions to the heatmap, making it a 4d tensor
    #the padding: heatmap is padded symmetrically to ensure output matches input size
    #stride is 1 to show there's no overlap in the pooling operation
    ret = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]

    #diff calculates the difference between original heatmap and modified version of heatmap
    #ret > heatmap: creates binary mask that's 'True' where pooled heatmap values are greater than original heatmap
    #.float() is to convert binary mask to a float tensor
    #binary mask is then multiplied by 1e4 to set valyes to a very large (-) number where the pooled heatmap is greater than the original
    diff = heatmap - (ret > heatmap).float() * 1e4

    #k = # of elements in the tensor that's created; counts # of elements that were set to a large (-) number
    k=(heatmap - (ret > heatmap).float() * 1e4).numel()

    #
    if max_det > k:
        max_det = k
    score, va = torch.topk(diff.view(-1), max_det)
    #return [(float(p), int(m) % heatmap.size(1), int(m) // heatmap.size(1))
        #for p, m in zip(score.cpu(), va.cpu())if p > min_score]
    peak = []
    for p, m in zip(score.cpu(), va.cpu()):
        if p > min_score :
            peak.append((float(p), int(m) % heatmap.size(1), int(m) //heatmap.size(1)))
    return peak


    raise NotImplementedError('extract_peak')


class Detector(torch.nn.Module):
    class Test(torch.nn.Module):
        def __init__(self, it, ot, kernel_size=3, stride=2):
            """
            Your code here.
            Setup your detection network
            """
            super().__init__()
            self.c1 = torch.nn.Conv2d(it, ot, 3, padding=kernel_size // 2, stride=stride)
            self.b1 = torch.nn.BatchNorm2d(ot)
            self.c2 = torch.nn.Conv2d(ot, ot, 3, padding=kernel_size // 2)
            self.b2 = torch.nn.BatchNorm2d(ot)
            self.c3 = torch.nn.Conv2d(ot, ot, 3, padding=kernel_size // 2)
            self.b3 = torch.nn.BatchNorm2d(ot)
            self.skip = torch.nn.Conv2d(it, ot, kernel_size=1, stride=stride)
        
        #raise NotImplementedError('Detector.__init__')

        def forward(self, a):
            """
            Your code here.
            Implement a forward pass through the network, use forward for training,
            and detect for detection
            """

            ba=self.b1(self.c1(a))
            na=self.c2(F.relu(ba))
            eta=self.b3(self.c3(F.relu(self.b2(na))))
            return F.relu(eta + self.skip(a))
            raise NotImplementedError('Detector.forward')

    class KTest(torch.nn.Module):
        def __init__(self, it, ot, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(it, ot, 3, padding=3 // 2, stride=stride, output_padding=1)

        def forward(self, a):
            return F.relu(self.c1(a))

        # def detect(self, image):
        #     """
        #     Your code here.
        #     Implement object detection here.
        #     @image: 3 x H x W image
        #     @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
        #                 return no more than 30 detections per image per class. You only need to predict width and height
        #                 for extra credit. If you do not predict an object size, return w=0, h=0.
        #     Hint: Use extract_peak here
        #     Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
        #             scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
        #             out of memory.
        #     """
        #     raise NotImplementedError('Detector.detect')

    def __init__(self, values=[16, 32, 64, 128], n_class=3, kernel_size=3, pskip=True):
        """
        Your code here.
        Setup your detection network
        """
        super().__init__()
        valueskip = [3] + values[:-1]
        c= 3
        self.pskip = pskip
        self.idev = torch.Tensor([0.18182722, 0.18656468, 0.15938024]) 
        self.n_stat = len(values)
        self.imean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])

        for i, m in enumerate(values):
            self.add_module('conv%d' % i, self.Test(c, m, 3, 2)) 
            c= m

        for i, m in list(enumerate(values))[::-1]: 
            self.add_module('upconv%d' % i, self.KTest(c, m, 3, 2)) 
            c= m
            if self.pskip:
                c =c + valueskip[i]

        self.classifier = torch.nn.Conv2d(c, n_class, 1)
        self.size = torch.nn.Conv2d(c, 2, 1)

    def forward(self, a):
        """
        Your code here.
        Implement a forward pass through the network, use forward for training,
        and detect for detection
        """
        alpha= (a - self.imean[None, :, None, None].to(a.device))
        beta= self.idev[None, :, None, None].to(a.device)
        kmagic = []
        c =  alpha/beta
        #for all the range, appending the value of c
        for i in range(self.n_stat):
            kmagic.append(c)
            c = self._modules['conv%d' % i](c)
        for i in reversed(range(self.n_stat)):
            #Dimension except 2
            c = self._modules['upconv%d' % i](c)
            c = c[:, :, :kmagic[i].size(2), :kmagic[i].size(3)]
            if self.pskip:
                c = torch.cat([c, kmagic[i]], dim=1)
        return self.classifier(c), self.size(c)

    def detect(self, image, **kwargs):
        """
       Your code here.
       Implement object detection here.
       @image: 3 x H x W image
       @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
need to predict width and height
w=0, h=0.
         for extra credit. If you do not predict an object size, return
Hint: Use extract_peak here
return no more than 30 detections per image per class. You only
           Hint: Make sure to return three python lists of tuples of (float, int,
int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the
background and your program will run
                 out of memory.
        """
        #original code
        #cls, size = self.forward(image[None])
        #return [[(p, a, b, float(size.cpu()[0, 0, b, a]), float(size.cpu()[0, 1, b, a])) 
        #for p, a, b in extract_peak(c, max_det=25, **kwargs)] for c in cls[0]]

        cls, size = self.forward(image[None])
        max_detections = 30  # Maximum number of detections per image per class
        detections_per_class = []

        for c in cls[0]:
            detections = extract_peak(c, max_det=max_detections, **kwargs)
            detections = sorted(detections, reverse=True)  # Sort by score in descending order
            detections = detections[:max_detections]  # Limit the number of detections
            detections_per_class.append([(float(p), a, b, 0, 0) for p, a, b in detections])

        return detections_per_class



def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
