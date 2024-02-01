import os
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import tinysizemodels as tvmodel
from .model_utils import init_net, Normalize

class Backbonemodel(BaseModel):

    def __init__(self, opt, meanstd):
        BaseModel.__init__(self, opt, meanstd)
        self.opt = opt
        self.target_model = init_net(tvmodel.__dict__[self.opt.backbone](opt.output_nc, opt.classes))
        self.model_names = ['backbone']
        self.loss_names = ['loss']
        self.data_name = self.opt.dataroot.split('/')[-1]
        self.y_class = {}
        with open(os.path.join('./datasets_dict', '{}_dict.txt'.format(self.data_name)), "r") as f:
            lines = f.readlines()
            for i in lines:
                classes = i.strip().split(":")[0]
                idx = i.strip().split(":")[1]
                self.y_class[int(idx)] = classes

        self.acc_names = ['totalacc', 'perclass']
        self.acc_perclass = [0 for i in range(opt.classes)]
        if self.isTrain:
            self.optimizer = torch.optim.SGD(self.target_model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizers.append(self.optimizer)
        assert (torch.cuda.is_available())
        self.norm = Normalize(meanstd=meanstd).cuda()
        self.netbackbone = self.target_model.cuda()
        
    def set_input(self, data):
        self.imgs = data[0].to(self.device)
        self.label = data[1].to(self.device)

    def forward(self):
        self.logits = self.netbackbone(self.norm(self.imgs))

    def backward(self):
        self.loss_loss = F.cross_entropy(self.logits, self.label).to(self.device)
        self.loss_loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def test(self, final=True, visualtsne=False):
        with torch.no_grad():
            self.forward()
            self.pred = torch.argmax(self.logits, 1)
            self.acc_totalacc = torch.sum(self.pred == self.label, 0)
            if final:
                self.acc_perclass = [0 for i in range(self.opt.classes)]
                if self.opt.batch_size == 1:
                    self.acc_perclass[self.label] = float(torch.sum(self.pred == self.label, 0))
                else:
                    for j, (label, cls_name) in enumerate(self.y_class.items()):
                        pred_ids = self.pred[self.label == label]
                        cls_num = torch.sum(self.label == label)
                        if len(pred_ids) == 0:
                            continue
                        assert int(torch.sum(label == pred_ids, 0)) <= cls_num
                        self.acc_perclass[label] += float(torch.sum(label == pred_ids, 0))
