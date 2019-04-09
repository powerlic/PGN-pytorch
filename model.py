import torch
from torch import nn
from torch.nn import functional as F

class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(10, 20, 30, 60)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=3,stride=1,padding=2)
        self.relu = nn.ReLU()
        #self.parsing_pp_dropout=nn.Dropout2d()#不知道dropout作用大不大


    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=3, stride=1,padding=1,bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PGN(nn.Module):
    def __init__(self,n_classes=20):
        super(PGN,self).__init__()
        self.relu=nn.ReLU(inplace=True)
        self.pool1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=2,bias=False),#padding不知道是否是1，论文是same
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),)#不晓得这里为什么加pooling

        self.bn2a_branch1=nn.Sequential(
            nn.Conv2d(64,256,1,1,bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.bn2a_branch2c=nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1,1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.bn2b_branch2c = nn.Sequential(
            nn.Conv2d(256, 64, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1,1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.bn2c_branch2c = nn.Sequential(
            nn.Conv2d(256, 64, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=256),
        )

        self.bn3a_branch1=nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=512),
        )

        self.bn3a_branch2c = nn.Sequential(
            nn.Conv2d(256, 128, 1, 2,  bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 512, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=512),
        )

        self.bn3b1_branch2c = nn.Sequential(
            nn.Conv2d(512, 128, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 512, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=512),
        )

        self.bn3b2_branch2c = nn.Sequential(
            nn.Conv2d(512, 128, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 512, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=512),
        )

        self.bn3b3_branch2c = nn.Sequential(
            nn.Conv2d(512, 128, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 512, 1, 1, bias=False),
            nn.BatchNorm2d(num_features=512),
        )

        self.bn4a_branch1 = nn.Sequential(
            nn.Conv2d(512, 1024, 1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4a_branch2c = nn.Sequential(
            nn.Conv2d(512, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1,2,dilation=2,bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b1_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b2_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b3_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b4_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b5_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b6_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b7_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b8_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b9_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b10_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b11_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b12_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b13_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b14_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b15_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b16_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b17_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b18_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b19_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b20_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b21_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        self.bn4b22_branch2c = nn.Sequential(
            nn.Conv2d(1024, 256, 1,1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 1024, 1,1, bias=False),
            nn.BatchNorm2d(num_features=1024),
        )

        #parsing network
        self.bn5a_branch1=nn.Sequential(
            nn.Conv2d(1024, 2048, 1,1, bias=False),
            nn.BatchNorm2d(num_features=2048),
        )

        self.bn5a_branch2c = nn.Sequential(
            nn.Conv2d(1024, 512, 1,1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 512, 3, padding=4,dilation=4, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 2048, 1,1, bias=False),
            nn.BatchNorm2d(num_features=2048),
        )

        self.bn5b_branch2c = nn.Sequential(
            nn.Conv2d(2048, 512, 1,1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 2048, 1,1, bias=False),
            nn.BatchNorm2d(num_features=2048),
        )

        self.bn5c_branch2c = nn.Sequential(
            nn.Conv2d(2048, 512, 1,1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 2048, 1,1, bias=False),
            nn.BatchNorm2d(num_features=2048),
        )



        self.parsing_branch1=nn.Conv2d(512,512,3,1,1,bias=True)
        self.parsing_branch2 = nn.Conv2d(1024, 512, 3,1,1, bias=True)
        self.parsing_branch3 = nn.Conv2d(2048, 512, 3,1,1, bias=True)

        self.parsing_branch=nn.Conv2d(1536,512,3,1,1,bias=True)
        self.parsing_pp_conv=PSPModule(512,512)
        self.parsing_fc=nn.Conv2d(512,n_classes,kernel_size=3,bias=True)

        #edge network
        self.edge_branch1=nn.Conv2d(512,256,3,1,1,bias=True)
        self.edge_branch2 = nn.Conv2d(1024, 256, 3,1,1, bias=True)
        self.edge_branch3 = nn.Conv2d(2048, 256, 3,1,1, bias=True)
        
        self.edge_branch=nn.Conv2d(768,256,3,1,bias=True)
        self.edge_pp_conv=PSPModule(256,512)
        self.edge_fc=nn.Conv2d(512,1,3,1,1,bias=True)
        
        #intermidium supervision
        self.fc1_edge_c0_res5 = nn.Conv2d(256, 1, 3, padding=1,bias=True)
        self.fc1_edge_c1_res5 = nn.Conv2d(256, 1, 3, padding=2,dilation=2, bias=False)
        self.fc1_edge_c2_res5 = nn.Conv2d(256, 1, 3, padding=4,dilation=4, bias=False)
        self.fc1_edge_c3_res5 = nn.Conv2d(256, 1, 3, padding=8, dilation=8, bias=False)
        self.fc1_edge_c4_res5 = nn.Conv2d(256, 1, 3, padding=16,dilation=16, bias=False)
        
        self.fc1_edge_c0_res4 = nn.Conv2d(256, 1, 3, padding=1, bias=True)
        self.fc1_edge_c1_res4 = nn.Conv2d(256, 1, 3, padding=2, dilation=2, bias=False)
        self.fc1_edge_c2_res4 = nn.Conv2d(256, 1, 3, padding=4, dilation=4, bias=False)
        self.fc1_edge_c3_res4 = nn.Conv2d(256, 1, 3, padding=8, dilation=8, bias=False)
        self.fc1_edge_c4_res4 = nn.Conv2d(256, 1, 3, padding=16, dilation=16, bias=False)
        
        self.fc1_edge_c0_res3 = nn.Conv2d(256, 1, 3, padding=1,  bias=True)
        self.fc1_edge_c1_res3 = nn.Conv2d(256, 1, 3, padding=2, dilation=2, bias=False)
        self.fc1_edge_c2_res3 = nn.Conv2d(256, 1, 3, padding=4, dilation=4, bias=False)
        self.fc1_edge_c3_res3 = nn.Conv2d(256, 1, 3, padding=8, dilation=8, bias=False)
        self.fc1_edge_c4_res3 = nn.Conv2d(256, 1, 3, padding=16, dilation=16, bias=False)
        
        # refine networks #
        
        self.parsing_fea = nn.Conv2d(512, 256, 3, bias=True)
        self.parsing_remap = nn.Conv2d(20, 128, 1, bias=True)
        self.edge_fea = nn.Conv2d(512, 256, 3,1,1,bias=True)
        self.edge_remap=nn.Conv2d(1,128,1,1,bias=True)
        
        self.parsing_rf=nn.Conv2d(768,512,3,1,1,bias=True)
        self.parsing_rf_conv=PSPModule(512,256)
        
        self.parsing_rf_fc=nn.Conv2d(256,n_classes,3,1,bias=True)
        
        self.edge_rf=nn.Conv2d(768,512,3,1,bias=True)
        self.edge_rf_conv=PSPModule(512,512)
        self.edge_rf_fc=nn.Conv2d(512,1,3,1,1,bias=True)

    def forward(self, input_data):
        pool1=self.pool1(input_data)
        bn2a_branch1=self.bn2a_branch1(pool1)
        bn2a_branch2c=self.bn2a_branch2c(pool1)
        res2a_relu=self.relu(bn2a_branch1+bn2a_branch2c)
        bn2b_branch2c=self.bn2b_branch2c(res2a_relu)
        res2b_relu=self.relu(res2a_relu+bn2b_branch2c)

        bn2c_branch2c = self.bn2b_branch2c(res2a_relu)
        res2c_relu = self.relu(res2b_relu + bn2c_branch2c)
        bn3a_branch1=self.bn3a_branch1(res2c_relu)
        bn3a_branch2c=self.bn3a_branch2c(res2c_relu)
        res3a_relu=self.relu(bn3a_branch1+bn3a_branch2c)
        bn3b1_branch2c=self.bn3b1_branch2c(res3a_relu)
        res3b1_relu=self.relu(res3a_relu+bn3b1_branch2c)
        bn3b2_branch2c=self.bn3b2_branch2c(res3b1_relu)
        res3b2_relu=self.relu(res3b1_relu+bn3b2_branch2c)
        bn3b3_branch2c=self.bn3b3_branch2c(res3b2_relu)
        res3b3_relu=self.relu(res3b2_relu+bn3b3_branch2c)
        bn4a_branch1=self.bn4a_branch1(res3b3_relu)
        bn4a_branch2c=self.bn4a_branch2c(res3b3_relu)
        res4a_relu=self.relu(bn4a_branch1+bn4a_branch2c)

        bn4b1_branch2c=self.bn4b1_branch2c(res4a_relu)
        res4b1_relu=self.relu(res4a_relu+bn4b1_branch2c)

        bn4b2_branch2c=self.bn4b2_branch2c(res4b1_relu)
        res4b2_relu=self.relu(res4b1_relu+bn4b2_branch2c)

        bn4b3_branch2c=self.bn4b3_branch2c(res4b2_relu)
        res4b3_relu=self.relu(res4b2_relu+bn4b3_branch2c)

        bn4b4_branch2c = self.bn4b4_branch2c(res4b3_relu)
        res4b4_relu = self.relu(res4b3_relu + bn4b4_branch2c)

        bn4b5_branch2c = self.bn4b5_branch2c(res4b4_relu)
        res4b5_relu = self.relu(res4b4_relu + bn4b5_branch2c)

        bn4b6_branch2c = self.bn4b6_branch2c(res4b5_relu)
        res4b6_relu = self.relu(res4b5_relu + bn4b6_branch2c)

        bn4b7_branch2c = self.bn4b7_branch2c(res4b6_relu)
        res4b7_relu = self.relu(res4b6_relu + bn4b7_branch2c)

        bn4b8_branch2c = self.bn4b8_branch2c(res4b7_relu)
        res4b8_relu = self.relu(res4b7_relu + bn4b8_branch2c)

        bn4b9_branch2c = self.bn4b9_branch2c(res4b8_relu)
        res4b9_relu = self.relu(res4b8_relu + bn4b9_branch2c)

        bn4b10_branch2c = self.bn4b10_branch2c(res4b9_relu)
        res4b10_relu = self.relu(res4b9_relu + bn4b10_branch2c)

        bn4b11_branch2c = self.bn4b11_branch2c(res4b10_relu)
        res4b11_relu = self.relu(res4b10_relu + bn4b11_branch2c)

        bn4b12_branch2c = self.bn4b12_branch2c(res4b11_relu)
        res4b12_relu = self.relu(res4b11_relu + bn4b12_branch2c)

        bn4b13_branch2c = self.bn4b13_branch2c(res4b12_relu)
        res4b13_relu = self.relu(res4b12_relu + bn4b13_branch2c)

        bn4b14_branch2c = self.bn4b14_branch2c(res4b13_relu)
        res4b14_relu = self.relu(res4b13_relu + bn4b14_branch2c)

        bn4b15_branch2c = self.bn4b15_branch2c(res4b6_relu)
        res4b15_relu = self.relu(res4b14_relu + bn4b15_branch2c)

        bn4b16_branch2c = self.bn4b16_branch2c(res4b15_relu)
        res4b16_relu = self.relu(res4b15_relu + bn4b16_branch2c)

        bn4b17_branch2c = self.bn4b17_branch2c(res4b16_relu)
        res4b17_relu = self.relu(res4b16_relu + bn4b17_branch2c)

        bn4b18_branch2c = self.bn4b18_branch2c(res4b17_relu)
        res4b18_relu = self.relu(res4b17_relu + bn4b18_branch2c)

        bn4b19_branch2c = self.bn4b19_branch2c(res4b18_relu)
        res4b19_relu = self.relu(res4b18_relu + bn4b19_branch2c)

        bn4b20_branch2c = self.bn4b20_branch2c(res4b19_relu)
        res4b20_relu = self.relu(res4b19_relu + bn4b20_branch2c)

        bn4b21_branch2c = self.bn4b21_branch2c(res4b20_relu)
        res4b21_relu = self.relu(res4b20_relu + bn4b21_branch2c)

        bn4b22_branch2c = self.bn4b22_branch2c(res4b20_relu)
        res4b22_relu = self.relu(res4b21_relu + bn4b22_branch2c)

        #parsing network#
        bn5a_branch1=self.bn5a_branch1(res4b22_relu)
        bn5a_branch2c=self.bn5a_branch2c(res4b22_relu)


        res5a_relu=self.relu(bn5a_branch1+bn5a_branch2c)
        bn5b_branch2c=self.bn5b_branch2c(res5a_relu)

        res5b_relu=self.relu(res5a_relu+bn5b_branch2c)
        bn5c_branch2c=self.bn5c_branch2c(res5b_relu)

        res5c_relu=self.relu(res5b_relu+bn5c_branch2c)
        parsing_branch1=self.parsing_branch1(res3b3_relu)
        parsing_branch2=self.parsing_branch2(res4b22_relu)
        parsing_branch3=self.parsing_branch3(res5c_relu)
        parsing_branch_concat=torch.cat([parsing_branch1,parsing_branch2,parsing_branch3],dim=1)
        parsing_branch=self.parsing_branch(parsing_branch_concat)

        parsing_pp_conv=self.parsing_pp_conv(parsing_branch)
        parsing_fc=self.parsing_fc(parsing_pp_conv)

        #edge network#
        edge_branch1=self.edge_branch1(res3b3_relu)
        edge_branch2 = self.edge_branch2(res4b22_relu)
        edge_branch3 = self.edge_branch3(res5c_relu)

        edge_branch_concat=torch.cat([edge_branch1,edge_branch2,edge_branch3],dim=1)

        edge_branch=self.edge_branch(edge_branch_concat)

        edge_pp_conv=self.edge_pp_conv(edge_branch)
        edge_fc=self.edge_fc(edge_pp_conv)

        # intermidium supervision#
        fc1_edge_c0_res5=self.fc1_edge_c0_res5(edge_branch3)
        fc1_edge_c1_res5 = self.fc1_edge_c1_res5(edge_branch3)
        fc1_edge_c2_res5 = self.fc1_edge_c2_res5(edge_branch3)
        fc1_edge_c3_res5 = self.fc1_edge_c3_res5(edge_branch3)
        fc1_edge_c4_res5 = self.fc1_edge_c4_res5(edge_branch3)
        fc1_edge_res5=fc1_edge_c0_res5+fc1_edge_c1_res5+fc1_edge_c2_res5+fc1_edge_c3_res5+fc1_edge_c4_res5

        fc1_edge_c0_res4 = self.fc1_edge_c0_res4(edge_branch2)
        fc1_edge_c1_res4 = self.fc1_edge_c1_res4(edge_branch2)
        fc1_edge_c2_res4 = self.fc1_edge_c2_res4(edge_branch2)
        fc1_edge_c3_res4 = self.fc1_edge_c3_res4(edge_branch2)
        fc1_edge_c4_res4 = self.fc1_edge_c4_res4(edge_branch2)
        fc1_edge_res4 = fc1_edge_c0_res4 + fc1_edge_c1_res4 + fc1_edge_c2_res4 + fc1_edge_c3_res4 + fc1_edge_c4_res4

        fc1_edge_c0_res3 = self.fc1_edge_c0_res3(edge_branch1)
        fc1_edge_c1_res3 = self.fc1_edge_c1_res3(edge_branch1)
        fc1_edge_c2_res3 = self.fc1_edge_c2_res3(edge_branch1)
        fc1_edge_c3_res3 = self.fc1_edge_c3_res3(edge_branch1)
        fc1_edge_c4_res3 = self.fc1_edge_c4_res3(edge_branch1)
        fc1_edge_res3 = fc1_edge_c0_res3 + fc1_edge_c1_res3 + fc1_edge_c2_res3 + fc1_edge_c3_res3 + fc1_edge_c4_res3

        parsing_fea=self.parsing_fea(parsing_pp_conv)
        parsing_remap=self.parsing_remap(parsing_fc)
        edge_fea=self.edge_fea(edge_pp_conv)
        edge_remap=self.edge_remap(edge_fc)

        parsing_rf_concat=torch.cat([parsing_fea,parsing_remap,edge_fea,edge_remap],dim=1)
        parsing_rf=self.parsing_rf(parsing_rf_concat)

        parsing_rf_conv=self.parsing_rf_conv(parsing_rf)
        parsing_rf_fc=self.parsing_rf_fc(parsing_rf_conv)

        edge_rf_concat=torch.cat([edge_fea,edge_remap,parsing_fea,parsing_remap],dim=1)
        edge_rf=self.edge_rf(edge_rf_concat)

        edge_rf_conv=self.edge_rf_conv(edge_rf)
        edge_rf_fc=self.edge_rf_fc(edge_rf_conv)

        return parsing_fc,parsing_rf_fc,edge_fc,fc1_edge_res5,fc1_edge_res4,fc1_edge_res3,edge_rf_fc















