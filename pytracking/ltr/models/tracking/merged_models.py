

import torch
import torch.nn as nn
from importedsiammask.experiments.siammask_sharp.custom import Refine,UP
from ltr.models.tracking import dimpnet
from ltr import model_constructor
import time
import torch.onnx
import torchvision.models as models
import netron
from torchviz import make_dot
class MergedModel(nn.Module):
    def __init__(self, dimpnet, **kwargs):
        super(MergedModel, self).__init__()
        #TODO have to adjust these params to match the format 
        self.dimppart = dimpnet
        self.rest=Refine()
        self.conv1 = nn.Conv2d(2, 512, kernel_size=1, stride=1, padding=0) # 3
        self.rpn = UP(5, feature_in=256, feature_out=256)
        self.downsample = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256))


                            
                                
    
    def forward(self, train_imgs, test_imgs,train_masks,test_masks, train_bb):
        '''print(train_imgs.shape)
        print(test_imgs.shape)
        print(train_masks.shape)
        print(test_masks.shape)
        print(train_bb.shape)
        time.sleep(1000)'''

        




        target_scores,skipInfo,template_feature=self.dimppart.forward_merged(train_imgs, test_imgs, train_bb)
       
        best_score=target_scores[-1]
        
        

        
        #scetch stuff end
        rpn_pred_mask=self.rest(f=skipInfo,corr_feature=best_score)
        
        
        
       
        template_feature=template_feature[-1]
        template_feature=self.downsample(template_feature)
        search_feature=skipInfo[-1]
        search_feature=self.downsample(search_feature)
        
        return  rpn_pred_mask,template_feature,search_feature
    
    def extract_backbone_features(self,im_patches):
        return self.dimppart.extract_backbone_features(im_patches)
    
    def extract_classification_feat(self,backbone_feat):
        return self.dimppart.extract_classification_feat(backbone_feat)
    
    def classify_target(self,target_filter, sample_x):
        return self.dimppart.classifier.classify(target_filter, sample_x)
    def create_seg(self,scores_raw,skipInfo):
        return self.rest(f=skipInfo,corr_feature=scores_raw)
@model_constructor
def dimpmerged(dimppart):
    net=MergedModel(dimppart)
    return net