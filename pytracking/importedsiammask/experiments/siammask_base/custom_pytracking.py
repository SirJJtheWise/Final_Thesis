from models.siammask import SiamMask
from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr
from models.mask import Mask
import torch
import torch.nn as nn
from utils.load_helper import load_pretrain
from resnet import resnet50


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x

#TODO this is the resnet before the cross correlation
class ResDown(MultiStageFeature):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        if pretrain:
            load_pretrain(self.features, 'resnet.model')
        #after resentet before adjust
        self.downsample = ResDownS(1024, 256)

        self.layers = [self.downsample, self.features.layer2, self.features.layer3]
        self.train_nums = [1, 3]
        self.change_point = [0, 0.5]

        self.unfix(0.0)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(filter(lambda x:x.requires_grad, module.parameters()))
            if len(params):
                return [{'params': params, 'lr': lr * mult}]
            else:
                return []

        groups = []
        groups += _params(self.downsample)
        groups += _params(self.features, 0.1)
        return groups

    def forward(self, x):
        output = self.features(x)
        #after resentet before adjust
        p3 = self.downsample(output[1])
        return p3


class UP(RPN):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MaskCorr(Mask):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz**2)

    def forward(self, z, x):
        return self.mask(z, x)

#TODO this is the model architecture
class Custom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()

    def template(self, template):
        self.zf = self.features(template)

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc
    
    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches


    #TODO this is what we want to edit
    def track_mask(self, search):
        #this is the resnet
        #search = self.features(search)
        # Extract backbone features
        #TODO this is the tranferable backbone part

        #TODO this is the tranferable backbone part
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        #TODO this together with backbone is feature extractor F
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        #TODO this is the convolution part 
        scores_raw = self.classify_target(test_x)
        #here is the cut where the format needs to be adjusted
        pred_mask = self.mask(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc, pred_mask

'''
def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        #TODO this is the tranferable backbone part
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        #TODO this together with backbone is feature extractor F
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        #TODO this is the convolution part 
        scores_raw = self.classify_target(test_x)

        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
        new_pos = sample_pos[scale_ind,:] + translation_vec

        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])


        # ------- UPDATE ------- #

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()

        # Visualize and set debug info
        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))
        self.debug_info['flag' + self.id_str] = flag
        self.debug_info['max_score' + self.id_str] = max_score
        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        out = {'target_bbox': output_state}
        return out


'''
'''
import math
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
from ltr.models.target_classifier import residual_modules
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor


class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))


    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})



@model_constructor
def dimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=256, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=()):

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50_simple(filter_size=1, optim_iter=5, optim_init_reg=0.01,
                     classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                     clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                     out_feature_dim=512, hinge_threshold=0.05, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                     activation_leak=0.0, score_act='relu', act_param=None,
                     detach_length=float('Inf'), frozen_backbone_layers=()):

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Residual module that defined the online loss
    residual_module = residual_modules.LinearFilterHinge(feat_stride=feat_stride, init_filter_reg=optim_init_reg,
                                                         hinge_threshold=hinge_threshold, activation_leak=activation_leak,
                                                   score_act=score_act, act_param=act_param)

    # Construct generic optimizer module
    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def L2dimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=256, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              detach_length=float('Inf'), hinge_threshold=-999, gauss_sigma=1.0, alpha_eps=0):
    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPL2SteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step, hinge_threshold=hinge_threshold,
                                                    init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                    detach_length=detach_length, alpha_eps=alpha_eps)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def klcedimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                  classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
                  clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                  out_feature_dim=256, gauss_sigma=1.0,
                  iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                  detach_length=float('Inf'), alpha_eps=0.0, train_feature_extractor=True,
                  init_uni_weight=None, optim_min_reg=1e-3, init_initializer='default', normalize_label=False,
                  label_shrink=0, softmax_reg=None, label_threshold=0, final_relu=False, init_pool_square=False,
                  frozen_backbone_layers=()):

    if not train_feature_extractor:
        frozen_backbone_layers = 'all'

    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim, final_relu=final_relu)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim, init_weights=init_initializer,
                                                          pool_square=init_pool_square)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.PrDiMPSteepestDescentNewton(num_iter=optim_iter, feat_stride=feat_stride,
                                                          init_step_length=optim_init_step,
                                                          init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                          detach_length=detach_length, alpha_eps=alpha_eps,
                                                          init_uni_weight=init_uni_weight,
                                                          min_filter_reg=optim_min_reg, normalize_label=normalize_label,
                                                          label_shrink=label_shrink, softmax_reg=softmax_reg,
                                                          label_threshold=label_threshold)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def klcedimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                  classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                  clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                  out_feature_dim=512, gauss_sigma=1.0,
                  iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                  detach_length=float('Inf'), alpha_eps=0.0, train_feature_extractor=True,
                  init_uni_weight=None, optim_min_reg=1e-3, init_initializer='default', normalize_label=False,
                  label_shrink=0, softmax_reg=None, label_threshold=0, final_relu=False, frozen_backbone_layers=()):

    if not train_feature_extractor:
        frozen_backbone_layers = 'all'

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_bottleneck(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim, final_relu=final_relu)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim, init_weights=init_initializer)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.PrDiMPSteepestDescentNewton(num_iter=optim_iter, feat_stride=feat_stride,
                                                          init_step_length=optim_init_step,
                                                          init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                          detach_length=detach_length, alpha_eps=alpha_eps,
                                                          init_uni_weight=init_uni_weight,
                                                          min_filter_reg=optim_min_reg, normalize_label=normalize_label,
                                                          label_shrink=label_shrink, softmax_reg=softmax_reg,
                                                          label_threshold=label_threshold)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net
    '''
