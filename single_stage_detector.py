import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from tools import *
import matplotlib.pyplot as plt

def GenerateAnchor(anc, grid):
  '''
  Generating Anchor boxes
  '''

  anchors = None
  anc = anc.view(-1,1,1,1,2)
  left_cord = grid - anc*0.5
  right_cord = grid + anc*0.5
  anchors = torch.cat((left_cord,right_cord),dim=-1)
  anchors=anchors.permute(1,0,2,3,4)

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  anc_trans = torch.zeros_like(anchors)
  new_anc_trans = torch.zeros_like(anchors)
  proposals = torch.zeros_like(anchors)
  
  anc_trans[...,2:] = anchors[...,2:] - anchors[...,:2]
  anc_trans[...,:2] = (anchors[...,2:] + anchors[...,:2])/2.
  
  if method == 'YOLO':
    new_anc_trans[...,:2] = anc_trans[...,:2] + offsets[...,:2]
    new_anc_trans[...,2:] = anc_trans[...,2:] * torch.exp(offsets[...,2:])
  if method == 'FasterRCNN':
    new_anc_trans[...,:2] = anc_trans[...,:2] + offsets[...,:2] * anc_trans[...,2:]

    new_anc_trans[...,2:] = anc_trans[...,2:] * torch.exp(offsets[...,2:])


  proposals[...,:2] =new_anc_trans[...,:2]-(new_anc_trans[...,2:]/2.) # mid pt - 1/2(w,h) = top left point
  proposals[...,2:] =  new_anc_trans[...,:2]+(new_anc_trans[...,2:]/2.) #mid pt + 1/2(w,h) =bottom right point
  
  return proposals


def IoU(proposals, bboxes):
  
  iou_mat = None
  gtboxes = bboxes[:,:,:4].clone()
  B,A,H,W,_=proposals.shape
  #to get 0 iou for cases where objects in an image<max no of objects.
  gtboxes[gtboxes==-1]=0
  #area of Union
  #area of proposal : finding width, height
  prop_wh = proposals[:,:,:,:,2:] - proposals[:,:,:,:,:2]
  prop_area = prop_wh[:,:,:,:,0] * prop_wh[:,:,:,:,1]
  #area of gtboxes : finding width, height
  gtboxes_wh = gtboxes[:,:,2:] - gtboxes[:,:,:2]
  gtboxes_area = gtboxes_wh[:,:,0] * gtboxes_wh[:,:,1]

  proposals = proposals.reshape(B,A*H*W,4,1)
  gtboxes = gtboxes.reshape(B,-1,4,1).permute(0,3,2,1)
  #area of intersection
  area_x_tl_pt = torch.maximum(proposals[:,:,0,:],gtboxes[:,:,0,:])
  area_x_br_pt = torch.minimum(proposals[:,:,2,:],gtboxes[:,:,2,:])
  area_y_tl_pt = torch.maximum(proposals[:,:,1,:],gtboxes[:,:,1,:])
  area_y_br_pt = torch.minimum(proposals[:,:,3,:],gtboxes[:,:,3,:])
  x_overlap = torch.maximum(area_x_br_pt-area_x_tl_pt,torch.zeros_like(area_x_br_pt))
  y_overlap = torch.maximum(area_y_br_pt-area_y_tl_pt,torch.zeros_like(area_y_br_pt))
  intersection_area = x_overlap*y_overlap #

  #prop area : B,A,H',W'
  #gt area : B,N
  #int area : B,A*H'*W',N
  
  prop_area = prop_area.reshape(B,-1,1)
  gtboxes_area = gtboxes_area.reshape(B,1,-1)
  union_area = prop_area+gtboxes_area-intersection_area
  iou_mat = intersection_area/union_area

  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    self.pred_layer = None
    out_dim=(5*self.num_anchors)+self.num_classes
    self.pred_layer = nn.Sequential(
      nn.Conv2d(in_channels=1280,out_channels=hidden_dim,kernel_size=1),
      nn.Dropout(p=drop_ratio),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=hidden_dim,out_channels=out_dim,kernel_size=1)
    )
  
  def _extract_anchor_data(self, anchor_data, anchor_idx):
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
  
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    conf_scores, offsets, class_scores = None, None, None
    out = self.pred_layer(features) #B,5A+C,H,W
    A=self.num_anchors
    B,_,H,W=out.shape
    # Slicing anchor data from output
    anchor_data = out[:,:5*A].clone().view(B,A,5,H,W)
    
    # Slicing class data
    class_scores = out[:,5*A:].clone()
    offsets = anchor_data[:,:,1:]
    offsets[:,:,[0,1]]=torch.sigmoid(offsets[:,:,[0,1]]) - 0.5
    # 0th index stores predicted classification scores
    # Passing through sigmoid to squash b/w 0,1
    conf_scores = torch.sigmoid(anchor_data[:,:,0])

    #for training
    if pos_anchor_idx is not None:
      #combining pos and neg anchors indicies
      anchors_idx = torch.cat([pos_anchor_idx,neg_anchor_idx],dim=0)
      conf_scores = conf_scores.view(-1)
      # getting predicted scores for pos and neg anchors
      conf_scores = conf_scores[anchors_idx].unsqueeze(1)
      ###Offsets###
      offsets=offsets.permute(0,1,3,4,2).reshape(-1,4)
      offsets = offsets[pos_anchor_idx]
      ###class_scores###
      class_scores = self._extract_class_scores(class_scores,pos_anchor_idx)

    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    B = images.shape[0]
    image_feats = self.feat_extractor(images) #B,1280*7*7
    self.anchor_list=self.anchor_list.to(image_feats.device)
    grid_list = GenerateGrid(B)
    anc_list = GenerateAnchor(self.anchor_list,grid_list) #B,A,H,W,4
    iou_mat = IoU(anc_list,bboxes)
    activated_anc_ind, negative_anc_ind, GT_conf_scores, \
      GT_offsets, GT_class, \
        activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(anc_list,bboxes,grid_list,iou_mat,neg_thresh=0.2)
    #print(image_feats.shape[1],self.anchor_list.shape[0])
    
    #DONT EVER DEFINE A NETWORK VARIABLE INSIDE FUNCTION as the network gets defined again and again and thus no learning!!!
    
    # pred_network = PredictionNetwork(1280,num_anchors = 9, num_classes=self.num_classes)
    # pred_network = pred_network.to(device=image_feats.device)
    conf_scores, offsets, class_prob = self.pred_network(image_feats, activated_anc_ind, negative_anc_ind)
    
    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss = BboxRegression(offsets, GT_offsets)
    anc_per_img=torch.prod(torch.tensor(anc_list.shape[1:-1]))

    cls_loss = ObjectClassification(class_prob, GT_class, B, anc_per_img, activated_anc_ind)
    

    
    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss 

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    final_proposals, final_conf_scores, final_class = [], [], []
    with torch.no_grad():
      B = images.shape[0]
      image_feats = self.feat_extractor(images) #B,1280*7*7
      self.anchor_list=self.anchor_list.to(image_feats.device)
      grid_list = GenerateGrid(B)
      anc_list = GenerateAnchor(self.anchor_list,grid_list) #B,A,H,W,4
      conf_scores, offsets, class_prob = self.pred_network(image_feats)
      #print(conf_scores.shape)
      class_prob = class_prob.permute(0,2,3,1)
      offsets = offsets.permute(0,1,3,4,2)
      proposals = GenerateProposal(anc_list,offsets)
      final_proposals_list, final_conf_scores_list, final_class_list= [],[],[]
      C = class_prob.shape[-1]
      A,H,W = offsets.shape[1:4]
      for idx in range(0,B):
        #indexing data
        img_conf_scores = conf_scores[idx].view(-1)
        cf_idxs = (img_conf_scores>thresh).nonzero().view(-1)
        img_class_prob=class_prob[idx].unsqueeze(0).expand(A,H,W,C)
        img_class_prob = img_class_prob.argmax(dim=3).view(-1)
        img_proposals = proposals[idx].reshape(-1,4)
        #---

        img_proposals =img_proposals[cf_idxs,:] 
        img_conf_scores = img_conf_scores[cf_idxs]
        img_class_prob=img_class_prob[cf_idxs]
        keep_idxs = torchvision.ops.nms(img_proposals,img_conf_scores,nms_thresh)
        img_final_proposals = img_proposals[keep_idxs]
        img_conf_scores = img_conf_scores[keep_idxs]
        img_class_prob = img_class_prob[keep_idxs]
        final_proposals_list.append(img_final_proposals)
        final_conf_scores_list.append(img_conf_scores.unsqueeze(1))
        final_class_list.append(img_class_prob.unsqueeze(1))

      final_proposals=final_proposals_list
      final_conf_scores=final_conf_scores_list
      final_class=final_class_list
    return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
 
  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = []
  used =[]
  boxes_=boxes.reshape(1,boxes.shape[0],1,1,4)
  scores=scores.clone()
  boxes=boxes.clone()
  while(True):
    curr_idx = scores.argmax().item()
    used.append(curr_idx)
    if scores[curr_idx]==-10:
      break
    keep.append(curr_idx)
    scores[used] = -10
    curr_bbox = boxes[curr_idx].reshape(1,1,-1)
    iou_mat = IoU(boxes_,curr_bbox).view(-1)

    invalid_idxs = (iou_mat>iou_threshold).nonzero().view(-1)
    scores[invalid_idxs]=-100

    # cand_scores=scores[valid_idxs].clone()
    # cand_boxes = boxes[valid_idxs].clone()
    # max_score_idx = cand_scores.argmax()
    # keep.append(max_score_idx.item())
    # curr_box = cand_boxes[max_score_idx].reshape(1,1,-1)  
    # cand_boxes= torch.cat([cand_boxes[:max_score_idx],cand_boxes[max_score_idx+1:]])
    # valid_boxes = cand_boxes
    # valid_boxes=valid_boxes.reshape(1,valid_boxes.shape[0],1,1,4)
    # iou_mat = IoU(valid_boxes,curr_box).view(-1)
    # valid_idxs = (iou_mat<iou_threshold).nonzero().view(-1)
    # print()

  keep = torch.tensor(keep)
  keep_scores = scores[keep]
  idxs = torch.argsort(keep_scores,0,descending=True)
  keep = keep[idxs]
  if topk:
    keep=keep[:topk]
 
  return keep

def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

