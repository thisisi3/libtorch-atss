#include <bbox.hpp>

namespace frcnn
{
  /*
    class BBoxRegressCoder
   */
  BBoxRegressCoder::BBoxRegressCoder(const std::vector<float> &means,
				     const std::vector<float> &stds)
    : _means(means), _stds(stds)
  {
    ASSERT(means.size()==4 && stds.size()==4, "target means and stds must have size 4");
  }

  BBoxRegressCoder::BBoxRegressCoder(const json &opts)
    : _means(opts["means"].get<std::vector<float>>()),
      _stds(opts["stds"].get<std::vector<float>>())
  { }

  torch::Tensor BBoxRegressCoder::encode(torch::Tensor base, torch::Tensor bboxes){
    base = base.to(torch::kFloat32);
    bboxes = bboxes.to(torch::kFloat32);
    auto base_xywh = xyxy2xywh(base);
    auto bboxes_xywh = xyxy2xywh(bboxes);
    auto base_wh = base_xywh.index({Slice(), Slice(2, None)});
    auto xy_delta = bboxes_xywh.index({Slice(), Slice(None, 2)}) - base_xywh.index({Slice(), Slice(None, 2)});
    xy_delta = xy_delta / base_wh;
    auto wh_delta = torch::log
      (bboxes_xywh.index({Slice(), Slice(2, None)}) / base_wh);
    auto delta = torch::cat({xy_delta, wh_delta}, 1);
    delta = delta - torch::tensor(_means).to(base).view({1, 4});
    delta = delta / torch::tensor(_stds).to(base).view({1, 4});
    return delta;
  }

  torch::Tensor BBoxRegressCoder::decode(torch::Tensor base, torch::Tensor delta,
					 const std::vector<int64_t> &max_shape){
    base = base.to(torch::kFloat32);
    delta = delta.to(torch::kFloat32);
    delta = delta * torch::tensor(_stds).to(base).view({1, 4});
    delta = delta + torch::tensor(_means).to(base).view({1, 4});
    auto base_xywh = xyxy2xywh(base);
    auto base_wh = base_xywh.index({Slice(), Slice(2, None)});
    auto bboxes_xy = delta.index({Slice(), Slice(None, 2)}) * base_wh
      + base_xywh.index({Slice(), Slice(None, 2)});
    auto bboxes_wh = torch::exp(delta.index({Slice(), Slice(2, None)})) * base_wh;
    auto bboxes =  torch::cat({bboxes_xy, bboxes_wh}, 1);
    bboxes = xywh2xyxy(bboxes);
    if (max_shape.size() >= 2){
      bboxes = restrict_bbox(bboxes, max_shape);
    }
    return bboxes;
  }


  torch::Tensor calc_iou(torch::Tensor a, torch::Tensor b){
    a = a.to(torch::kFloat32);
    b = b.to(torch::kFloat32);
    ASSERT(a.dim()==2 && b.dim()==2 && a.size(1)==4 && b.size(1)==4,
	   "in order to calculate IoU, tensors must have [n, 4] dimensions");
    a = a.view({-1, 1, 4});
    // top-left
    auto tl = torch::max(a.index({Slice(), Slice(), Slice(None, 2)}),
			 b.index({Slice(), Slice(None, 2)}));
    // bottom-right
    auto br = torch::min(a.index({Slice(), Slice(), Slice(2, None)}),
			 b.index({Slice(), Slice(2, None)}));
    auto inter_wh = br - tl;
    auto pos_mask = torch::logical_and
      ((inter_wh.index({Slice(), Slice(), 0}) > 0) ,(inter_wh.index({Slice(), Slice(), 1}) > 0));
    auto inter_area =
      inter_wh.index({Slice(), Slice(), 0}) *
      inter_wh.index({Slice(), Slice(), 1});
    inter_area.index_put_({torch::logical_not(pos_mask)}, 0.0);
    auto a_area = bbox_area(a.view({-1, 4})).view({-1, 1});
    auto b_area = bbox_area(b);
    auto iou = inter_area / (a_area + b_area - inter_area);
    return iou;
  }

  torch::Tensor giou(torch::Tensor a, torch::Tensor b){
    double eps = 1e-7;
    a = a.to(torch::kFloat32);
    b = b.to(torch::kFloat32);
    auto tl = torch::max(a.index({Slice(), Slice(None, 2)}), b.index({Slice(), Slice(None, 2)}));
    auto br = torch::min(a.index({Slice(), Slice(2, None)}), b.index({Slice(), Slice(2, None)}));
    auto inter_wh = br - tl;
    auto pos_mask = torch::logical_and(inter_wh.index({Slice(), 0})>0, inter_wh.index({Slice(), 1})>0);
    auto inter_area = inter_wh.index({Slice(), 0}) * inter_wh.index({Slice(), 1});
    inter_area.index_put_({torch::logical_not(pos_mask)}, 0.0);
    auto a_area = bbox_area(a);
    auto b_area = bbox_area(b);
    auto union_area = a_area + b_area - inter_area;
    auto iou = inter_area / (union_area + eps);
    auto rect_min_xy = torch::min(a.index({Slice(), Slice(None, 2)}), b.index({Slice(), Slice(None, 2)}));
    auto rect_max_xy = torch::max(a.index({Slice(), Slice(2, None)}), b.index({Slice(), Slice(2, None)}));
    auto rect = torch::cat({rect_min_xy, rect_max_xy}, 1);
    auto rect_area = bbox_area(rect);
    return iou - (rect_area - union_area)/(rect_area + eps);
  }

  torch::Tensor giou_mmdet(torch::Tensor pred, torch::Tensor target){
    double eps = 1e-7;

    // overlap
    auto lt = torch::max(pred.index({Slice(), Slice(None, 2)}), target.index({Slice(), Slice(None, 2)}));
    auto rb = torch::min(pred.index({Slice(), Slice(2, None)}), target.index({Slice(), Slice(2, None)}));
    auto wh = (rb - lt).clamp(0);
    auto overlap = wh.index({Slice(), 0}) * wh.index({Slice(), 1});

    // union
    auto ap = (pred.index({Slice(), 2}) - pred.index({Slice(), 0}))
      * (pred.index({Slice(), 3}) - pred.index({Slice(), 1}));
    auto ag = (target.index({Slice(), 2}) - target.index({Slice(), 0}))
      * (target.index({Slice(), 3}) - target.index({Slice(), 1}));
    auto _union = ap + ag - overlap + eps;

    // IoU
    auto ious = overlap / _union;

    // enclose area
    auto enclose_x1y1 = torch::min(pred.index({Slice(), Slice(None, 2)}), target.index({Slice(), Slice(None, 2)}));
    auto enclose_x2y2 = torch::max(pred.index({Slice(), Slice(2, None)}), target.index({Slice(), Slice(2, None)}));
    auto enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(0);
    auto enclose_area = enclose_wh.index({Slice(), 0}) * enclose_wh.index({Slice(), 1}) + eps;

    // GIoU
    auto gious = ious - (enclose_area - _union) / enclose_area;
    return gious;
  }

  torch::Tensor batched_nms(torch::Tensor bboxes, // bboxes to apply NMS to
                            torch::Tensor scores, // scores
                            torch::Tensor labels, // labels
                            float iou_thr){
    if (bboxes.size(0)==0){
      return torch::empty({0}).to(torch::kLong);
    }
    auto nms_bboxes = bboxes;
    if (labels.defined() && labels.numel()==bboxes.size(0)){
      auto max_range = bboxes.max();
      nms_bboxes = bboxes + (labels * max_range).to(bboxes).view({bboxes.size(0), 1});
    }
    return nms(nms_bboxes, scores, iou_thr);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
  multiclass_nms(torch::Tensor bboxes,  // [n, #class*4] or [n, 4]                                      
		 torch::Tensor scores,  // [n, #class]                                                  
		 float iou_thr,
		 float score_thr,
		 int max_num,
		 torch::Tensor weight){
    int num_classes = scores.size(1), num_bboxes=bboxes.size(0);
    if (bboxes.size(1)>4){
      bboxes = bboxes.view({-1, 4});
    } else {
      bboxes = bboxes.view({num_bboxes, 1, 4});
      bboxes = bboxes.repeat({1, num_classes, 1}).view({-1, 4});
    }
    scores = scores.reshape({-1});
    auto labels = torch::arange(num_classes).to(scores).view({1, -1}).repeat({num_bboxes, 1}).view({-1});
    auto chosen = (scores>=score_thr);
    if (weight.defined() && weight.size(0)!=0){
      weight = weight.view({-1, 1}).repeat({1, num_classes}).view({-1});
      scores *= weight;
    }
    bboxes = bboxes.index({chosen});
    scores = scores.index({chosen});
    labels = labels.index({chosen});
    auto keep = batched_nms(bboxes, scores, labels, iou_thr);
    if (max_num != -1 & keep.size(0) > max_num){
      keep = keep.index({Slice(0, max_num)});
    }
    return std::make_tuple(bboxes.index({keep}), scores.index({keep}), labels.index({keep}));

  }


  
}
