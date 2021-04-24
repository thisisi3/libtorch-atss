#ifndef BBOX_HPP
#define BBOX_HPP
#include <torch/torch.h>
#include <utils.hpp>
#include <cvops.h>
#include <json.hpp>

namespace frcnn
{
  using json = nlohmann::json;
  /**
     convert btw bbox and delta given a base_bbox
   */
  class BBoxRegressCoder
  {
  public:
    BBoxRegressCoder(const std::vector<float> &means={0.0, 0.0, 0.0, 0.0},
		     const std::vector<float> &stds ={1.0, 1.0, 1.0, 1.0});
    BBoxRegressCoder(const json &opts);
    // calculate delta given base bboxes and target bboxes
    torch::Tensor encode(torch::Tensor base, torch::Tensor bboxes);
    // calculate bboxes given base bboxes and delta
    torch::Tensor decode(torch::Tensor base, torch::Tensor delta,
			 const std::vector<int64_t> &max_shape = std::vector<int64_t>());
  private:
    std::vector<float> _means;
    std::vector<float> _stds;
  };

  // calculate IoUs between a and b, return an IoU table of size [a.size(0), b.size(0)]
  torch::Tensor calc_iou(torch::Tensor a, torch::Tensor b);
  // element-wise IoU, assume a.size(0)==b.size(0)
  torch::Tensor elem_iou(torch::Tensor a, torch::Tensor b);
  // giou 
  torch::Tensor giou(torch::Tensor a, torch::Tensor b);


  // do nms on bboxes with different labels at the same time
  // the trick is to move bboxes of different label far away from each other
  // return keep inds ordered by score
  torch::Tensor batched_nms(torch::Tensor bboxes, // bboxes to apply NMS to
			    torch::Tensor scores, // scores
			    torch::Tensor labels, // labels 
			    float iou_thr);
  
  // do nms on all bboxes of the same category, and repeat this for all categories
  // under rare cases one bbox may have multiple labels
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  multiclass_nms(torch::Tensor bboxes,  // [n, #class*4] or [n, 4]
		 torch::Tensor scores,  // [n, #class]
		 float iou_thr,
		 float score_thr=-1,
		 int max_num=-1,
		 torch::Tensor weight=torch::Tensor());
  
}

#endif
