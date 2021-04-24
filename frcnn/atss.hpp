#ifndef ATSS_HEAD_HPP
#define ATSS_HEAD_HPP

#include <json.hpp>
#include <bbox.hpp>
#include <anchor.hpp>
#include <losses.hpp>
#include <data.hpp>
#include <necks.hpp>
#include <backbones.hpp>
#include <utils.hpp>

namespace atss
{
  using namespace frcnn;
  using json = nlohmann::json;

  //
  // There are two versions of ATSS algorithm, one is for anchor-based detector
  // and the other is for anchor-free detector. Here we present the anchor-based one.
  // The overall setting and implementation are following mmdetection's, but the impl of
  // ATSS pos/neg sample assigning process is a bit different than mmdet's. This is simpler
  // as it does not need to fit in mmdet's big framework and it only supports single-image.
  //

  //
  // ATSSHead is the core of ATSS detector, it does detections on multi-level features which are outputs of a FPN.
  class ATSSHeadImpl : public torch::nn::Module
  {
  public:
    ATSSHeadImpl(int in_channels,
		 int feat_channels,
		 int num_classes,
		 int stacked_convs,
		 const json &anchor_opts,
		 const json &bbox_coder_opts,
		 const json &loss_cls_opts,
		 const json &loss_bbox_opts,
		 const json &loss_centerness_opts,
		 const json &train_opts,
		 const json &test_ops);
    
    ATSSHeadImpl(const json &opts);
    // return cls_scores, bbox_preds, ctr_scores of all levels
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
    forward(std::vector<torch::Tensor> feats);

    // return a map of losses
    std::map<std::string, torch::Tensor>
    forward_train(std::vector<torch::Tensor> feats, ImgData &img_data);
    
    // return det_bboxes, det_scores, det_labels
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_test
    (std::vector<torch::Tensor> feats, ImgData &img_data);
    
  private:
    int _in_channels;
    int _feat_channels;
    int _num_classes;
    int _stacked_convs;
    json _anchor_opts;
    json _bbox_coder_opts;
    json _train_opts;
    json _test_opts;
    json _loss_cls_opts;
    json _loss_bbox_opts;
    json _loss_centerness_opts;

    BBoxRegressCoder _bbox_coder;
    AnchorGenerator _anchor_generator;

    torch::nn::Sequential _cls_convs{nullptr};
    torch::nn::Sequential _reg_convs{nullptr};
    torch::nn::Conv2d _classifier{nullptr};
    torch::nn::Conv2d _regressor{nullptr};
    torch::nn::Conv2d _centerness{nullptr};
    torch::Tensor _scales;
    

    std::shared_ptr<Loss> _loss_cls{nullptr};
    std::shared_ptr<Loss> _loss_bbox{nullptr};
    std::shared_ptr<Loss> _loss_centerness{nullptr};

  };
  TORCH_MODULE(ATSSHead);

  // calculate ltrb(distance to left, top, right, bottom of some bbox) of a point in a bbox
  torch::Tensor calc_ltrb(torch::Tensor point, torch::Tensor bboxes);
  // ltrb is a [n, 4] tensor
  torch::Tensor calc_centerness(torch::Tensor ltrb);


  //
  // The ATSS detector based on anchor-based one-stage detector---RetinaNet
  //
  class ATSSImpl : public torch::nn::Module
  {
  public:
    ATSSImpl(const json &backbone_opts,
	     const json &fpn_opts,
	     const json &atss_head_opts);
    // return a map of losses
    std::map<std::string, torch::Tensor> forward_train
    (torch::Tensor img_tsr, ImgData &img_data);

    // return det_bboxes, det_scores, det_labels
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_test
    (torch::Tensor img_tsr, ImgData &img_data);

  private:
    json _backbone_opts;
    json _fpn_opts;
    json _atss_head_opts;

    std::shared_ptr<Backbone> _backbone{nullptr};
    FPN _neck{nullptr};
    ATSSHead _atss_head{nullptr};
  };

  TORCH_MODULE(ATSS);
  
}


#endif
