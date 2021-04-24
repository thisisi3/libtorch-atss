#include <losses.hpp>



namespace frcnn{

  std::shared_ptr<Loss> build_loss(const json &opts){
    std::string type = opts["type"].get<std::string>();
    if (type=="L1Loss"){
      return std::make_shared<L1Loss>(opts);
    } else if (type=="GIoULoss"){
      return std::make_shared<GIoULoss>(opts);
    } else if (type=="CrossEntropyLoss"){
      return std::make_shared<CrossEntropyLoss>(opts);
    } else if (type=="BinaryCrossEntropyLoss"){
      return std::make_shared<BinaryCrossEntropyLoss>(opts);
    } else if (type=="FocalLoss"){
      return std::make_shared<FocalLoss>(opts);
    } else {
      throw std::runtime_error("not supported loss type: "+type);
    }
    return nullptr;
  }


  //
  // L1 loss
  //
  L1Loss::L1Loss(double loss_weight)
    : _loss_weight(loss_weight)
  {}
  L1Loss::L1Loss(const json &opts)
    : _loss_weight(opts["loss_weight"].get<double>())
  { }
  torch::Tensor L1Loss::forward(torch::Tensor pred, torch::Tensor target, double avg_factor){
    return (pred - target).abs().sum() / avg_factor * _loss_weight;
  }

  //
  // GIoU loss
  //
  GIoULoss::GIoULoss(double loss_weight)
    : _loss_weight(loss_weight)
  { }
  GIoULoss::GIoULoss(const json &opts)
    : _loss_weight(opts["loss_weight"].get<double>())
  { }
  // assume both pred and target represent a same amount of bboxes                                                       
  torch::Tensor GIoULoss::forward(torch::Tensor pred, torch::Tensor target, double avg_factor){
    return (1 - giou(pred, target)).sum() / avg_factor * _loss_weight;
  }
  
  torch::Tensor GIoULoss::forward(torch::Tensor pred, torch::Tensor target, torch::Tensor weight, double avg_factor){
    auto giou_val = giou(pred, target);
    auto loss = 1 - giou_val; // [n]
    loss *= weight;
    return loss.sum() / avg_factor * _loss_weight;
  }
 

  //
  // CrossEntropy loss
  //
  CrossEntropyLoss::CrossEntropyLoss(double loss_weight)
    : _loss_weight(loss_weight)
  { }
  CrossEntropyLoss::CrossEntropyLoss(const json &opts)
    : _loss_weight(opts["loss_weight"].get<double>())
  { }
  torch::Tensor CrossEntropyLoss::forward(torch::Tensor pred, torch::Tensor target, double avg_factor) {
    return torch::nn::functional::cross_entropy
      (pred, target, torch::nn::CrossEntropyLossOptions().reduction(torch::kSum)) / avg_factor * _loss_weight;
  }

  //
  // BCE loss
  //
  BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(double loss_weight)
    : _loss_weight(loss_weight)
  { }
  BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(const json &opts)
    : _loss_weight(opts["loss_weight"].get<double>())
  { }

  torch::Tensor BinaryCrossEntropyLoss::forward(torch::Tensor pred, torch::Tensor target, double avg_factor) {
    auto opts = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kSum);
    return torch::nn::functional::binary_cross_entropy_with_logits
      (pred, target, opts) / avg_factor * _loss_weight;
  }

  //
  // Focal loss
  //
  FocalLoss::FocalLoss(double alpha, double gamma, double loss_weight)
    : _alpha(alpha), _gamma(gamma), _loss_weight(loss_weight)
  { }
  FocalLoss::FocalLoss(const json &opts)
    : _alpha(opts["alpha"].get<double>()),
      _gamma(opts["gamma"].get<double>()),
      _loss_weight(opts["loss_weight"].get<double>())
  { }

  torch::Tensor FocalLoss::forward(torch::Tensor pred, torch::Tensor label, double avg_factor){
    int num_classes = pred.size(1);
    auto tar_label = torch::full({pred.size(0), num_classes + 1}, 0, pred.options());
    tar_label.index_put_({torch::arange(pred.size(0)), label}, 1);
    tar_label = tar_label.index({Slice(), Slice(None, num_classes)});
    auto pred_sigmoid = pred.sigmoid();
    auto pt = pred_sigmoid * tar_label + (1-pred_sigmoid) * (1 - tar_label);
    torch::Tensor focal_weight = _alpha * tar_label + (1-_alpha) * (1-tar_label);
    focal_weight = focal_weight * (1-pt).pow(_gamma);
    auto opts = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone);
    auto focal_loss = torch::nn::functional::binary_cross_entropy_with_logits(pred, tar_label, opts) * focal_weight;
    return focal_loss.sum() / avg_factor * _loss_weight;
  }

}
