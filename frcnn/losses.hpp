#ifndef LOSS_HPP
#define LOSS_HPP

#include <torch/torch.h>
#include <bbox.hpp>

namespace frcnn{
  // loss base class
  class Loss : public torch::nn::Module
  {
  public:
    virtual torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor){
      throw std::runtime_error("Please implement loss");
    }
    virtual torch::Tensor forward(torch::Tensor pred, torch::Tensor target,
				  torch::Tensor weight, double avg_factor){
      throw std::runtime_error("Please implement loss");
    }
  };

  // L1 loss
  class L1Loss : public Loss
  {
  public:
    L1Loss(double loss_weight);
    L1Loss(const json &opts);
    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor) override;
  private:
    double _loss_weight;
  };

  // GIoU loss
  class GIoULoss : public Loss
  {
  public:
    GIoULoss(double loss_weight);
    GIoULoss(const json &opts);
    // assume both pred and target represent a same amount of bboxes
    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor) override;
    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, torch::Tensor weight, double avg_factor) override;
  private:
    double _loss_weight;
  };

  // CE loss
  class CrossEntropyLoss : public Loss
  {
  public:
    CrossEntropyLoss(double loss_weight);
    CrossEntropyLoss(const json &opts);
    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor) override;
  private:
    double _loss_weight;
  };

  // BCE loss
  class BinaryCrossEntropyLoss : public Loss
  {
  public:
    BinaryCrossEntropyLoss(double loss_weight);
    BinaryCrossEntropyLoss(const json &opts);
    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor) override;
  private:
    double _loss_weight;
  };

  // focal loss
  class FocalLoss : public Loss
  {
  public:
    FocalLoss(double alpha, double gamma, double loss_weight);
    FocalLoss(const json &opts);
    // pred: [n, num_cls]
    // target: [n] where 0-num_cls-1 is fg labels and num_cls is bg label
    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, double avg_factor) override;
  private:
    double _alpha;
    double _gamma;
    double _loss_weight;
    
  };
    
  

  std::shared_ptr<Loss> build_loss(const json &opts);

}


#endif
