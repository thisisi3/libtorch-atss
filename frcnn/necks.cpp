#include <necks.hpp>
#include <utils.hpp>

namespace frcnn
{
  FPNImpl::FPNImpl(const json &opts)
    : FPNImpl(opts["out_channels"].get<int>(),
	      opts["feat_channels"].get<std::vector<int>>(),
	      opts["start_level"].get<int>(),
	      opts["num_outs"].get<int>())
  { }
  
  FPNImpl::FPNImpl(int out_channels,
		   const std::vector<int> &feat_channels,
		   int start_level,
		   int num_outs)
    : _out_channels(out_channels),
      _feat_channels(feat_channels),
      _start_level(start_level),
      _num_outs(num_outs)
  {
    _lateral_convs = torch::nn::ModuleList();
    _fpn_convs = torch::nn::ModuleList();
    _extra_convs = torch::nn::ModuleList();
    for(int i=_start_level; i<feat_channels.size(); i++){
      int feat_channel = feat_channels[i];
      _lateral_convs->push_back
	(torch::nn::Conv2d(torch::nn::Conv2dOptions(feat_channel, out_channels, 1)));
      _fpn_convs->push_back
	(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)));
    }
    _num_extra_convs = _num_outs - (feat_channels.size() - start_level);
    
    if (_num_extra_convs > 0){
      for (int i=0; i<_num_extra_convs; i++){
	_extra_convs->push_back
	  (torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(2).padding(1)));
      }
    }
    register_module("lateral_convs", _lateral_convs);
    register_module("fpn_convs", _fpn_convs);
    register_module("extra_convs", _extra_convs);
    
    init_weights();
  }

  std::vector<torch::Tensor> FPNImpl::forward(std::vector<torch::Tensor> feats)
  {
    ASSERT(feats.size() == _feat_channels.size(),
	   "number of feature levels does not match number of feat_channels");
    std::vector<torch::Tensor> lateral_outs;
    for(int i=_start_level; i < feats.size(); i++){
      lateral_outs.push_back(_lateral_convs[i-_start_level]->as<torch::nn::Conv2d>()->forward(feats[i]));
    }
    std::vector<torch::Tensor> upsample_outs({lateral_outs.back()});
    for(int i=lateral_outs.size()-2; i>=0; i--){
      auto up_size = lateral_outs[i].sizes();
      auto inter_opts = F::InterpolateFuncOptions().mode(torch::kNearest);
      inter_opts.size(std::vector<int64_t>({up_size[2], up_size[3]}));
      upsample_outs.push_back(F::interpolate(upsample_outs.back(), inter_opts) + lateral_outs[i]);
    }
    std::reverse(upsample_outs.begin(), upsample_outs.end());
    
    std::vector<torch::Tensor> outs;
    for(int i=_start_level; i<feats.size(); i++){
      outs.push_back(_fpn_convs[i-_start_level]->as<torch::nn::Conv2d>()->forward(upsample_outs[i-_start_level]));
    }
    for(int i=0; i<_num_extra_convs; i++){
      outs.push_back(_extra_convs[i]->as<torch::nn::Conv2d>()->forward(outs.back()));
    }
    return outs;
  }

  void FPNImpl::init_weights(){
    for(auto& module : modules(/*include_self=*/false)) {
      if(auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())){
	torch::nn::init::xavier_uniform_(M->weight);
	torch::nn::init::constant_(M->bias, 0);
      }
    }
  }


}
