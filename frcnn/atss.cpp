#include <atss.hpp>

namespace atss
{
  //
  // for simplicity, do model construction and weight initialization in one place
  //
  ATSSHeadImpl::ATSSHeadImpl(int in_channels,
			     int feat_channels,
			     int num_classes,
			     int stacked_convs,
			     const json &anchor_opts,
			     const json &bbox_coder_opts,
			     const json &loss_cls_opts,
			     const json &loss_bbox_opts,
			     const json &loss_centerness_opts,
			     const json &train_opts,
			     const json &test_opts)
    : _in_channels(in_channels),
      _feat_channels(feat_channels),
      _num_classes(num_classes),
      _stacked_convs(stacked_convs),
      _anchor_opts(anchor_opts),
      _anchor_generator(anchor_opts),
      _bbox_coder_opts(bbox_coder_opts),
      _loss_cls_opts(loss_cls_opts),
      _loss_bbox_opts(loss_bbox_opts),
      _loss_centerness_opts(loss_centerness_opts),
      _train_opts(train_opts),
      _test_opts(test_opts),
      _bbox_coder(bbox_coder_opts)
  {
    _cls_convs = torch::nn::Sequential();
    _reg_convs = torch::nn::Sequential();
    int in_chan = _in_channels;
    for (int i=0; i<_stacked_convs; i++){
      _cls_convs->push_back(torch::nn::Conv2d
			    (torch::nn::Conv2dOptions(in_chan, _feat_channels, 3).padding(1).stride(1).bias(false)));
      _cls_convs->push_back(torch::nn::GroupNorm(32, _feat_channels));
      _cls_convs->push_back(torch::nn::ReLU(true));
      _reg_convs->push_back(torch::nn::Conv2d
			    (torch::nn::Conv2dOptions(in_chan, _feat_channels, 3).padding(1).stride(1).bias(false)));
      _reg_convs->push_back(torch::nn::GroupNorm(32, _feat_channels));
      _reg_convs->push_back(torch::nn::ReLU(true));
      in_chan = _feat_channels;
    }
    _classifier = torch::nn::Conv2d(torch::nn::Conv2dOptions(_feat_channels, _num_classes, 3).padding(1).stride(1));
    _regressor = torch::nn::Conv2d(torch::nn::Conv2dOptions(_feat_channels, 4, 3).padding(1).stride(1));
    _centerness = torch::nn::Conv2d(torch::nn::Conv2dOptions(_feat_channels, 1, 3).padding(1).stride(1));
    _scales = torch::tensor(std::vector<double>(_anchor_generator->num_strides(), 1),
			    torch::TensorOptions().dtype(torch::kFloat32));
    
    _loss_cls = build_loss(loss_cls_opts);
    _loss_bbox = build_loss(loss_bbox_opts);
    _loss_centerness = build_loss(loss_centerness_opts);

    register_module("cls_convs", _cls_convs);
    register_module("reg_convs", _reg_convs);
    register_module("classifier", _classifier);
    register_module("regressor", _regressor);
    register_module("centerness", _centerness);
    register_module("loss_cls", _loss_cls);
    register_module("loss_bbox", _loss_bbox);
    register_module("loss_centerness", _loss_centerness);
    register_module("anchor_generator", _anchor_generator);
    register_parameter("scales", _scales);

    // init_weights
    for (int i=0; i<_stacked_convs; i++){
      torch::nn::init::normal_(_cls_convs[i*3]->as<torch::nn::Conv2d>()->weight, 0, 0.01);
      torch::nn::init::normal_(_reg_convs[i*3]->as<torch::nn::Conv2d>()->weight, 0, 0.01);
    }
    torch::nn::init::normal_(_classifier->weight, 0, 0.01);
    torch::nn::init::normal_(_regressor->weight, 0, 0.01);
    torch::nn::init::normal_(_centerness->weight, 0, 0.01);
    torch::nn::init::constant_(_classifier->bias, -4.6); // sigmoid of which is 0.01
    torch::nn::init::constant_(_regressor->bias, 0);
    torch::nn::init::constant_(_centerness->bias, 0);

  }

  ATSSHeadImpl::ATSSHeadImpl(const json &opts)
    : ATSSHeadImpl(opts["in_channels"].get<int>(),
		   opts["feat_channels"].get<int>(),
		   opts["num_classes"].get<int>(),
		   opts["stacked_convs"].get<int>(),
		   opts["anchor_opts"].get<json>(),
		   opts["bbox_coder_opts"].get<json>(),
		   opts["loss_cls_opts"].get<json>(),
		   opts["loss_bbox_opts"].get<json>(),
		   opts["loss_centerness_opts"].get<json>(),
		   opts["train_opts"].get<json>(),
		   opts["test_opts"].get<json>())
  {/* construct ATSS head using json-format options */ }

  std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
  ATSSHeadImpl::forward(std::vector<torch::Tensor> feats){
    std::vector<torch::Tensor> cls_scores, bbox_preds, ctr_scores;
    int idx=0;
    for(auto &x : feats){
      auto cls_x = _cls_convs->forward(x);
      auto reg_x = _reg_convs->forward(x);
      cls_scores.push_back(_classifier->forward(cls_x));
      bbox_preds.push_back((_regressor->forward(reg_x))*_scales[idx]);
      ctr_scores.push_back(_centerness->forward(reg_x));
      idx += 1;
    }
    return std::make_tuple(cls_scores, bbox_preds, ctr_scores);
  }

  //
  // Here you can find the main algorithm of postive-negative sample assignment
  //
  std::map<std::string, torch::Tensor>
  ATSSHeadImpl::forward_train(std::vector<torch::Tensor> feats, ImgData &img_data){
    auto outs = forward(feats);
    auto mlvl_cls_scores=std::get<0>(outs), mlvl_bbox_preds=std::get<1>(outs), mlvl_ctr_scores=std::get<2>(outs);
    auto device = mlvl_cls_scores[0].device();
    auto feat_sizes = get_grid_size(mlvl_cls_scores);
    int num_lvls = mlvl_cls_scores.size();
    // flatten everything into [n, -1]
    auto mlvl_anchors    = _anchor_generator->get_anchors(feat_sizes);
    mlvl_anchors = batch_reshape(mlvl_anchors, {-1, 4});
    mlvl_cls_scores = batch_permute(mlvl_cls_scores, {0, 2, 3, 1});
    mlvl_bbox_preds = batch_permute(mlvl_bbox_preds, {0, 2, 3, 1});
    mlvl_cls_scores = batch_reshape(mlvl_cls_scores, {-1, _num_classes});
    mlvl_bbox_preds = batch_reshape(mlvl_bbox_preds, {-1, 4});
    mlvl_ctr_scores = batch_reshape(mlvl_ctr_scores, {-1});

    // the following effort is to find topk closest anchor points in each level
    auto anchor_ctr_list = std::vector<torch::Tensor>();
    auto topk_ind_list = std::vector<torch::Tensor>();
    int num_prev_anchors = 0;
    for (int lvl=0; lvl<num_lvls; lvl++){
      auto anchor = mlvl_anchors[lvl];
      auto anchor_ctr = bbox_center(anchor);
      auto gt_ctr = bbox_center(img_data.gt_bboxes);
      auto ctr_dist = (anchor_ctr.view({-1, 1, 2}) - gt_ctr).norm(2, -1);
      auto topk_closest = ctr_dist.topk(9, 0, false);
      auto topk_closest_val=std::get<0>(topk_closest), topk_closest_ind=std::get<1>(topk_closest);
      // need to add num_prev_anchors to get the absolute position of topk inds
      topk_ind_list.push_back(topk_closest_ind + num_prev_anchors);
      anchor_ctr_list.push_back(anchor_ctr);
      num_prev_anchors += anchor.size(0);
    }
    

    // cat everything
    auto topk_inds = torch::cat(topk_ind_list);
    auto anchors_ctr = torch::cat(anchor_ctr_list);
    auto anchors = torch::cat(mlvl_anchors);
    auto cls_scores = torch::cat(mlvl_cls_scores);
    auto bbox_preds = torch::cat(mlvl_bbox_preds);
    auto ctr_scores = torch::cat(mlvl_ctr_scores);
    int num_anchors = anchors.size(0);

    // find iou_thr by adding mean and std of ious of topk closest anchor points
    auto iou_table = calc_iou(anchors, img_data.gt_bboxes);
    auto topk_iou = iou_table.gather(0, topk_inds);
    // pay close attention to how std is caculated here, it will save you a lot debugging time
    auto iou_thr = topk_iou.mean(0) + topk_iou.std(0, true);

    // the following effort is to find mask of positively assigned samples in iou_table
    // they have ious bigger than iou_thr and centers must fall inside gts
    auto pos_mask_on_topk = topk_iou >= iou_thr;
    auto pos_mask = torch::full(iou_table.sizes(), false, torch::TensorOptions().dtype(torch::kBool).device(device));
    pos_mask.scatter_(0, topk_inds, pos_mask_on_topk);

    // left, top, right, bottom distance to four sides of gt bboxes
    auto ltrb = calc_ltrb(bbox_center(anchors), img_data.gt_bboxes);
    auto ctr_in_mask = std::get<0>(ltrb.min(-1)) > 0.01;
    pos_mask = pos_mask & ctr_in_mask;

    // the following trick is to find the best gt if there are more than
    // one gt assigned to a anchor_bbox/anchor_point
    iou_table.index_put_({~pos_mask}, -1);
    auto max_gt = iou_table.max(1);
    auto max_gt_val=std::get<0>(max_gt), max_gt_ind=std::get<1>(max_gt);

    // next to get index of assigned gt
    auto pos_mask_1d = pos_mask.any(1);
    int64_t num_pos = pos_mask_1d.sum().item<int64_t>();
    auto assigned_gt_inds = torch::full({num_anchors}, -1, torch::TensorOptions().dtype(torch::kLong).device(device));
    assigned_gt_inds.index_put_({pos_mask_1d}, max_gt_ind.index({pos_mask_1d}));

    // find assigned class labels
    auto cls_labels = img_data.gt_labels.index({assigned_gt_inds});
    cls_labels.index_put_({~pos_mask_1d}, _num_classes);

    // calc cls loss
    auto cls_loss = _loss_cls->forward(cls_scores, cls_labels, std::max(num_pos, (int64_t)1));

    // in case no positive samples are found
    auto ctr_loss = torch::tensor(0.0, torch::TensorOptions().device(device).requires_grad(true));
    auto bbox_loss = torch::tensor(0.0, torch::TensorOptions().device(device).requires_grad(true));
    if(num_pos > 0){
      // calc centerness loss
      auto ltrb_1d = ltrb.index({torch::arange(num_anchors), assigned_gt_inds, Slice()});
      auto pos_ctrness_tar = calc_centerness(ltrb_1d).index({pos_mask_1d});
      ctr_loss = _loss_centerness->forward(ctr_scores.index({pos_mask_1d}),
					   pos_ctrness_tar,
					   num_pos);
      // calc bbox loss
      auto pos_decoded_bboxes = _bbox_coder.decode(anchors.index({pos_mask_1d}),
						   bbox_preds.index({pos_mask_1d}),
						   img_data.img_shape);
      auto pos_tar_bboxes = img_data.gt_bboxes.index({assigned_gt_inds.index({pos_mask_1d})});
      bbox_loss = _loss_bbox->forward(pos_decoded_bboxes,
				      pos_tar_bboxes,
				      pos_ctrness_tar,
				      1.0) / pos_ctrness_tar.sum();
    }
    auto losses = std::map<std::string, torch::Tensor>({
	{"loss_cls", cls_loss},
	{"loss_bbox", bbox_loss},
	{"loss_centerness", ctr_loss}
      });
    return losses; 
  }


  torch::Tensor calc_ltrb(torch::Tensor point, torch::Tensor bboxes){
    auto x=point.index({Slice(), 0}).view({-1, 1}), y=point.index({Slice(), 1}).view({-1, 1});
    return torch::stack({
	x - bboxes.index({Slice(), 0}),
	  y - bboxes.index({Slice(), 1}),
	  bboxes.index({Slice(), 2}) - x,
	  bboxes.index({Slice(), 3}) - y
	  }, -1);
  }
  
  torch::Tensor calc_centerness(torch::Tensor ltrb){
    double eps = 1e-6;
    auto l=ltrb.index({"...", 0}), t=ltrb.index({"...", 1}), r=ltrb.index({"...", 2}), b=ltrb.index({"...", 3});
    auto vert_ratio = torch::min(l, r) / (torch::max(l, r) + eps);
    auto hori_ratio = torch::min(t, b) / (torch::max(t, b) + eps);
    return (vert_ratio * hori_ratio).sqrt();
  }

  // return det_bboxes, det_scores, det_labels
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ATSSHeadImpl::forward_test
  (std::vector<torch::Tensor> feats, ImgData &img_data){
    int num_lvls = feats.size();
    auto feat_sizes = get_grid_size(feats);
    auto anchors    = _anchor_generator->get_anchors(feat_sizes);
    auto pred_outs = forward(feats);
    auto cls_scores=std::get<0>(pred_outs), bbox_preds=std::get<1>(pred_outs), ctr_scores=std::get<2>(pred_outs);

    cls_scores = batch_permute(cls_scores, {0, 2, 3, 1});
    bbox_preds = batch_permute(bbox_preds, {0, 2, 3, 1});
    ctr_scores = batch_permute(ctr_scores, {0, 2, 3, 1});

    anchors    = batch_reshape(anchors,    {-1, 4});
    cls_scores = batch_reshape(cls_scores, {-1, _num_classes});
    bbox_preds = batch_reshape(bbox_preds, {-1, 4});
    ctr_scores = batch_reshape(ctr_scores, {-1});

    std::vector<torch::Tensor> mlvl_cls_scores, mlvl_bbox_preds, mlvl_ctr_scores, mlvl_anchors;
    for (int i=0; i<num_lvls; i++){
      auto anchor    = anchors[i];
      auto bbox_pred = bbox_preds[i];
      auto cls_score = cls_scores[i].sigmoid();
      auto ctr_score = ctr_scores[i].sigmoid();
      // apply nms_pre
      int nms_pre = _test_opts["nms_pre"].get<int>();
      if (nms_pre >= 0 && nms_pre < cls_score.size(0)){
	auto max_score = (cls_score * ctr_score.view({-1, 1})).max(1);
	auto topk = std::get<0>(max_score).topk(nms_pre);
	auto topk_inds = std::get<1>(topk);
	anchor    = anchor.index({topk_inds});
	bbox_pred = bbox_pred.index({topk_inds});
	cls_score = cls_score.index({topk_inds});
	ctr_score = ctr_score.index({topk_inds});
      }
      mlvl_anchors.push_back(anchor);
      mlvl_bbox_preds.push_back(bbox_pred);
      mlvl_cls_scores.push_back(cls_score);
      mlvl_ctr_scores.push_back(ctr_score);
    }

    auto cat_anchors    = torch::cat(mlvl_anchors);
    auto cat_bbox_preds = torch::cat(mlvl_bbox_preds);
    auto cat_decoded_bboxes = _bbox_coder.decode(cat_anchors, cat_bbox_preds, img_data.img_shape);

    auto cat_cls_scores = torch::cat(mlvl_cls_scores);
    auto cat_ctr_scores = torch::cat(mlvl_ctr_scores);

    auto nms_res = multiclass_nms(cat_decoded_bboxes,
				  cat_cls_scores,
				  _test_opts["nms_thr"].get<float>(),
				  _test_opts["score_thr"].get<float>(),
				  _test_opts["max_per_img"].get<float>(),
				  cat_ctr_scores);
    return nms_res;
  }



  /*
    ATSS detector
   */
  ATSSImpl::ATSSImpl(const json &backbone_opts,
		     const json &fpn_opts,
		     const json &atss_head_opts)
    : _backbone_opts(backbone_opts),
      _fpn_opts(fpn_opts),
      _atss_head_opts(atss_head_opts)
  {
    _backbone = build_backbone(backbone_opts);
    _neck = FPN(fpn_opts);
    _atss_head = ATSSHead(atss_head_opts);

    std::string pretrained = backbone_opts["pretrained"].get<std::string>();
    if (pretrained != "None"){
      std::cout << "loading weights for backbone...\n";
      torch::load(_backbone, pretrained);
    }

    register_module("backbone", _backbone);
    register_module("neck", _neck);
    register_module("atss_head", _atss_head);
  }

  std::map<std::string, torch::Tensor> ATSSImpl::forward_train
  (torch::Tensor img_tsr, ImgData &img_data){
    auto feats = _backbone->forward(img_tsr);
    feats = _neck->forward(feats);
    auto losses = _atss_head->forward_train(feats, img_data);
    return losses;
  }

  // return det_bboxes, det_scores, det_labels                                                                           
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ATSSImpl::forward_test
  (torch::Tensor img_tsr, ImgData &img_data){
    auto feats = _backbone->forward(img_tsr);
    feats = _neck->forward(feats);
    auto det_res = _atss_head->forward_test(feats, img_data);
    return det_res;
  }
  


}
