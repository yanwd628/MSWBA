"""
MSWBA face forgery detector with an EfficientNet-B5 backbone.

The implementation follows the supplied design:
1. Shallow, middle and final image encoders are taken from EfficientNet-B5.
2. An attention-based DWT module is inserted after every encoder.
3. Only HH produces the high-frequency supplement added to LL before the next
   image encoder.
4. HL and LH exchange attention and only form the wavelet-frequency maps used
   by the final multiscale bidirectional cross-attention.
5. Image and frequency classifiers jointly produce the final prediction.

The detector keeps the same training-framework interface and output keys as
`6.2_effnet_fad_detector.py`.
"""

import logging
import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b5

try:
    from torchvision.models import EfficientNet_B5_Weights
    _TV_HAS_WEIGHTS_ENUM = True
except Exception:
    EfficientNet_B5_Weights = None
    _TV_HAS_WEIGHTS_ENUM = False

try:
    from torchvision.ops import DeformConv2d
except Exception:
    DeformConv2d = None

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='effnet_fad')
class MswbaEffnetB5Detector(AbstractDetector):
    """EfficientNet-B5 detector implementing the MSWBA design."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = int(config.get('num_classes', 2))
        self.label_smoothing = float(config.get('label_smoothing', 0.1))
        self.image_logit_weight = float(config.get('image_logit_weight', 0.5))
        self.branch_dropout_prob = float(
            config.get('branch_dropout_prob', 0.1)
        )
        self.test_time_flip = bool(config.get('test_time_flip', True))
        self.test_time_scales = _parse_test_time_scales(
            config.get('test_time_scales', [1.0])
        )
        self.image_aux_loss_weight = float(
            config.get('image_aux_loss_weight', 0.2)
        )
        self.frequency_aux_loss_weight = float(
            config.get('frequency_aux_loss_weight', 0.3)
        )
        self.consistency_loss_weight = float(
            config.get('consistency_loss_weight', 0.0)
        )
        self.consistency_temperature = float(
            config.get('consistency_temperature', 2.0)
        )
        # 小权重成对排序辅助 CE；只对二分类硬标签启用。
        self.ranking_loss_weight = float(
            config.get('ranking_loss_weight', 0.1 if self.num_classes == 2 else 0.0)
        )
        self.ranking_temperature = float(config.get('ranking_temperature', 1.0))
        # DWT 改变了后段输入分布；默认只冻结加载过预训练权重的浅层 BN。
        self.freeze_backbone_bn = bool(config.get('freeze_backbone_bn', False))
        self.compensate_dwt_stride = bool(config.get('compensate_dwt_stride', True))
        if self.num_classes < 2:
            raise ValueError('num_classes must be at least 2')
        if not 0.0 <= self.label_smoothing <= 1.0:
            raise ValueError('label_smoothing must be in [0, 1]')
        if not 0.0 <= self.image_logit_weight <= 1.0:
            raise ValueError('image_logit_weight must be in [0, 1]')
        if not 0.0 <= self.branch_dropout_prob < 1.0:
            raise ValueError('branch_dropout_prob must be in [0, 1)')
        if (
            self.branch_dropout_prob > 0
            and self.image_logit_weight in (0.0, 1.0)
        ):
            raise ValueError(
                'branch_dropout_prob requires both classifier branches'
            )
        if not all(math.isfinite(weight) and weight >= 0 for weight in (
            self.image_aux_loss_weight,
            self.frequency_aux_loss_weight,
            self.consistency_loss_weight,
            self.ranking_loss_weight,
        )):
            raise ValueError('Auxiliary loss weights must be finite and non-negative')
        if not math.isfinite(self.consistency_temperature) or self.consistency_temperature <= 0:
            raise ValueError('consistency_temperature must be finite and positive')
        if not math.isfinite(self.ranking_temperature) or self.ranking_temperature <= 0:
            raise ValueError('ranking_temperature must be finite and positive')
        if self.ranking_loss_weight > 0 and self.num_classes != 2:
            raise ValueError('ranking_loss_weight requires num_classes=2')

        wavelet_dim = int(config.get('wavelet_dim', 64))
        attention_heads = int(config.get('attention_heads', 4))
        window_size = int(config.get('wavelet_window_size', 7))
        frequency_dim = int(config.get('frequency_dim', 128))
        fusion_grid_size = int(config.get('fusion_grid_size', 7))
        attention_dropout = float(config.get('attention_dropout', 0.1))
        frequency_map_dropout = float(
            config.get('frequency_map_dropout', 0.1)
        )
        if wavelet_dim < 2 or frequency_dim < 2:
            raise ValueError('Feature dimensions must be at least 2')
        if attention_heads <= 0:
            raise ValueError('attention_heads must be positive')
        if wavelet_dim % attention_heads != 0:
            raise ValueError('wavelet_dim must be divisible by attention_heads')
        if frequency_dim % attention_heads != 0:
            raise ValueError('frequency_dim must be divisible by attention_heads')
        if window_size <= 0 or fusion_grid_size <= 0:
            raise ValueError('Attention spatial sizes must be positive')
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError('attention_dropout must be in [0, 1)')
        if not 0.0 <= frequency_map_dropout < 1.0:
            raise ValueError('frequency_map_dropout must be in [0, 1)')

        style_prob = float(config.get('ll_style_prob', 0.0))
        style_alpha = float(config.get('ll_style_alpha', 0.3))
        style_strength = float(config.get('ll_style_strength', 0.25))
        if not 0.0 <= style_prob <= 1.0 or not 0.0 <= style_strength <= 1.0:
            raise ValueError('LL style probability and strength must be in [0, 1]')
        if not math.isfinite(style_alpha) or style_alpha <= 0:
            raise ValueError('ll_style_alpha must be finite and positive')
        band_aug_prob = float(config.get('wavelet_band_aug_prob', 0.5))
        band_min_scale = float(config.get('wavelet_band_min_scale', 0.5))
        band_spatial_grid = int(config.get('wavelet_band_spatial_grid', 4))
        if not 0.0 <= band_aug_prob <= 1.0:
            raise ValueError('wavelet_band_aug_prob must be in [0, 1]')
        if not 0.0 < band_min_scale <= 1.0:
            raise ValueError('wavelet_band_min_scale must be in (0, 1]')
        if band_spatial_grid < 1:
            raise ValueError('wavelet_band_spatial_grid must be positive')
        quality_aug_prob = float(config.get('quality_aug_prob', 0.5))
        quality_resize_min = float(config.get('quality_resize_min', 0.65))
        quality_blur_prob = float(config.get('quality_blur_prob', 0.3))
        quality_noise_prob = float(config.get('quality_noise_prob', 0.3))
        quality_noise_std = float(config.get('quality_noise_std', 0.03))
        if not 0.0 <= quality_aug_prob <= 1.0:
            raise ValueError('quality_aug_prob must be in [0, 1]')
        if not 0.0 < quality_resize_min <= 1.0:
            raise ValueError('quality_resize_min must be in (0, 1]')
        if not 0.0 <= quality_blur_prob <= 1.0:
            raise ValueError('quality_blur_prob must be in [0, 1]')
        if not 0.0 <= quality_noise_prob <= 1.0:
            raise ValueError('quality_noise_prob must be in [0, 1]')
        if not 0.0 <= quality_noise_std <= 1.0:
            raise ValueError('quality_noise_std must be in [0, 1]')
        erase_prob = float(config.get('quality_erase_prob', 0.25))
        erase_max_area = float(config.get('quality_erase_max_area', 0.15))
        if not 0.0 <= erase_prob <= 1.0:
            raise ValueError('quality_erase_prob must be in [0, 1]')
        if not 0.0 <= erase_max_area < 1.0:
            raise ValueError('quality_erase_max_area must be in [0, 1)')
        attention_drop_path = float(config.get('attention_drop_path', 0.1))
        if not 0.0 <= attention_drop_path < 1.0:
            raise ValueError('attention_drop_path must be in [0, 1)')
        deform_max_offset = float(config.get('deform_max_offset', 2.0))
        if not math.isfinite(deform_max_offset) or deform_max_offset <= 0:
            raise ValueError('deform_max_offset must be finite and positive')
        bn_momentum = float(config.get('backbone_bn_momentum', 0.01))
        if not 0.0 < bn_momentum <= 1.0:
            raise ValueError('backbone_bn_momentum must be in (0, 1]')

        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.freeze_shallow_bn = bool(
            config.get('freeze_shallow_bn', self._backbone_pretrained)
        )
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.momentum = bn_momentum

        self.quality_augmentation = StochasticQualityAugmentation(
            probability=quality_aug_prob,
            resize_min=quality_resize_min,
            blur_probability=quality_blur_prob,
            noise_probability=quality_noise_prob,
            noise_std=quality_noise_std,
            erase_probability=erase_prob,
            erase_max_area=erase_max_area,
        )
        stage_channels = _efficientnet_b5_stage_channels(self.backbone)
        self.wavelet_modules = nn.ModuleList(
            [
                AttentionDWTModule(
                    channels=channels,
                    hidden_channels=wavelet_dim,
                    num_heads=attention_heads,
                    window_size=window_size,
                    attention_dropout=attention_dropout,
                    map_dropout=frequency_map_dropout,
                    style_prob=style_prob if index == 0 else 0.0,
                    style_alpha=style_alpha,
                    style_strength=style_strength,
                    band_aug_prob=band_aug_prob,
                    band_min_scale=band_min_scale,
                    band_spatial_grid=band_spatial_grid,
                    drop_path=attention_drop_path,
                )
                for index, channels in enumerate(stage_channels)
            ]
        )
        self.frequency_fusion = MultiScaleBidirectionalAttention(
            input_channels=[wavelet_dim] * 3,
            embed_dim=frequency_dim,
            num_heads=attention_heads,
            grid_size=fusion_grid_size,
            attention_dropout=attention_dropout,
            deform_max_offset=deform_max_offset,
            drop_path=attention_drop_path,
        )
        self.frequency_classifier = nn.Sequential(
            nn.LayerNorm(frequency_dim),
            nn.Dropout(p=float(config.get('frequency_dropout', 0.3))),
            nn.Linear(frequency_dim, self.num_classes),
        )

        image_feature_dim = self.backbone.classifier[-1].in_features
        self.image_feature_dim = int(image_feature_dim)
        self.frequency_feature_dim = int(frequency_dim)
        self.image_feature_norm = nn.LayerNorm(self.image_feature_dim)
        # 初始化后直接 forward 也必须应用相同的 BN 策略。
        self.train(self.training)

    def build_backbone(self, config):
        """Build the EfficientNet-B5 image encoder and image classifier."""
        use_pretrained = bool(config.get('use_torchvision_pretrained', True))
        weights = None
        self._backbone_pretrained = False
        if use_pretrained and _TV_HAS_WEIGHTS_ENUM:
            weights = EfficientNet_B5_Weights.IMAGENET1K_V1

        try:
            backbone = efficientnet_b5(weights=weights)
            self._backbone_pretrained = weights is not None
            logger.info(
                'Loaded EfficientNet-B5 with %s weights',
                'ImageNet' if weights is not None else 'randomly initialized',
            )
        except Exception as exc:
            logger.warning(
                'Failed to load EfficientNet-B5 pretrained weights (%s); '
                'falling back to random initialization',
                exc,
            )
            backbone = efficientnet_b5(weights=None)

        if self.compensate_dwt_stride:
            _compensate_dwt_downsampling(backbone)

        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=float(config.get('head_dropout', 0.5)), inplace=False),
            nn.Linear(in_features, self.num_classes),
        )
        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        """Return the concatenated image-domain and frequency-domain features."""
        image = self.quality_augmentation(data_dict['image'])
        image_feature, frequency_feature = self._extract_features(image)
        return torch.cat([image_feature, frequency_feature], dim=1)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """Fuse the decisions of the image and frequency classifiers."""
        _, _, fused_logits = self._classify_branches(features)
        return fused_logits

    def _classify_branches(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_dim = self.image_feature_dim + self.frequency_feature_dim
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(
                'Expected features with shape [B, {}], got {}'.format(
                    expected_dim, tuple(features.shape)
                )
            )

        image_feature, frequency_feature = torch.split(
            features,
            [self.image_feature_dim, self.frequency_feature_dim],
            dim=1,
        )
        image_logits = self.backbone.classifier(
            self.image_feature_norm(image_feature)
        )
        frequency_logits = self.frequency_classifier(frequency_feature)
        fused_logits = _fuse_classifier_logits(
            image_logits,
            frequency_logits,
            image_weight=self.image_logit_weight,
            branch_dropout_prob=(
                self.branch_dropout_prob if self.training else 0.0
            ),
        )
        return image_logits, frequency_logits, fused_logits

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        fused_loss = self._classification_loss(pred_dict['cls'], label)
        image_loss = self._classification_loss(
            pred_dict['image_cls'],
            label,
        )
        frequency_loss = self._classification_loss(
            pred_dict['frequency_cls'],
            label,
        )
        consistency_loss = fused_loss.new_zeros(())
        if self.consistency_loss_weight > 0:
            consistency_loss = _symmetric_kl_divergence(
                pred_dict['image_cls'],
                pred_dict['frequency_cls'],
                temperature=self.consistency_temperature,
            )
        ranking_loss = fused_loss.new_zeros(())
        if self.training and self.ranking_loss_weight > 0:
            # 排序对象使用推理时的固定融合，避免分支丢弃让不同样本不可比。
            ranking_logits = _fuse_classifier_logits(
                pred_dict['image_cls'].float(),
                pred_dict['frequency_cls'].float(),
                image_weight=self.image_logit_weight,
                branch_dropout_prob=0.0,
            )
            ranking_loss = _binary_pairwise_ranking_loss(
                ranking_logits, label, self.ranking_temperature
            )
        overall = (
            fused_loss
            + self.image_aux_loss_weight * image_loss
            + self.frequency_aux_loss_weight * frequency_loss
            + self.consistency_loss_weight * consistency_loss
            + self.ranking_loss_weight * ranking_loss
        )
        return {
            'overall': overall,
            'cls': fused_loss,
            'image_aux': image_loss,
            'frequency_aux': frequency_loss,
            'consistency': consistency_loss,
            'ranking': ranking_loss,
        }

    def _classification_loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        # 避免半精度 softmax/log 运算使辅助损失出现非有限值。
        pred = pred.float()
        if self.label_smoothing > 0 and label.dtype == torch.long:
            return F.cross_entropy(
                pred,
                label,
                label_smoothing=self.label_smoothing,
            )
        return self.loss_func(pred, label)

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(),
            pred.detach(),
        )
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        _ = inference
        if self.training or not self._test_time_augmented():
            features = self.features(data_dict)
            image_logits, frequency_logits, pred = self._classify_branches(
                features
            )
        else:
            features, image_logits, frequency_logits, pred = (
                self._forward_test_time_ensemble(data_dict)
            )
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {
            'cls': pred,
            'prob': prob,
            'feat': features,
            'image_cls': image_logits,
            'frequency_cls': frequency_logits,
        }

    def _test_time_augmented(self) -> bool:
        """推理时是否需要多视图集成。"""
        return self.test_time_flip or self.test_time_scales != [1.0]

    def _forward_test_time_ensemble(
        self,
        data_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """平均原图/水平翻转与多分辨率视图的分支特征和 logits。"""
        base_image = data_dict['image']
        original_size = base_image.shape[-2:]
        feature_sum = None
        image_logit_sum = None
        frequency_logit_sum = None
        pred_sum = None
        view_count = 0
        flips = (False, True) if self.test_time_flip else (False,)
        for scale in self.test_time_scales:
            scaled_image = _resize_keep_range(base_image, scale, original_size)
            for do_flip in flips:
                view_image = (
                    torch.flip(scaled_image, dims=(-1,))
                    if do_flip
                    else scaled_image
                )
                view_data = dict(data_dict)
                view_data['image'] = view_image
                view_features = self.features(view_data)
                (
                    view_image_logits,
                    view_frequency_logits,
                    view_pred,
                ) = self._classify_branches(view_features)
                if feature_sum is None:
                    feature_sum = view_features
                    image_logit_sum = view_image_logits
                    frequency_logit_sum = view_frequency_logits
                    pred_sum = view_pred
                else:
                    feature_sum = feature_sum + view_features
                    image_logit_sum = image_logit_sum + view_image_logits
                    frequency_logit_sum = (
                        frequency_logit_sum + view_frequency_logits
                    )
                    pred_sum = pred_sum + view_pred
                view_count += 1
        scale_factor = 1.0 / view_count
        return (
            feature_sum * scale_factor,
            image_logit_sum * scale_factor,
            frequency_logit_sum * scale_factor,
            pred_sum * scale_factor,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            frozen_stages = (
                self.backbone.features
                if self.freeze_backbone_bn
                else self.backbone.features[:3] if self.freeze_shallow_bn else []
            )
            for stage in frozen_stages:
                for module in stage.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        return self

    def _extract_features(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the three image encoders and their corresponding DWT modules."""
        x = image
        frequency_maps = []

        # EfficientNet-B5 features[0:3]: stem + shallow MBConv stages.
        for index in range(0, 3):
            x = self.backbone.features[index](x)
        x, frequency_map = self.wavelet_modules[0](x)
        frequency_maps.append(frequency_map)

        # EfficientNet-B5 features[3:5]: middle MBConv stages.
        for index in range(3, 5):
            x = self.backbone.features[index](x)
        x, frequency_map = self.wavelet_modules[1](x)
        frequency_maps.append(frequency_map)

        # EfficientNet-B5 features[5:9]: final MBConv stages and final 1x1 conv.
        for index in range(5, len(self.backbone.features)):
            x = self.backbone.features[index](x)
        x, frequency_map = self.wavelet_modules[2](x)
        frequency_maps.append(frequency_map)

        image_feature = self.backbone.avgpool(x)
        image_feature = torch.flatten(image_feature, 1)
        frequency_feature = self.frequency_fusion(frequency_maps)
        return image_feature, frequency_feature


class StochasticQualityAugmentation(nn.Module):
    """训练期模拟质量与遮挡变化，不修改标签；噪声可能略超原始数值范围。"""

    def __init__(
        self,
        probability: float,
        resize_min: float,
        blur_probability: float,
        noise_probability: float,
        noise_std: float,
        erase_probability: float = 0.0,
        erase_max_area: float = 0.0,
    ):
        super().__init__()
        self.probability = probability
        self.resize_min = resize_min
        self.blur_probability = blur_probability
        self.noise_probability = noise_probability
        self.noise_std = noise_std
        self.erase_probability = erase_probability
        self.erase_max_area = erase_max_area

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return image
        if self.probability <= 0 and self.erase_probability <= 0:
            return image

        batch, _, height, width = image.shape
        values = image.float()
        output = values
        if self.probability > 0:
            output = self._degrade(values, batch, height, width, image.device)
        if self.erase_probability > 0 and self.erase_max_area > 0:
            output = self._erase(output, batch, height, width)
        return output.to(image.dtype)

    def _degrade(
        self,
        values: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        selected = (
            torch.rand(batch, 1, 1, 1, device=device)
            < self.probability
        )
        if not bool(selected.any()):
            return values

        scale = float(
            torch.empty((), device=device).uniform_(
                self.resize_min,
                1.0,
            )
        )
        resized_height = max(2, int(round(height * scale)))
        resized_width = max(2, int(round(width * scale)))
        degraded = F.interpolate(
            values,
            size=(resized_height, resized_width),
            mode='bilinear',
            align_corners=False,
        )
        degraded = F.interpolate(
            degraded,
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )

        if self.blur_probability > 0:
            blurred = F.avg_pool2d(
                degraded,
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=False,
            )
            blur_mask = selected & (
                torch.rand(batch, 1, 1, 1, device=device)
                < self.blur_probability
            )
            degraded = torch.where(blur_mask, blurred, degraded)

        if self.noise_probability > 0 and self.noise_std > 0:
            sample_std = values.flatten(1).std(
                dim=1,
                unbiased=False,
            ).view(batch, 1, 1, 1)
            noise_scale = (
                torch.rand(batch, 1, 1, 1, device=device)
                * self.noise_std
                * sample_std.clamp_min(1e-6)
            )
            noise_mask = selected & (
                torch.rand(batch, 1, 1, 1, device=device)
                < self.noise_probability
            )
            noisy = degraded + torch.randn_like(degraded) * noise_scale
            degraded = torch.where(noise_mask, noisy, degraded)

        return torch.where(selected, degraded, values)

    def _erase(
        self,
        values: torch.Tensor,
        batch: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """随机遮挡一个矩形区域，填充逐样本均值以保持输入范围，避免依赖局部线索。"""
        if int(height * width * self.erase_max_area) < 1:
            return values
        device = values.device
        erase_mask = (
            torch.rand(batch, 1, 1, 1, device=device) < self.erase_probability
        )
        if not bool(erase_mask.any()):
            return values

        area_ratio = torch.empty(batch, device=device).uniform_(
            min(0.02, self.erase_max_area),
            self.erase_max_area,
        )
        aspect = torch.empty(batch, device=device).uniform_(0.5, 2.0)
        region_area = (area_ratio * (height * width)).floor().clamp(min=1.0)
        region_h = (region_area * aspect).sqrt().floor().clamp(min=1, max=height)
        region_h = torch.minimum(region_h, region_area)
        region_w = (region_area / region_h).floor().clamp(min=1, max=width)
        region_h, region_w = region_h.long(), region_w.long()

        rows = torch.arange(height, device=device).view(1, 1, height, 1)
        cols = torch.arange(width, device=device).view(1, 1, 1, width)
        top = (
            torch.rand(batch, device=device)
            * (height - region_h + 1).clamp(min=1).float()
        ).long()
        left = (
            torch.rand(batch, device=device)
            * (width - region_w + 1).clamp(min=1).float()
        ).long()
        top4 = top.view(batch, 1, 1, 1)
        left4 = left.view(batch, 1, 1, 1)
        bottom4 = (top + region_h).view(batch, 1, 1, 1)
        right4 = (left + region_w).view(batch, 1, 1, 1)
        region = (
            (rows >= top4)
            & (rows < bottom4)
            & (cols >= left4)
            & (cols < right4)
        )
        region = region & erase_mask
        fill = values.mean(dim=(-2, -1), keepdim=True)
        return torch.where(region, fill, values)


class HaarDWT2d(nn.Module):
    """Parameter-free 2D Haar wavelet decomposition."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class ConvNormAct(nn.Sequential):
    """Convolution with domain-independent group normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class DropPath(nn.Module):
    """随机深度：训练时按样本概率丢弃残差分支，推理时恒等，无新增参数。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError('drop_prob must be in [0, 1)')
        self.drop_prob = float(drop_prob)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0:
            return residual
        keep_prob = 1.0 - self.drop_prob
        shape = (residual.shape[0],) + (1,) * (residual.ndim - 1)
        mask = residual.new_empty(shape).bernoulli_(keep_prob)
        # 保持期望不变，避免训练/推理的残差尺度偏移。
        return residual / keep_prob * mask


class HourglassAttention(nn.Module):
    """Hourglass-shaped attention used to refine the HH sub-band."""

    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = max(channels // 2, 8)
        self.down = ConvNormAct(
            channels,
            hidden_channels,
            kernel_size=3,
            stride=2,
        )
        self.bottleneck = nn.Sequential(
            ConvNormAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
            ),
            ConvNormAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
            ),
        )
        self.attention = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_resolution = self.bottleneck(self.down(x))
        attention = F.interpolate(
            self.attention(low_resolution),
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return x * torch.sigmoid(attention)


class WindowCrossAttention(nn.Module):
    """Memory-bounded cross-attention between equally sized feature maps."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if num_heads <= 0 or channels < 2 or channels % num_heads != 0:
            raise ValueError('channels must be divisible by num_heads')
        if window_size <= 0:
            raise ValueError('window_size must be positive')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('attention dropout must be in [0, 1)')
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = float(dropout)

        self.query_norm = nn.GroupNorm(1, channels)
        self.context_norm = nn.GroupNorm(1, channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        query_map: torch.Tensor,
        context_map: torch.Tensor,
    ) -> torch.Tensor:
        if query_map.shape != context_map.shape:
            raise ValueError('Window cross-attention inputs must have equal shapes')

        query_windows, shape_info, valid = self._partition(self.query_norm(query_map))
        context_windows, _, _ = self._partition(self.context_norm(context_map))
        batch_windows, token_count, _ = query_windows.shape

        query = self.query(query_windows).reshape(
            batch_windows,
            token_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.key(context_windows).reshape(
            batch_windows,
            token_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.value(context_windows).reshape(
            batch_windows,
            token_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 禁止 padding token 作为 K/V，避免边界重复值改变注意力分布。
        # 关闭局部 autocast，确保大幅值 Q/K 的点积和 softmax 在 float32 中计算。
        with torch.autocast(device_type=query.device.type, enabled=False):
            attention = torch.matmul(query.float(), key.float().transpose(-2, -1))
            attention = attention * self.scale
            attention = attention.masked_fill(~valid[:, None, None, :], float('-inf'))
            attention = torch.softmax(attention, dim=-1).to(value.dtype)
        attention = F.dropout(
            attention,
            p=self.dropout,
            training=self.training,
        )
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(
            batch_windows,
            token_count,
            self.channels,
        )
        attended = self.output(attended)
        attended = self._reverse(attended, shape_info)
        return query_map + self.drop_path(
            torch.tanh(self.residual_scale) * attended
        )

    def _partition(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, ...], torch.Tensor]:
        batch, channels, height, width = x.shape
        window_h = min(self.window_size, height)
        window_w = min(self.window_size, width)
        pad_h = (-height) % window_h
        pad_w = (-width) % window_w
        valid = torch.ones(1, 1, height, width, device=x.device, dtype=torch.bool)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            valid = F.pad(valid, (0, pad_w, 0, pad_h), value=False)

        padded_h, padded_w = x.shape[-2:]
        windows_h = padded_h // window_h
        windows_w = padded_w // window_w
        x = x.view(
            batch,
            channels,
            windows_h,
            window_h,
            windows_w,
            window_w,
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(batch * windows_h * windows_w, window_h * window_w, channels)
        valid = valid.view(1, 1, windows_h, window_h, windows_w, window_w)
        valid = valid.permute(0, 2, 4, 3, 5, 1).reshape(
            windows_h * windows_w, window_h * window_w
        ).repeat(batch, 1)
        shape_info = (
            batch,
            channels,
            height,
            width,
            padded_h,
            padded_w,
            windows_h,
            windows_w,
            window_h,
            window_w,
        )
        return x, shape_info, valid

    def _reverse(
        self,
        windows: torch.Tensor,
        shape_info: Tuple[int, ...],
    ) -> torch.Tensor:
        (
            batch,
            channels,
            height,
            width,
            padded_h,
            padded_w,
            windows_h,
            windows_w,
            window_h,
            window_w,
        ) = shape_info
        x = windows.view(
            batch,
            windows_h,
            windows_w,
            window_h,
            window_w,
            channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch, channels, padded_h, padded_w)
        return x[:, :, :height, :width]


class BidirectionalWaveletCrossAttention(nn.Module):
    """HL and LH query each other in both directions."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.hl_from_lh = WindowCrossAttention(
            channels,
            num_heads,
            window_size,
            dropout,
            drop_path,
        )
        self.lh_from_hl = WindowCrossAttention(
            channels,
            num_heads,
            window_size,
            dropout,
            drop_path,
        )

    def forward(
        self,
        hl: torch.Tensor,
        lh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hl_from_lh(hl, lh), self.lh_from_hl(lh, hl)


class AttentionDWTModule(nn.Module):
    """Route HH to LL and route HL/LH exclusively to frequency fusion."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        num_heads: int,
        window_size: int,
        attention_dropout: float,
        map_dropout: float,
        style_prob: float = 0.0,
        style_alpha: float = 0.3,
        style_strength: float = 0.25,
        band_aug_prob: float = 0.0,
        band_min_scale: float = 0.5,
        drop_path: float = 0.0,
        band_spatial_grid: int = 4,
    ):
        super().__init__()
        self.dwt = HaarDWT2d()

        self.lh_projection = ConvNormAct(channels, hidden_channels)
        self.hl_projection = ConvNormAct(channels, hidden_channels)
        self.hh_projection = ConvNormAct(channels, hidden_channels)

        self.directional_attention = BidirectionalWaveletCrossAttention(
            hidden_channels,
            num_heads,
            window_size,
            attention_dropout,
            drop_path,
        )
        self.hourglass_attention = HourglassAttention(hidden_channels)
        self.frequency_map = ConvNormAct(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
        )

        self.high_frequency_supplement = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.map_dropout = nn.Dropout2d(p=map_dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.style_prob = style_prob
        self.style_alpha = style_alpha
        self.style_strength = style_strength
        self.band_aug_prob = band_aug_prob
        self.band_min_scale = band_min_scale
        self.band_spatial_grid = band_spatial_grid

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ll, lh, hl, hh = self.dwt(x)
        if self.training:
            lh, hl, hh = _attenuate_wavelet_bands(
                (lh, hl, hh),
                probability=self.band_aug_prob,
                minimum_scale=self.band_min_scale,
                spatial_grid_size=self.band_spatial_grid,
            )

        # HL/LH 仅构造供最终三级融合使用的频率图。
        lh_hidden = self.lh_projection(lh)
        hl_hidden = self.hl_projection(hl)
        hl_hidden, lh_hidden = self.directional_attention(
            hl_hidden,
            lh_hidden,
        )
        frequency_map = self.map_dropout(
            self.frequency_map(hl_hidden + lh_hidden)
        )

        # 图像主路径只允许 HH 生成的高频补充图与 LL 融合。
        hh_hidden = self.hourglass_attention(self.hh_projection(hh))
        high_frequency_map = self.high_frequency_supplement(hh_hidden)
        if self.training:
            ll = _mix_low_frequency_style(
                ll, self.style_prob, self.style_alpha, self.style_strength
            )
        next_image_feature = (
            ll + torch.tanh(self.residual_scale) * high_frequency_map
        )
        return next_image_feature, frequency_map


class TokenAttentionBlock(nn.Module):
    """Pre-norm token attention used by the cross-scale fusion module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.attention_drop_path = DropPath(drop_path)
        self.ffn_drop_path = DropPath(drop_path)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_context = self.context_norm(context)
        attended, _ = self.attention(
            normalized_query,
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        x = query + self.attention_drop_path(self.attention_dropout(attended))
        return x + self.ffn_drop_path(self.ffn(self.ffn_norm(x)))


class BidirectionalTokenCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.first_from_second = TokenAttentionBlock(
            embed_dim,
            num_heads,
            dropout,
            drop_path,
        )
        self.second_from_first = TokenAttentionBlock(
            embed_dim,
            num_heads,
            dropout,
            drop_path,
        )

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.first_from_second(first, second),
            self.second_from_first(second, first),
        )


class DeformableFusionConv(nn.Module):
    """3x3 deformable convolution with a standard-convolution fallback."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_offset: float,
    ):
        super().__init__()
        self.max_offset = max_offset
        if DeformConv2d is not None:
            self.offset = nn.Conv2d(
                in_channels,
                18,
                kernel_size=3,
                padding=1,
            )
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            self.deform = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.fallback = None
        else:
            self.offset = None
            self.deform = None
            self.fallback = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            logger.warning(
                'torchvision.ops.DeformConv2d is unavailable; using Conv2d fallback'
            )

        self.norm = nn.GroupNorm(
            _group_count(out_channels),
            out_channels,
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deform is not None:
            offset = torch.tanh(self.offset(x)) * self.max_offset
            x = self.deform(x, offset)
        else:
            x = self.fallback(x)
        return self.activation(self.norm(x))


class MultiScaleBidirectionalAttention(nn.Module):
    """Fuse the three DWT maps in the exact order shown in the design."""

    def __init__(
        self,
        input_channels: Sequence[int],
        embed_dim: int,
        num_heads: int,
        grid_size: int,
        attention_dropout: float,
        deform_max_offset: float,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if len(input_channels) != 3:
            raise ValueError('Exactly three frequency maps are required')
        if embed_dim % num_heads != 0:
            raise ValueError('frequency_dim must be divisible by attention_heads')

        self.grid_size = int(grid_size)
        self.projections = nn.ModuleList(
            [ConvNormAct(channels, embed_dim) for channels in input_channels]
        )
        self.position_encoders = nn.ModuleList(
            [
                nn.Conv2d(
                    embed_dim,
                    embed_dim,
                    kernel_size=3,
                    padding=1,
                    groups=embed_dim,
                    bias=False,
                )
                for _ in input_channels
            ]
        )
        for position_encoder in self.position_encoders:
            nn.init.zeros_(position_encoder.weight)

        # First row in the diagram: F1 <-> F2, while F3 uses self-attention.
        self.first_cross = BidirectionalTokenCrossAttention(
            embed_dim,
            num_heads,
            attention_dropout,
            drop_path,
        )
        self.third_self = TokenAttentionBlock(
            embed_dim,
            num_heads,
            attention_dropout,
            drop_path,
        )

        # Second row in the diagram: F1 uses self-attention, F2 <-> F3.
        self.first_self = TokenAttentionBlock(
            embed_dim,
            num_heads,
            attention_dropout,
            drop_path,
        )
        self.second_cross = BidirectionalTokenCrossAttention(
            embed_dim,
            num_heads,
            attention_dropout,
            drop_path,
        )

        self.deformable_conv = DeformableFusionConv(
            embed_dim * 3,
            embed_dim,
            max_offset=deform_max_offset,
        )
        self.output_dim = embed_dim

    def forward(self, frequency_maps: List[torch.Tensor]) -> torch.Tensor:
        if len(frequency_maps) != 3:
            raise ValueError('Exactly three frequency maps are required')

        maps = []
        spatial_sizes = []
        for projection, position_encoder, frequency_map in zip(
            self.projections,
            self.position_encoders,
            frequency_maps,
        ):
            projected = projection(frequency_map)
            target_size = (
                min(projected.shape[-2], self.grid_size),
                min(projected.shape[-1], self.grid_size),
            )
            if projected.shape[-2:] != target_size:
                projected = F.adaptive_avg_pool2d(projected, target_size)
            projected = projected + position_encoder(projected)
            maps.append(projected)
            spatial_sizes.append(target_size)

        tokens = [self._to_tokens(feature_map) for feature_map in maps]
        first, second = self.first_cross(tokens[0], tokens[1])
        third = self.third_self(tokens[2], tokens[2])
        first = self.first_self(first, first)
        second, third = self.second_cross(second, third)

        attended_maps = [
            self._to_map(first, spatial_sizes[0]),
            self._to_map(second, spatial_sizes[1]),
            self._to_map(third, spatial_sizes[2]),
        ]
        fusion_size = (
            max(size[0] for size in spatial_sizes),
            max(size[1] for size in spatial_sizes),
        )
        aligned_maps = [
            feature_map
            if feature_map.shape[-2:] == fusion_size
            else F.interpolate(
                feature_map,
                size=fusion_size,
                mode='bilinear',
                align_corners=False,
            )
            for feature_map in attended_maps
        ]
        fused_map = torch.cat(
            aligned_maps,
            dim=1,
        )
        fused_map = self.deformable_conv(fused_map)
        return F.adaptive_avg_pool2d(fused_map, 1).flatten(1)

    def _to_tokens(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(2).transpose(1, 2)

    def _to_map(
        self,
        tokens: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        batch, _, channels = tokens.shape
        return tokens.transpose(1, 2).reshape(
            batch,
            channels,
            spatial_size[0],
            spatial_size[1],
        )


def _resize_keep_range(
    image: torch.Tensor,
    scale: float,
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """按比例缩放再插值回原分辨率，保持输入尺寸与数值范围不变。"""
    if scale == 1.0:
        return image
    height, width = original_size
    resized_height = max(2, int(round(height * scale)))
    resized_width = max(2, int(round(width * scale)))
    scaled = F.interpolate(
        image.float(),
        size=(resized_height, resized_width),
        mode='bilinear',
        align_corners=False,
    )
    scaled = F.interpolate(
        scaled,
        size=(height, width),
        mode='bilinear',
        align_corners=False,
    )
    return scaled.to(image.dtype)


def _parse_test_time_scales(scales) -> List[float]:
    """解析并去重推理期多尺度视图，始终包含原尺度 1.0。"""
    if isinstance(scales, (int, float)):
        scales = [scales]
    parsed = []
    for value in scales:
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('test_time_scales must be finite and positive')
        if scale not in parsed:
            parsed.append(scale)
    if 1.0 not in parsed:
        parsed.insert(0, 1.0)
    return parsed


def _fuse_classifier_logits(
    image_logits: torch.Tensor,
    frequency_logits: torch.Tensor,
    image_weight: float,
    branch_dropout_prob: float,
) -> torch.Tensor:
    """训练时随机屏蔽一个分类分支，推理时恢复固定权重融合。"""
    if image_logits.shape != frequency_logits.shape:
        raise ValueError('Classifier logits must have equal shapes')
    if branch_dropout_prob <= 0:
        return (
            image_weight * image_logits
            + (1.0 - image_weight) * frequency_logits
        )

    batch = image_logits.shape[0]
    random_value = torch.rand(
        batch,
        1,
        device=image_logits.device,
    )
    drop_image = random_value < branch_dropout_prob * 0.5
    drop_frequency = (
        (random_value >= branch_dropout_prob * 0.5)
        & (random_value < branch_dropout_prob)
    )
    image_weights = (~drop_image).to(image_logits.dtype) * image_weight
    frequency_weights = (
        (~drop_frequency).to(frequency_logits.dtype)
        * (1.0 - image_weight)
    )
    weight_sum = image_weights + frequency_weights
    return (
        image_weights * image_logits
        + frequency_weights * frequency_logits
    ) / weight_sum


def _attenuate_wavelet_bands(
    bands: Tuple[torch.Tensor, ...],
    probability: float,
    minimum_scale: float,
    spatial_grid_size: int = 1,
) -> Tuple[torch.Tensor, ...]:
    """共享平滑空间衰减，保留逐位置子带比例；grid=1 复现旧标量扰动。"""
    if probability <= 0 or minimum_scale >= 1.0:
        return bands
    batch = bands[0].shape[0]
    height, width = bands[0].shape[-2:]
    grid_height = min(spatial_grid_size, height)
    grid_width = min(spatial_grid_size, width)
    selected = (
        torch.rand(batch, 1, 1, 1, device=bands[0].device)
        < probability
    )
    scale = torch.empty(
        batch,
        1,
        grid_height,
        grid_width,
        device=bands[0].device,
        dtype=torch.float32,
    ).uniform_(minimum_scale, 1.0)
    # 整图标量缩放会被后续 GroupNorm 近乎抵消，空间变化保留相对局部能量。
    # 该掩码不依赖子带内容，避免引入 HH/LL 到本级 HL/LH 的信息连接。
    if spatial_grid_size > 1 and (grid_height, grid_width) != (height, width):
        scale = F.interpolate(
            scale, size=(height, width), mode='bilinear', align_corners=False
        )
    scale = torch.where(selected, scale, torch.ones_like(scale))
    return tuple(band * scale.to(band.dtype) for band in bands)


def _binary_pairwise_ranking_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """用 Fake/Real 分数差的 softplus 近似排序错误，只使用当前 batch 的标签。"""
    # 先转 float32 再求差，避免 AMP 下两个有限半精度 logits 相减溢出。
    values = logits.float()
    zero = values.sum() * 0.0
    if (
        values.ndim != 2 or values.shape[1] != 2
        or label.ndim != 1 or label.shape[0] != values.shape[0]
        or label.dtype != torch.long
    ):
        # 多分类与 MixUp 等软标签沿用原分类损失，不做硬配对。
        return zero
    scores = values[:, 1] - values[:, 0]
    positive = scores[label == 1]
    negative = scores[label == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        # 单类别 batch 返回可反传的零，不产生 NaN，也不保留跨 batch 状态。
        return zero
    differences = positive[:, None] - negative[None, :]
    return F.softplus(-differences / temperature).mean() * temperature


def _compensate_dwt_downsampling(backbone: nn.Module) -> None:
    """保留 B5 全部层和权重形状，由三次 DWT 承担后段的三次降采样。"""
    for stage_index in (3, 4, 6):
        first_block = backbone.features[stage_index][0]
        downsample_convs = [
            layer for layer in first_block.modules()
            if isinstance(layer, nn.Conv2d) and layer.stride == (2, 2)
        ]
        if len(downsample_convs) != 1:
            raise ValueError(
                'Expected one stride-2 convolution at B5 stage {}'.format(stage_index)
            )
        downsample_convs[0].stride = (1, 1)
    logger.info('B5 stages 3/4/6 use stride 1; three DWT modules provide downsampling')


def _mix_low_frequency_style(
    ll: torch.Tensor,
    probability: float,
    alpha: float,
    strength: float,
) -> torch.Tensor:
    """仅扰动 LL 的样本级通道统计，保留归一化空间内容，不混合标签或高频子带。"""
    batch = ll.shape[0]
    if probability <= 0 or strength <= 0 or batch < 2 or ll.shape[-2] * ll.shape[-1] < 2:
        return ll

    values = ll.float()
    mean = values.mean(dim=(-2, -1), keepdim=True).detach()
    std = (values.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-6).sqrt().detach()
    permutation = torch.randperm(batch, device=ll.device)
    concentration = torch.full((batch, 1, 1, 1), alpha, device=ll.device)
    coefficient = torch.distributions.Beta(concentration, concentration).sample()
    selected = torch.rand(batch, 1, 1, 1, device=ll.device) < probability
    amount = (1.0 - coefficient) * strength * selected
    mixed_mean = mean + amount * (mean[permutation] - mean)
    mixed_std = std + amount * (std[permutation] - std)
    perturbed = (values - mean) / std * mixed_std + mixed_mean
    # 未选中的样本逐值保持不变，避免纯粹的数值往返扰动。
    return torch.where(selected, perturbed, values).to(ll.dtype)


def _symmetric_kl_divergence(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """可选的双向概率一致性约束；使用 float32 避免 AMP 下 KL 溢出。"""
    first_log_prob = F.log_softmax(first_logits.float() / temperature, dim=1)
    second_log_prob = F.log_softmax(second_logits.float() / temperature, dim=1)
    # log_target 避免概率下溢为 0 时，target 侧反向传播出现 log(0)。
    first_to_second = F.kl_div(
        first_log_prob,
        second_log_prob,
        reduction='batchmean',
        log_target=True,
    )
    second_to_first = F.kl_div(
        second_log_prob,
        first_log_prob,
        reduction='batchmean',
        log_target=True,
    )
    return 0.5 * (first_to_second + second_to_first) * temperature ** 2


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Choose the largest valid group count while retaining group statistics."""
    for groups in range(min(channels, max_groups), 0, -1):
        if channels % groups == 0 and channels // groups > 1:
            return groups
    return 1


def _efficientnet_b5_stage_channels(backbone: nn.Module) -> List[int]:
    """Read the output channels at the three encoder split points."""
    split_indices = (2, 4, len(backbone.features) - 1)
    channels = [
        _last_conv_out_channels(backbone.features[index])
        for index in split_indices
    ]
    if channels != [40, 128, 2048]:
        logger.info('Resolved EfficientNet-B5 stage channels as %s', channels)
    return channels


def _last_conv_out_channels(module: nn.Module) -> int:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            return int(child.out_channels)
    raise ValueError('Could not infer output channels from module {}'.format(module))
