import torch
import torch.nn.functional as F
import warnings

from ..builder import ROTATED_DETECTORS
from .rotated_retinanet import RotatedRetinaNet
import numpy as np
import mmcv
from mmrotate.core import rbbox2result

from mmseg.models import builder
from mmseg.ops import resize
from mmseg.core import add_prefix


@ROTATED_DETECTORS.register_module()
class RotatedSegRetinaNet(RotatedRetinaNet):
    """Implementation of Rotated `RetinaNet.` with Segmentor.
    """

    def __init__(self,
                 backbone,
                 neck,
                 decode_head,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedSegRetinaNet,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)
        self._init_decode_head(decode_head)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.decode_align_corners = self.decode_head.align_corners
        self.decode_out_channels = self.decode_head.out_channels
    
    def decode(self, feature, img_metas):
        """Decode into a semantic segmentation map of the same size as input."""
        out = self._decode_head_forward_test(feature, img_metas)
        out = resize(
            input=out,
            size=img_metas[0]['img_shape'][:2],
            mode='bilinear',
            align_corners=self.decode_align_corners)
        return out

    def _decode_head_forward_train(self, feature, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(feature, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        gt_semantic_seg = gt_semantic_seg.long()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        return losses

    def inference(self, feature, img_metas, rescale):
        """Inference with slide/whole style.

        Args:
            feature (Tensor): The feature map of input image.
            img_metas (list[dict]): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        ori_shape = img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_metas)
        seg_logit = self.decode(feature, img_metas)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = feature.shape[2:]
            else:
                # remove padding area
                resize_shape = img_metas[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_metas[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.decode_align_corners,
                warning=False)

        if self.decode_out_channels == 1:
            output = F.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)

        return output

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        seg_logit = self.inference(x, img_metas, rescale)
        if self.decode_out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)

        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        for i in range(len(bbox_results)):
            bbox_results[i].append(seg_pred[i])
        return bbox_results

    def show_result(self,
                    img,
                    result,
                    palette,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, list):
            seg = result[0]
        else:
            seg = result
        palette = np.array(palette)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img