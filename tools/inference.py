import warnings
import os
import argparse
import mmcv
import torch
import numpy as np
from pathlib import Path

from mmcv.ops import RoIPool, nms_rotated
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmrotate.models import build_detector
from mmrotate.core import imshow_det_rbboxes
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

def get_model(args):
    PALETTE = [[255, 0, 0], [0, 255, 0]]  # borrowed from cityscapes
    model = init_model(args.config, args.checkpoint, device='cuda')
    model.PALETTE = PALETTE
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def init_model(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a model from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        model = NPUDataParallel(model)
        model.cfg = config

    return model

def inference(model, imgs):
    """Inference image(s) with the model.

    Args:
        model (nn.Module): The loaded model.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results


def inference_sample(args):
    class_names = ['u_parking_lines', 'a_parking_lines']
    SEG_PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                   [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                   [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                   [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                   [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    model = get_model(args)

    img = mmcv.imread(args.image_path)
    result = inference(model, img)
    bbox_result = result[:2]
    seg_result = result[2]
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    # NMS among different classes
    bboxes_for_nms = torch.from_numpy(bboxes[:, :5])
    scores_for_nms = torch.from_numpy(bboxes[:, 5])
    _, keep = nms_rotated(bboxes_for_nms, scores_for_nms, iou_threshold=0.1)
    bboxes = bboxes[keep]
    labels = labels[keep]

    img_name = args.image_path.split('/')[-1].split('.')[0] + ".png"
    bin_name = args.image_path.split('/')[-1].split('.')[0] + ".bin"
    seg_name = args.image_path.split('/')[-1].split('.')[0] + "_seg.png"
    out_img_file = os.path.join(args.save_dir, img_name)
    out_bin_file = os.path.join(args.save_dir, bin_name)
    out_seg_file = os.path.join(args.save_dir, seg_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    bin_results = np.concatenate([bboxes, labels[:, None]], axis=-1)
    bin_results.tofile(out_bin_file)

    print('Prediction Saved!')
    # bin_results = np.fromfile(out_bin_file).reshape(-1, 7)

    if args.rgb:
        imshow_det_rbboxes(img, bboxes, labels, class_names=class_names, out_file=out_img_file,
                           bbox_color=model.PALETTE, text_color=model.PALETTE, thickness=5, font_size=26)
        model.show_result(img, seg_result, palette=SEG_PALETTE, out_file=out_seg_file, opacity=0.5)
        print('Visualization Results Saved!')
    else:
        mmcv.imwrite(seg_result, out_seg_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str,
                        default="configs/rotated_retinanet_obb_r50_fpn_1x_avm_le90.py")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/latest.pth")
    parser.add_argument("--image_path", type=str, default="demo.png")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--rgb", type=lambda x: x == "True", default="True", choices=["True", "False"])
    parser.add_argument("--shape", type=int, default=None, nargs='+', help="img_scale: [H x W]")

    args = parser.parse_args()

    inference_sample(args)
