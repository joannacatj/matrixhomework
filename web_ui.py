import numpy as np
import gradio as gr
from PIL import Image
import os, sys
import argparse
from pathlib import Path
from utils import util_image

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url


def gray(input_img):
    # 灰度值 = 0.2989 * R + 0.5870 * G + 0.1140 * B
    # image[..., :3]表示提取图像的前三个通道（即R、G、B通道）
    # 省略号可以在索引中表示对应维度的完整范围。
    gray = np.dot(input_img[..., :3], [0.2989, 0.5870, 0.1140])
    gray = gray.astype(np.uint8)  # 将灰度图像转换为无符号整型 ,如果不加一般会报错
    # pil_image = Image.fromarray(gray)  # 将灰度图像数组转换为PIL图像对象
    return gray


def get_configs(task='bicsrx4_opencv'):
    if task=='bicsrx4_opencv':
        configs = OmegaConf.load('./configs/bicubic_swinunet_bicubic256.yaml')

    # prepare the checkpoint
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    # ckpt_path = ckpt_dir / f'resshift_{args.task}_s{args.steps}.pth'
    ckpt_path="/home/t2f/ResShift_text/ResShift-master/saved_logs/2024-03-20-01-33/ckpts/model_260000.pth"
    vqgan_path = ckpt_dir / f'autoencoder/autoencoder_vq_f4.pth'

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = 15
    configs.diffusion.params.sf = 4
    configs.autoencoder.ckpt_path = str(vqgan_path)

    return configs

def predict(in_path,  seed=12345,task='bicsrx4_opencv',):
    configs = get_configs(task)
    resshift_sampler = ResShiftSampler(
        configs,
        chop_size=256,
        chop_stride=224,
        chop_bs=1,
        use_fp16=True,
        seed=seed,
    )
    out_dir = Path('restored_output')
    if not out_dir.exists():
        out_dir.mkdir()
    resshift_sampler.inference(in_path, out_dir, bs=1, noise_repeat=False)
    out_path = out_dir / f"{Path(in_path).stem}.png"
    assert out_path.exists(), 'Super-resolution failed!'
    im_sr = util_image.imread(out_path, chn="rgb", dtype="uint8")
    print(im_sr.shape)
    return im_sr
    


title="ResTFR"
demo = gr.Interface(predict,  
    inputs=[
        gr.Image(type="filepath", label="Input: Low Quality Image"),
        gr.Number(value=12345, precision=0, label="Ranom seed")
    ],
    outputs=[
            gr.Image(type="numpy", label="Output: High Quality Image",format="JPEG",width=256,height=256),
        ],
    # outputs="image",
    title=title,
    examples=[
        ['/home/t2f/ResShift_text/ResShift-master/testdata/bicubic_x4/699.png', 12345],
        ['/home/t2f/ResShift_text/ResShift-master/testdata/bicubic_x4/707.png', 12345],
        ['/home/t2f/ResShift_text/ResShift-master/testdata/bicubic_x4/794.png', 12345],
        ['/home/t2f/ResShift_text/ResShift-master/testdata/bicubic_x4/963.png',  12345],
        ['/home/t2f/ResShift_text/ResShift-master/testdata/bicubic_x4/4664.png', 12345],
    ])
# demo.launch(server_port=7862)
'''
如果需要在服务器部署后，局域网访问, 添加服务名 server_name 修改为：
'''
demo.queue().launch( server_name="0.0.0.0",server_port=7862,inbrowser=True)
# https://blog.csdn.net/xyl295528322/article/details/131995889
