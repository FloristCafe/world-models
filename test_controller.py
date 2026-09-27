""" Test controller """
import argparse
from os.path import join, exists
import gymnasium as gym  # 引入现代接口，用以创建带界面的环境
import pygame
from pygame import _sdl2
from utils.misc import RolloutGenerator
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
parser.add_argument('--steering-smoothing', type=float, default=0.6,
                    help='EMA coefficient for steering; 0 disables smoothing.')
args = parser.parse_args()

if not 0.0 <= args.steering_smoothing < 1.0:
    parser.error('--steering-smoothing must be in [0, 1).')

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

# 1. 正常实例化，底层会自动加载你 778 分的 best.tar 权重
generator = RolloutGenerator(args.logdir, device, 1000)

# 2. 劫持并替换环境（强行注入图形界面）
if hasattr(generator, 'env'):
    generator.env.close()  # 关掉原本无界面的静默环境
# 重新挂载带界面的 v2 环境
generator.env = gym.make('CarRacing-v3', render_mode='human')
generator.env.reset()
pygame.display.set_caption('World Models - CarRacing-v3')
window = _sdl2.Window.from_display_module()
window.focus()

with torch.no_grad():
    # 3. 开始让 Controller 接管方向盘
    print('Starting CarRacing-v3 human rendering...')
    generator.rollout(
        None,
        render=True,
        steering_smoothing=args.steering_smoothing,
    )
