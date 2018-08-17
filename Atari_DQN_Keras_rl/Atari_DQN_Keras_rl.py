from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
from gym import wrappers

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    # 状態への前処理 Memoryに格納される前に呼ばれる
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # 画像サイズ縮小・グレースケール化
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # メモリ節約のため

    # バッチ全体への前処理 学習直前に呼ばれる
    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    # 報酬への前処理 Memoryに格納される前に呼ばれる
    def process_reward(self, reward):
        return np.clip(reward, -1., 1.) #報酬のクリッピング

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Gymの環境宣言
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# モデル構築
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu", input_shape=(*INPUT_SHAPE, WINDOW_LENGTH)))
model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(nb_actions, activation="linear"))
print(model.summary())

# 
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# e-greedy epsが線形減少
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)


# エージェントの宣言
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # log と途中の重み出力
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]

    # 学習
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # 最終結果保存
    dqn.save_weights(weights_filename, overwrite=True)

    # 動画保存
    env = wrappers.Monitor(env, './movie_folder', video_callable=(lambda ep: True), force=True)

    # テスト実行
    dqn.test(env, nb_episodes=1, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
