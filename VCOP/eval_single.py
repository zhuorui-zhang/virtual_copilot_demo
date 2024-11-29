import torch
import torchaudio as ta
import numpy as np
import random
import os, sys
from otrans.model import Transformer
from otrans.recognizer import TransformerRecognizer
from otrans.data import load_vocab, spec_augment, normalization, concat_and_subsample, apply_cmvn
import kaldiio as kio
import yaml
import argparse

EOS = 0
BOS = 0
PAD = 1
UNK = 2
MASK = 2
unk = '<UNK>'
compute_fbank = ta.compliance.kaldi.fbank

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

def load_model_and_params(args):

    if args.ngpu == 0:
        checkpoint = torch.load(args.load_model, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.load_model)
    
    if 'params' in checkpoint:
        params = checkpoint['params']
    else:
        assert os.path.isfile(args.config), 'please specify a configure file.'
        with open(args.config, 'r') as f:
            params = yaml.load(f)
    # 加载预训练模型
    model = Transformer(params['model'])
    model.load_state_dict(checkpoint['model'])
    print('Load pre-trained model from %s' % args.load_model)
    model.eval()
    if args.ngpu > 0:
        model.cuda()
    return model, params

def process_audio(args, wav_file_path, params):
    if params['from_kaldi']:
        feature = kio.load_mat(wav_file_path)
    else:
        # 读取音频文件
        wavform, sample_frequency = ta.load(wav_file_path)
        wavform = wavform.float()
        # 提取fbank特征
        feature = compute_fbank(wavform, num_mel_bins=params['num_mel_bins'], sample_frequency=sample_frequency)
    targets_dict = {}
    unit2idx = load_vocab(params['vocab'])
    name = args.decode_set
    with open(os.path.join(params[name], params['text']), 'r', encoding='utf-8') as t:
        for line in t:
            parts = line.strip().split()
            utt_id = parts[0]
            label = []
            for c in parts[1:]:
                label.append(unit2idx[c] if c in unit2idx else unit2idx[unk])
            targets_dict[utt_id] = label

    # 应用CMVN (如果需要)
    if params['apply_cmvn']:
        utt2spk = {}
        with open(os.path.join(params[name], 'utt2spk'), 'r') as f:
            for line in f:
                utt_id, spk_id = line.strip().split()
                utt2spk[utt_id] = spk_id
            print('Load Speaker INFO')
        
        cmvns = {}
        with open(os.path.join(params[name], 'cmvn.scp'), 'r') as f:
            for line in f:
                spk_id, path = line.strip().split()
                cmvns[spk_id] = path
            print('Load CMVN Stats')
        spk_id = utt2spk[utt_id]  
        stats = kio.load_mat(cmvns[spk_id])
        feature = apply_cmvn(feature, stats)

    # 归一化特征
    if params['normalization']:
        feature = normalization(feature)
    
    # 进行spec-augment
    if params['spec_argument']:
        feature = spec_augment(feature)
    
    # 将特征按照需要的帧数进行裁剪
    left_frames = params["left_frames"]
    right_frames = params["right_frames"]
    skip_frames = params["skip_frames"]
    if left_frames > 0 or right_frames > 0:
        feature = concat_and_subsample(feature, left_frames=left_frames,
                                       right_frames=right_frames, skip_frames=skip_frames)

    feature_length = feature.shape[0]
    targets = targets_dict[utt_id]
    targets_length = len(targets)
    return feature, feature_length, targets, targets_length


def predict_audio(wav_file_path, params, args, model, unit2char):
    # 读取和预处理音频文件
    utt_id = os.path.basename(wav_file_path)
    feature, feature_length, target, target_length = process_audio(args, wav_file_path, params)
    # 将特征转化为Tensor
    # feature = torch.FloatTensor(feature).unsqueeze(0)  # 添加batch维度
    # feature_length = torch.IntTensor([feature_length])
    max_feature_length = feature_length
    max_target_length = target_length
    batch = [(utt_id, feature, feature_length, target, target_length)]
    padded_features = []
    padded_targets = []
    for _, feat, feat_len, target, target_len in batch:
        padded_features.append(np.pad(feat, ((
            0, max_feature_length-feat_len), (0, 0)), mode='constant', constant_values=0.0))
        padded_targets.append(
            [BOS] + target + [EOS] + [PAD] * (max_target_length - target_len))

    features = torch.FloatTensor(padded_features)
    features_length = torch.IntTensor(feature_length)
    # targets = torch.LongTensor(padded_targets)
    # targets_length = torch.IntTensor(target_length)

    if args.ngpu > 0:
        model.cuda()
        features = features.cuda()
        features_length = features_length.cuda()

    # 初始化预测器
    recognizer = TransformerRecognizer(model, unit2char=unit2char, beam_width=args.beam_width,
                                       max_len=args.max_len, penalty=args.penalty, lamda=args.lamda, ngpu=args.ngpu)
    # 进行预测
    pred_text = recognizer.recognize(features, features_length)[0]
    return pred_text

# 使用示例
def predict_text_from_audio(args, wav_file_path):
    set_seed(1234)
    # 加载模型&參數
    model, params = load_model_and_params(args)
    params["data"]["vocab"] = "VCOP/egs/aishell/data/vocab"
    params["data"]["train"] = "VCOP/egs/aishell/data/train"
    params["data"]["test"] = "VCOP/egs/aishell/data/test"
    params["data"]["dev"] = "VCOP/egs/aishell/data/dev"
    ### params:
    # {'data': {'name': 'aishell', 'vocab': 'egs/aishell/data/vocab', 
    #           'batch_size': 16, 'text': 'character', 'train': 'egs/aishell/data/train', 
    #           'test': 'egs/aishell/data/test', 'dev': 'egs/aishell/data/dev', 'short_first': False, 
    #           'num_mel_bins': 40, 'apply_cmvn': False, 'normalization': True, 
    #           'spec_argument': True, 'left_frames': 0, 'right_frames': 0, 'skip_frames': 0, 
    #           'from_kaldi': False, 'num_works': 4}, 
    #           'model': {'type': 'transformer', 
    #                     'd_model': 256, 'normalize_before': False, 
    #                     'concat_after': False, 'pos_dropout_rate': 0.1, 
    #                     'ffn_dropout_rate': 0.05, 'slf_attn_dropout_rate': 0.05, 
    #                     'src_attn_dropout_rate': 0.05, 'residual_dropout_rate': 0.1, 
    #                     'feat_dim': 40, 'num_enc_blocks': 6, 'enc_ffn_units': 1024, 
    #                     'enc_input_layer': 'conv2d', 'vocab_size': 99, 
    #                     'num_dec_blocks': 6, 'dec_ffn_units': 1024, 'n_heads': 4, 
    #                     'smoothing': 0.1, 'activation': 'glu', 'share_embedding': True}, 
    #                     'train': {'scheduler': 'stepwise', 'optimizer': 'adam', 'warmup_steps': 12000, 
    #                             'shuffle': True, 'lr': 1.0, 'clip_grad': 5, 'epochs': 60, 'accum_steps': 1, 
    #                             'grad_noise': False, 'load_model': False, 'save_name': 'transformer'}}
    
    # 加载词汇
    char2unit = load_vocab(params['data']['vocab'])
    unit2char = {i: c for c, i in char2unit.items()}
    # 进行音频文件的预测
    pred_text = predict_audio(wav_file_path, params["data"], args, model, unit2char)
    print("Predicted text: ", pred_text)
    return pred_text
    
if __name__ == '__main__':
    wav_file_path = "VCOP/data/2024-06-01T13-24-03.413453.wav"
    # wav_file_path = "out.wav"
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-n', '--ngpu', type=int, default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-bw', '--beam_width', type=int, default=5)
    parser.add_argument('-p', '--penalty', type=float, default=0.6)
    parser.add_argument('-ld', '--lamda', type=float, default=5)
    parser.add_argument('-m', '--load_model', type=str, default='VCOP/save/model.pt')
    parser.add_argument('-d', '--decode_set', type=str, default='test')
    parser.add_argument('-ml', '--max_len', type=int, default=100)
    parser.add_argument('-s', '--suffix', type=str, default=None)
    args = parser.parse_args()
    # print(args)
    predict_text_from_audio(args, wav_file_path)
