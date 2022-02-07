import sys
import torch
from torch import nn
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from video_encoding.models.s3dg import S3D
from paths import PROJECT_PATH


weights_path = osp.join(PROJECT_PATH, 'video_encoding', 'model_weights')
device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = S3D(osp.join(weights_path, 's3d_dict.npy'))
        state_dict = torch.load(osp.join(weights_path, 's3d_howto100m.pth'))
        self.net.load_state_dict(state_dict)
        self.net.to(device)
        self.net.eval()
        self.num_frames = 32

    @torch.no_grad()
    def retrieve(self, texts, videos):
        # video frames have to be normalized in [0, 1]
        video_descriptors = torch.cat([self.net(v[None, ...].cuda())['video_embedding'] for v in videos], 0)
        text_descriptors = self.net.text_module(texts)['text_embedding']
        scores = text_descriptors @ video_descriptors.t()

        decr_sim_inds = torch.argsort(scores, descending=True, dim=1)
        outs = []
        for i in range(len(texts)):
            sorted_videos = [{'video_ind': j, 'score': scores[i, j]} for j in decr_sim_inds[i]]
            outs.append(sorted_videos)
        return outs

    @torch.no_grad()
    def embed_full_video(self, frames):
        # assuming the video is at 10fps and that we take 32 frames
        # frames is a tensor of size [T, W, H, 3]
        T, W, H, _ = frames.shape
        frames = frames.permute(3, 0, 1, 2)
        N_chunks = T // self.num_frames
        n_last_frames = T % self.num_frames
        if n_last_frames > 0:
            zeros = torch.zeros((3, self.num_frames - n_last_frames, W, H), dtype=torch.uint8).to(frames.device)
            frames = torch.cat((frames, zeros), axis=1)
            N_chunks += 1

        # extract features
        chunk_features = []
        for i in range(0, N_chunks):
            chunk_frames = frames[:, i * self.num_frames : (i + 1) * self.num_frames, ...][None, ...]
            chunk_feat = self.net(chunk_frames.cuda())['video_embedding']
            chunk_features.append(chunk_feat)

        chunk_features = torch.cat(chunk_features, 0)
        return chunk_features

    @torch.no_grad()
    def embed_full_subs(self, subs):
        clipped_subs = [' '.join(s.split(' ')[:30]) for s in subs]
        sub_features = self.net.text_module(clipped_subs)['text_embedding']
        return sub_features
