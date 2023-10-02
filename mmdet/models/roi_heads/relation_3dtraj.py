import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import numpy as np

def stn(x, theta, padding_mode='zeros'):
    grid = F.affine_grid(theta, x.size())
    img = F.grid_sample(x, grid, padding_mode=padding_mode)
    return img

class Encoder3D(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Encoder3D, self).__init__()
        self.in_channel = in_channel
        self.conv3d_1 = nn.ConvTranspose3d(in_channel, hidden_channel, 4, stride=2, padding=1)
        # self.conv3d_2 = nn.ConvTranspose3d(hidden_channel, hidden_channel, 4, stride=2, padding=1)

    def forward(self, feat):
        B,C,H,W = feat.shape
        z_3d = feat.reshape([B, self.in_channel, -1, H, W])
        z_3d = F.leaky_relu(self.conv3d_1(z_3d))
        # z_3d = F.leaky_relu(self.conv3d_2(z_3d))
        return z_3d

class Rotate(nn.Module):
    def __init__(self, learn, in_channel):
        super(Rotate, self).__init__()
        self.padding_mode = 'zeros'
        self.learn = learn
        if self.learn:
            self.conv3d_1 = nn.Conv3d(in_channel,in_channel,3,padding=1)
            # self.conv3d_2 = nn.Conv3d(in_channel,in_channel,3,padding=1)

    def forward(self, code, theta):
        rot_code = stn(code, theta, self.padding_mode)
        if self.learn:
            rot_code = F.leaky_relu(self.conv3d_1(rot_code))
            # rot_code = F.leaky_relu(self.conv3d_2(rot_code))
        return rot_code

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.feat_remap = nn.Conv3d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 2048, 1)
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.upconv_final = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, code):
        # recon feat
        feat_3d = self.feat_remap(code)
        recon_feat_2d = feat_3d.view(-1, feat_3d.size(1) * feat_3d.size(2), feat_3d.size(3), feat_3d.size(4))
        # recn img
        feat_2d = F.leaky_relu(self.conv3(recon_feat_2d))
        feat_2d = F.leaky_relu(self.upconv1(feat_2d))
        feat_2d = F.leaky_relu(self.upconv2(feat_2d))
        feat_2d = F.leaky_relu(self.upconv3(feat_2d))
        feat_2d = F.leaky_relu(self.upconv4(feat_2d))
        img_2d = self.upconv_final(feat_2d)
        return img_2d

class Relate3DMix(nn.Module):
    def __init__(self, support_param, mode):
        super(Relate3DMix, self).__init__()
        # 3d mapping
        self.in_channel_vox = support_param['in_channel3d']
        self.hidden_vox = support_param['hidden_3d']
        self.encode3d = Encoder3D(self.in_channel_vox, self.hidden_vox)
        # rotate
        self.rot_learn = support_param['learn_rot']
        self.rotate = Rotate(support_param['learn_rot'], self.hidden_vox)
        # recon decoder, for recon mode only
        if mode == 'recon':
            self.decoder = Decoder()
            self.rotate_inv = Rotate(support_param['learn_rot'], self.hidden_vox)
        # vox relation
        self.vox_relate = nn.Sequential(
                            nn.Conv3d(self.hidden_vox*2, self.hidden_vox, kernel_size=3, groups=self.hidden_vox, padding=1),
                            nn.BatchNorm3d(self.hidden_vox),
                            nn.ReLU(),
                            nn.AvgPool3d(2, stride=2),
                            nn.Conv3d(self.hidden_vox, self.in_channel_vox, kernel_size=1),
                            nn.BatchNorm3d(self.in_channel_vox),
                            nn.ReLU(),
                            )
        # global branch
        self.in_channel = support_param['in_channel2d']
        self.hidden = support_param['hidden_2d']
        self.globalr = nn.Sequential(
                            nn.Conv2d(self.in_channel, self.hidden, kernel_size=1, padding=0, bias=False),
                            nn.BatchNorm2d(self.hidden),
                            nn.ReLU(),
                            nn.Conv2d(self.hidden, self.in_channel, kernel_size=1, bias=False),
                            nn.BatchNorm2d(self.in_channel),
                            nn.ReLU(),
                            )
        self.init_support()
        # traj estimate
        self.num_r = support_param['num_r']
        self.traj = nn.Sequential(
                            nn.Conv3d(self.hidden_vox*2, self.hidden_vox, kernel_size=3, groups=self.hidden_vox),
                            nn.BatchNorm3d(self.hidden_vox),
                            nn.ReLU(),
                            nn.Conv3d(self.hidden_vox, self.in_channel_vox, kernel_size=3),
                            nn.BatchNorm3d(self.in_channel_vox),
                            nn.ReLU()
                            )
        self.traj_fc = nn.Linear(self.in_channel_vox, 6*self.num_r)


    def init_support(self):
        for m in self.globalr.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        for m in self.vox_relate.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

    def forward(self, query, support, support_traj=None, return_support=False, return_rotvec=False):
        if support_traj!=None:
            B, N, C, w, h = support.shape
            query = query.reshape(B, -1, C, w, h)
            P = query.shape[1]
            voxel_s = self.encode3d(support.flatten(0,1))
            D, H, W = voxel_s.shape[2], voxel_s.shape[3], voxel_s.shape[4]
            theta_s = support_traj[:, :N].reshape(B*N, 3, 4)
            rot_voxels_support = self.rotate(voxel_s, theta_s).view(B, N, -1, D, H, W)
            voxels_support = rot_voxels_support.mean(dim=1, keepdim=True).repeat(1, P, 1, 1, 1, 1).flatten(0,1) # B, C, D, H, W
            support_g = support.mean([1, 3, 4], keepdim=True)
        else:
            support_vox = support['3d_support']
            support_g = support['2d_support']
            B, _, _, D, H, W = support_vox.shape
            C, w, h = query.shape[1], query.shape[2], query.shape[3]
            query = query.reshape(B, -1, C, w, h)
            P = query.shape[1]
            voxels_support = support_vox.repeat(1, P, 1, 1, 1, 1).flatten(0,1) # B, C, D, H, W
        # voxel query
        voxel_q = self.encode3d(query.flatten(0,1))
        if self.rot_learn:
            # measure r of query
            theta_q = torch.cat([torch.eye(3).unsqueeze(0).repeat(P*B, 1, 1), torch.zeros(3).unsqueeze(0).repeat(P*B, 1).unsqueeze(-1)], dim=-1).to(voxel_q.device)
            voxels_query = self.rotate(voxel_q, theta_q).view(B*P, -1, D, H, W)
            rot_mat = self.traj_estimate(voxels_support, voxels_query)
            rot_mat = rot_mat.flatten(0,1)
            voxel_q = voxel_q.unsqueeze(1).repeat(1, self.num_r, 1, 1, 1, 1).flatten(0,1)
            voxels_query = self.rotate(voxel_q, rot_mat).view(B*P, self.num_r, -1, D, H, W)
            voxels_query = voxels_query.mean(dim=1)
        else:
            voxels_query = voxel_q
        # 3d relation
        s = voxels_support
        q = voxels_query
        sta_feat = torch.stack([q, s], dim=2).flatten(1, 2)
        rela_feat3d = self.vox_relate(sta_feat)
        rela_feat3d = rela_feat3d.reshape(B*P, C, w, h)
        # global branch
        s = support_g
        q = query
        rela_g = torch.zeros_like(q)
        for b in range(B):
            s_b = s[b]
            q_b = q[b]
            rela_g[b] = F.conv2d(q_b, s_b.permute(1,0,2,3), groups = C)
        rela_g = rela_g.flatten(0,1)
        rela_g = self.globalr(rela_g)
        # fuse
        rela_feat = rela_g + rela_feat3d
        # return
        if return_support:
            return rela_feat, {
                "3d_support": rot_voxels_support.mean(dim=1, keepdim=True),
                "2d_support": support.mean([1, 3, 4], keepdim=True),
            }, rot_mat
        else:
            return rela_feat, rot_mat

    def rotation_6d_to_matrix(self, d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def traj_estimate(self, voxels_support, voxels_query):
        sta_feat = torch.stack([voxels_support, voxels_query], dim=2).flatten(1, 2)
        traj = self.traj(sta_feat)
        traj = traj.mean(dim=[2,3,4])
        traj = traj.flatten(1)
        traj_rot_6d = self.traj_fc(traj).reshape(-1, self.num_r, 6).flatten(0,1)
        traj_rot_mat = self.rotation_6d_to_matrix(traj_rot_6d).reshape(-1, self.num_r, 3, 3)
        traj_trans_vec = torch.zeros((1, 1, 3, 1), device=traj_rot_mat.device).repeat(traj_rot_mat.shape[0], self.num_r, 1, 1)
        traj = torch.cat([traj_rot_mat, traj_trans_vec], dim=-1)
        return traj

    def recon(self, support, support_traj, tar_pose):
        B, N, C, w, h = support.shape
        P = tar_pose.shape[1]
        # 3d mapping
        voxel_s = self.encode3d(support.flatten(0,1))
        D, H, W = voxel_s.shape[2], voxel_s.shape[3], voxel_s.shape[4]
        # 3d transform
        theta_s = support_traj[:, :N].reshape(B*N, 3, 4)
        rot_voxels_support = self.rotate(voxel_s, theta_s).view(B, N, -1, D, H, W)
        voxels_support = rot_voxels_support.mean(dim=1, keepdim=True).repeat(1, P, 1, 1, 1, 1).flatten(0,1) # B, C, D, H, W
        # decode
        theta_q = tar_pose.reshape(B*P, 3, 4)
        tar_voxel = self.rotate_inv(voxels_support, theta_q).view(B, P, -1, D, H, W)
        img_2d = self.decoder(tar_voxel.flatten(0,1))
        return img_2d
