import os
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.sam import SamVisionModel, SamImageProcessor, SamVisionConfig


class CLIPSamVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = [os.path.join(vision_tower, 'clip'), os.path.join(vision_tower, 'sam')]
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SamVisionConfig.from_pretrained(self.vision_tower_name[1])
            # self.cfg_only.clip = CLIPVisionConfig.from_pretrained(self.vision_tower_name[0])
            # self.cfg_only.sam = SamVisionConfig.from_pretrained(self.vision_tower_name[1])

    def load_model(self):
        print('load two tower')
        self.image_processor = [CLIPImageProcessor.from_pretrained(self.vision_tower_name[0]), SamImageProcessor()]
        self.clip_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name[0])
        self.sam_tower = SamVisionModel.from_pretrained(self.vision_tower_name[1])
        self.vision_tower = [self.clip_tower, self.sam_tower]
        for tower in self.vision_tower:
            tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs, idx):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if idx == 0:
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    def image_batch(self, images, image_processor):
        images_feature = [image_processor(image, return_tensors='pt')['pixel_values'][0] for image in images]
        return torch.stack(images_feature)

    @torch.no_grad()
    def forward(self, images):
        image_features = []
        i = 0
        for tower in self.vision_tower:
            image_pre = self.image_batch(images, self.image_processor[i])
            # print(image_pre.size())
            image_forward_out = tower(image_pre.to(device=tower.device, dtype=tower.dtype), output_hidden_states=True)
            vit_feature = self.feature_select(image_forward_out, i).to(image_pre.dtype)
            image_features.append(vit_feature)
            i += 1
        # print(image_features[0].size(), image_features[1].size())

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower[1].config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
