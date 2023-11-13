from .clip_encoder import CLIPVisionTower
from .clip_ch_encoder import ChineseCLIPVisionTower
from .sam_encoder import SamVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
    if "clip-vit" in vision_tower.lower() and "chinese" in vision_tower.lower():
        print("use chinese clip to build vision tower")
        return ChineseCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "clip-vit" in vision_tower.lower():
        print("use english clip to build vision tower")
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "sam-vit" in vision_tower.lower():
        print("use sam vit to build vision tower")
        return SamVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
