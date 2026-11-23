"""Load preserved image weights and reuse their exact frame-token pathway."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def image_state(checkpoint):
    obj = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    for key in ['model', 'model_state_dict', 'state_dict']:
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not obj or not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise ValueError('Checkpoint does not contain a tensor model state')
    return {k.removeprefix('module.'): v for k, v in obj.items()}


class EmbeddingBackbone(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        p = Path(model_dir)
        self.config = SimpleNamespace(**json.loads((p/'config.json').read_text()))
        index = json.loads((p/'pytorch_model.bin.index.json').read_text())
        name = index['weight_map']['model.embed_tokens.weight']
        weights = torch.load(p/name, map_location='cpu', weights_only=True, mmap=True)
        self.embedding = nn.Embedding.from_pretrained(weights['model.embed_tokens.weight'].to(torch.bfloat16), freeze=True)

    def get_input_embeddings(self):
        return self.embedding


class EmbeddingOnlyDecoder(nn.Module):
    """Frame retrieval uses token embeddings, so it need not load all LM layers."""
    def __init__(self, model_dir):
        super().__init__()
        self.llm = EmbeddingBackbone(model_dir)

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def _get_embed_fn(self):
        return self.llm.get_input_embeddings()


def load_frame_model(config_path, checkpoint, device='cuda:0'):
    from config import get_config
    from caption_model import CaptionModel
    from transformers import AutoTokenizer
    cfg = get_config(config_path)
    cfg.training.mode = 'online'
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.llm_path, local_files_only=True, use_fast=False)
    model = CaptionModel(cfg, decoder=EmbeddingOnlyDecoder(cfg.paths.llm_path))
    model.set_tokenizer(tokenizer)
    state = {k: v for k, v in image_state(checkpoint).items() if not k.startswith('decoder.')}
    saved_retrieval = any(k.startswith('retrieval_qformer.lavis_qformer.') for k in state)
    if saved_retrieval:
        from caption_model.retrieval_qformer import LAVISQFormerForRetrieval
        model.retrieval_qformer.lavis_qformer = LAVISQFormerForRetrieval()
    saved_eva = any(k.startswith('vision_frontend._eva_model.') for k in state)
    # Register the EVA module before restoring checkpoints that contain its state.
    model.to(device).eval()
    eva_state = {k[len('vision_frontend._eva_model.'):]: v for k,v in state.items() if k.startswith('vision_frontend._eva_model.')}
    model.vision_frontend.load_encoders(device, checkpoint_state=eva_state if saved_eva else None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith('decoder.')]
    if not saved_eva:
        missing = [k for k in missing if not k.startswith('vision_frontend._eva_model.')]
    if missing or unexpected:
        raise RuntimeError(f'Image checkpoint mismatch: missing={missing[:15]}, unexpected={unexpected[:15]}')
    if saved_retrieval:
        model.retrieval_qformer.lavis_qformer._loaded = True
        model.retrieval_qformer._loaded = True
    if model._pkl_retriever is None or not model._pkl_retriever._loaded:
        raise RuntimeError('Original name/attribute memories are required')
    if model._pkl_retriever.attr_keys is None:
        raise RuntimeError('Attribute memory is missing')
    model.requires_grad_(False)
    model.eval()
    print('IMAGE_CHECKPOINT_RESTORED', json.dumps({'saved_retrieval_branch': saved_retrieval,
          'saved_eva_branch': saved_eva, 'missing_active_keys': missing, 'unexpected_keys': unexpected}), flush=True)
    return model


@torch.no_grad()
def frame_tokens(model, images, boxes=None):
    feat = model.extract_online(images, lvis_boxes=boxes)
    patches = feat['eva_patches']
    compressed, retrieval_vectors = model.retrieval_qformer(patches)
    if not torch.isfinite(compressed).all() or float(compressed.float().abs().sum()) == 0:
        raise RuntimeError('Retrieval Q-Former must produce real finite features')
    fused = model._fuse_ms_patch(patches, feat['eva_mid_patches'])
    attrs = model._build_attr_feats_simple(feat['eva_rois'], feat['category_ids'],
                                          feat['tag_sequences'], retrieval_vecs=retrieval_vectors)
    return model.build_visual_tokens(
        feat['eva_global'], feat['eva_global_768'], feat['eva_rois'],
        abs_pos=feat['absolute_pos'], rel_pos=feat['relative_pos'],
        attr_feats=attrs, tag_sequences=feat['tag_sequences'], is_core=feat['is_core'],
        importance_scores=feat['importance_scores'], roi_mask=feat['roi_mask'],
        category_ids=feat['category_ids'], eva_patches_fused=fused,
        eva_mid_patches=feat['eva_mid_patches'], eva_patches_raw=patches,
        boxes_batch=feat['boxes_batch'], pre_global_32q=compressed,
        retrieval_vecs=retrieval_vectors)


@torch.no_grad()
def encode_video(model, path, frames=8, frame_batch=1, detector=None):
    from .data import sample_video
    images,times,valid=sample_video(path,frames)
    streams=[[] for _ in range(5)]
    for offset in range(0,len(images),frame_batch):
        batch=images[offset:offset+frame_batch]
        boxes=detector(batch) if detector is not None else None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=frame_tokens(model,batch,boxes)
        for stream,tensor in zip(streams,output):
            stream.append(tensor.detach().cpu() if tensor is not None else None)
    tensors=[]
    for i,stream in enumerate(streams):
        if any(x is None for x in stream):
            if not all(x is None for x in stream):
                raise RuntimeError('Inconsistent retrieval stream availability')
            value=torch.zeros(frames,8,768) if i<3 else torch.zeros(frames,8,dtype=torch.bool)
        else:
            value=torch.cat(stream)
        if i<3:
            if not torch.isfinite(value).all():
                raise RuntimeError('Non-finite frame features')
            value=value.to(torch.bfloat16)
        tensors.append(value)
    if float(tensors[0].float().abs().sum())==0:
        raise RuntimeError('Zero visual features cannot be used')
    result=dict(zip(['visual','names','attributes','name_mask','attribute_mask'],tensors))
    result.update(timestamps=times,frame_mask=valid)
    return result
