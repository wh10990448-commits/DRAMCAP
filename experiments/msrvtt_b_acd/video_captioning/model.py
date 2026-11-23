"""Frozen Vicuna captioner with hierarchical scene/entity video tokens."""
import torch
from torch import nn
from transformers import AutoTokenizer

from .hierarchical import HierarchicalSpatialTemporalAdapter
from .original import image_state


class VideoCaptioner(nn.Module):
    def __init__(self, model_dir, image_checkpoint, train_projector=True):
        super().__init__()
        from caption_model.decoder import MistralDecoder
        self.decoder = MistralDecoder(llm_path=model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=False, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.decoder.set_tokenizer(self.tokenizer)
        self.decoder._INSTRUCT_TEMPLATES = {
            key: '\nDescribe the video in one sentence.'
            for key in self.decoder._INSTRUCT_TEMPLATES}
        state = {
            key[len('decoder.'):]: value
            for key, value in image_state(image_checkpoint).items()
            if key.startswith('decoder.') and not key.startswith('decoder.llm.')}
        missing, unexpected = self.decoder.load_state_dict(state, strict=False)
        missing = [key for key in missing if not key.startswith('llm.')]
        if missing or unexpected:
            raise RuntimeError(
                f'Decoder projector checkpoint mismatch: {missing[:5]}, {unexpected[:5]}')
        self.decoder.requires_grad_(False)
        self.decoder.vis_proj_A.requires_grad_(train_projector)
        self.adapter = HierarchicalSpatialTemporalAdapter()

    def enable_gradient_checkpointing(self):
        self.decoder.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        self.decoder.llm.config.use_cache = False

    def initialize_from_baseline(self, state):
        """Transfer the validated temporal path and language projector."""
        self.adapter.initialize_scene_from_baseline(state['adapter'])
        self.decoder.vis_proj_A.load_state_dict(
            state['visual_projector'], strict=True)

    def tokens(self, batch):
        keys = [
            'visual', 'names', 'attributes', 'name_mask', 'attribute_mask',
            'timestamps', 'frame_mask', 'object_ids', 'object_geometry',
            'object_relations', 'direction_ids', 'object_mask']
        return self.adapter(**{key: batch[key] for key in keys})

    def _prefix(self, batch):
        visual, names, attributes, name_mask, attribute_mask = self.tokens(batch)
        project = self.decoder.vis_proj_A
        return self.decoder._build_prompt_embeds(
            visual.shape[0], visual.device,
            project(visual).to(torch.bfloat16),
            project(names).to(torch.bfloat16),
            project(attributes).to(torch.bfloat16),
            name_mask, attribute_mask)

    def _caption_tensors(self, captions, device, max_tokens=64):
        texts = [
            self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens - 1]
            + [self.tokenizer.eos_token_id] for text in captions]
        width = max(map(len, texts))
        ids = torch.full(
            (len(texts), width), self.tokenizer.pad_token_id,
            device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, text in enumerate(texts):
            ids[index, :len(text)] = torch.tensor(text, device=device)
            mask[index, :len(text)] = 1
        return ids, mask

    def caption_logprob(self, batch, captions):
        """Differentiable mean log-probability for SCST sampled captions."""
        prefix, prefix_mask, length = self._prefix(batch)
        ids, mask = self._caption_tensors(captions, prefix.device)
        attention = torch.cat([prefix_mask, mask], dim=1)
        positions = attention.long().cumsum(-1) - 1
        positions.masked_fill_(attention == 0, 1)
        outputs = self.decoder.llm(
            inputs_embeds=torch.cat(
                [prefix, self.decoder._get_embed_fn()(ids)], dim=1),
            attention_mask=attention, position_ids=positions,
            use_cache=False, return_dict=True)
        logits = outputs.logits[:, length - 1:-1].float()
        token_logprob = logits.log_softmax(-1).gather(
            -1, ids[:, :, None]).squeeze(-1)
        return (token_logprob * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    def forward(self, batch, captions=None, return_logprob=False, with_ce=False):
        logprob = None
        if return_logprob:
            if captions is None:
                raise ValueError('captions are required for log-probability mode')
            logprob = self.caption_logprob(batch, captions)
            if not with_ce:
                return logprob
        prefix, prefix_mask, length = self._prefix(batch)
        ids, mask = self._caption_tensors(batch['captions'], prefix.device)
        labels = ids.masked_fill(mask == 0, -100)
        labels = torch.cat([
            labels.new_full((ids.shape[0], length), -100), labels], dim=1)
        attention = torch.cat([prefix_mask, mask], dim=1)
        positions = attention.long().cumsum(-1) - 1
        positions.masked_fill_(attention == 0, 1)
        result = self.decoder.llm(
            inputs_embeds=torch.cat(
                [prefix, self.decoder._get_embed_fn()(ids)], dim=1),
            attention_mask=attention, position_ids=positions, labels=labels,
            use_cache=False, return_dict=True)
        if return_logprob:
            return logprob, result.loss
        return result.loss, mask.sum().detach()

    @torch.no_grad()
    def generate(self, batch, **kwargs):
        visual, names, attributes, name_mask, attribute_mask = self.tokens(batch)
        ids = self.decoder.generate(
            visual, name_tokens_768=names, attr_tokens_768=attributes,
            name_mask=name_mask, attr_mask=attribute_mask, **kwargs)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def adapted_state(self):
        return {
            'adapter': self.adapter.state_dict(),
            'visual_projector': self.decoder.vis_proj_A.state_dict()}

    def load_adapted_state(self, state):
        self.adapter.load_state_dict(state['adapter'], strict=True)
        self.decoder.vis_proj_A.load_state_dict(
            state['visual_projector'], strict=True)
