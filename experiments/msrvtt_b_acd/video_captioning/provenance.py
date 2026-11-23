"""Identity of the frozen frame encoder and its preprocessing configuration."""
import hashlib
import json
from pathlib import Path


def feature_provenance(root, config, frames, detector):
    root=Path(root);config=Path(config)
    verification=json.loads((root/'original_checkpoint_verified.json').read_text())
    backbones=json.loads((root/'backbone_manifest.json').read_text())
    if not verification['verified'] or not backbones['all_verified']:
        raise RuntimeError('Original checkpoint/backbone integrity gate failed')
    if (root/'stage2_final.pt').stat().st_size!=verification['bytes']:
        raise RuntimeError('Verified checkpoint size changed')
    result={'image_checkpoint_sha256':verification['sha256'],'frames':frames,
            'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'backbones':backbones,
            'roi_source':detector,'sampling':'nearest PTS to uniform time-bin centers'}
    result['eva_restore_policy']='Restore embedded image-checkpoint EVA when present; verified official EVA otherwise'
    if detector=='fasterrcnn-coco':
        result['detector']=json.loads((root/'detector_manifest.json').read_text())
        if not result['detector']['published_hash_prefix_verified']:
            raise RuntimeError('Detector weights are unverified')
    for name in ['ext_memory_lvis.pkl','attr_memory/attr_memory.pkl','attr_memory/name2attr_topk.json']:
        result[name]=hashlib.sha256((config.parent/name).read_bytes()).hexdigest()
    for name in ['caption_model/model.py','caption_model/vision_frontend.py','caption_model/qformer.py',
                 'caption_model/retriever.py','video_captioning/original.py','video_captioning/data.py',
                 'video_captioning/detector.py','video_captioning/provenance.py']:
        result['code:'+name]=hashlib.sha256((config.parent/name).read_bytes()).hexdigest()
    result['fingerprint']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result
