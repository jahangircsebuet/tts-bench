import torch
torch.backends.cudnn.enabled = False

# transformers 5.x rejects Ultravox's WhisperProcessor as `audio_processor`; skip that one check.
try:
    from transformers import processing_utils as _pu
    _orig = _pu.ProcessorMixin.check_argument_for_proper_class

    def _check(self, argument_name, argument):
        if argument_name == "audio_processor":
            return type(argument)
        return _orig(self, argument_name, argument)

    _pu.ProcessorMixin.check_argument_for_proper_class = _check
except Exception:
    pass
