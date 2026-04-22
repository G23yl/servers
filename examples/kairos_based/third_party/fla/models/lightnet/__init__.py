
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from examples.kairos_based.third_party.fla.models.lightnet.configuration_lightnet import LightNetConfig
from examples.kairos_based.third_party.fla.models.lightnet.modeling_lightnet import LightNetForCausalLM, LightNetModel

AutoConfig.register(LightNetConfig.model_type, LightNetConfig, exist_ok=True)
AutoModel.register(LightNetConfig, LightNetModel, exist_ok=True)
AutoModelForCausalLM.register(LightNetConfig, LightNetForCausalLM, exist_ok=True)


__all__ = ['LightNetConfig', 'LightNetForCausalLM', 'LightNetModel']
