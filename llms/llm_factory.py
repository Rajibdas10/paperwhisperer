def load_model(model_name ="minilm"):
    if model_name == "minilm":
        from .minilm_llm import MiniLMWrapper
        return MiniLMWrapper()
    
    elif model_name == "phi2":
        from .phi2_llm import Phi2Wrapper
        return Phi2Wrapper()
    
    elif model_name == "zephyr":
        from .zephyr_llm import ZephyrWrapper
        return ZephyrWrapper()
    else:
        raise ValueError ("Invalid model name")
    