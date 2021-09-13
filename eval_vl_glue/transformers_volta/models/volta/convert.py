# coding: utf-8


import transformers


def transfer_from(src_state_dict_or_bert_name, target_model, output_loading_info=False):
    
    if isinstance(src_state_dict_or_bert_name, str):
        src_model = transformers.AutoModel.from_pretrained(src_state_dict_or_bert_name)
        state_dict = src_model.state_dict()
        from_bert = True
    else:
        state_dict = src_state_dict_or_bert_name
        from_bert = False
        
    config = target_model.config
    cls = type(target_model)
    
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        print('weight mapping', old_key, '->', new_key)
        state_dict[new_key] = state_dict.pop(old_key)

    # Rename Bert parameters for our framework
    # NB: Assume 1 Bert layer is mapped to 1 layer only (cannot be used to init multiple layers)
    old_keys = []
    new_keys = []
    nums = []
    for key in state_dict.keys():
        new_key = None
        if ".layer." in key and from_bert:
            num = int(key.split(".layer.")[-1].split(".")[0])
            if ".attention." in key:
                new_key = key.replace(".layer.%d.attention." % num,
                                      ".layer.%d.attention_" % config.bert_layer2attn_sublayer.get(str(num), num))
            elif ".intermediate." in key:
                new_key = key.replace(".layer.%d.intermediate." % num,
                                      ".layer.%d.intermediate." % config.bert_layer2ff_sublayer.get(str(num), num))
            elif ".output." in key:
                new_key = key.replace(".layer.%d.output." % num,
                                      ".layer.%d.output." % config.bert_layer2ff_sublayer.get(str(num), num))
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
            nums.append(num)
    
    for old_key, new_key, _ in sorted(zip(old_keys, new_keys, nums), key=lambda x: x[2], reverse=True):
        print('weight mapping', old_key, '->', new_key)
        state_dict[new_key] = state_dict.pop(old_key)

    # Load from a PyTorch state_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
         )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ""
    model_to_load = target_model
    base_model_prefixes = [cls.base_model_prefix] if hasattr(target_model, cls.base_model_prefix) else []
    base_model_prefixes.append('bert')
    for base_model_prefix in base_model_prefixes:
        model_has_base_model = hasattr(target_model, base_model_prefix)
        base_model_in_state_dict = any(s.startswith(base_model_prefix) for s in state_dict.keys())
        if not model_has_base_model and base_model_in_state_dict:
            start_prefix = base_model_prefix + "."
            break
        if model_has_base_model and not base_model_in_state_dict:
            model_to_load = getattr(target_model, base_model_prefix)
            break
    print('start_prefix', start_prefix, 'model_to_load', model_to_load.__class__.__name__)
    
    original_type_vocab_size = model_to_load.config.type_vocab_size
    if original_type_vocab_size != 2 and from_bert:
        model_to_load.embeddings.token_type_embeddings = \
            model_to_load._get_resized_embeddings(model_to_load.embeddings.token_type_embeddings, 2)
    
    # load the weight to the model_to_load module
    load(model_to_load, prefix=start_prefix)
    
    if original_type_vocab_size != 2 and from_bert:
        model_to_load.embeddings.token_type_embeddings = \
            model_to_load._get_resized_embeddings(model_to_load.embeddings.token_type_embeddings, original_type_vocab_size)
    
    if len(missing_keys):
        print(
            "Weights of {} not initialized from pretrained model: {}".format(cls.__name__, missing_keys)
        )
    if len(unexpected_keys):
        print(
            "Weights from pretrained model not used in {}: {}".format(cls.__name__, unexpected_keys)
        )
    if len(error_msgs):
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(cls.__name__, "\n\t".join(error_msgs))
        )
    
    if hasattr(target_model, "tie_weights"):
        target_model.tie_weights()  # make sure word embedding weights are still tied

    # Set model in evaluation mode to desactivate DropOut modules by default
    target_model.eval()

    if output_loading_info:
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "error_msgs": error_msgs,
        }
        return target_model, loading_info
    return target_model