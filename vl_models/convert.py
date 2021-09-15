# coding: utf-8
# Convert weights to models for transformers_volta


import os
import json
import torch
from eval_vl_glue import transformers_volta, VoltaImageFeature, load_tsv


def make_model(config_path, weight_path, output_path, default_image_feature=None):
    """
    Make a volta model from a config and weight.
    Arguments:
        config_path: path to config.json for a volta model
        weight_path: path to a weight file
        output_path: path to save new model (model directory)
        default_image_feature: default image feature
    Returns:
        None
    """
    
    if os.path.exists(output_path):
        raise RuntimeError(f'output path already exists: {output_path}')
    
    # Config
    config = None
    with open(config_path, 'r') as f:
        config = json.load(f)
        del config['clf_hidden_size']
        config = transformers_volta.models.volta.VoltaConfig.from_dict(config)
    
    # Tokenizer
    base_tokenizer_name = 'bert-base-uncased'
    tokenizer = transformers_volta.models.volta.VoltaTokenizer.from_pretrained(
        base_tokenizer_name,
        model_max_length=config.max_position_embeddings,
    )
    
    # Model
    model = transformers_volta.models.volta.VoltaModel(config)
    model = transfer_from(torch.load(weight_path), model)
    if default_image_feature:
        model.set_default_image_feature(default_image_feature)
        print('default_image_feature was set.')
    
    # Save them in a model directory
    os.makedirs(output_path)
    config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    

def make_reinit_model(base_model_path, output_path, default_image_feature=None):
    """
    Make a volta model whose weight is reinitialized based on 
        a model original before the V&L pretraining (bert-base-uncased).
    Arguments:
        base_model_path: path to config.json for a volta model
        output_path: path to save new model (model directory)
        default_image_feature: default image feature
    Returns:
        None
    """    
    if os.path.exists(output_path):
        raise RuntimeError(f'output path already exists: {output_path}')
    
    # Grand means the base model of the base_model.
    grand_model = 'bert-base-uncased'
    
    # Config: we retain the base model's one
    config = transformers_volta.models.volta.VoltaConfig.from_pretrained(base_model_path)
    
    # Tokenizer: we retain the base model's one
    tokenizer = transformers_volta.models.volta.VoltaTokenizer.from_pretrained(base_model_path)
    
    # Model
    model = transformers_volta.models.volta.VoltaModel(config)
    model = transfer_from(grand_model, model)
    if default_image_feature:
        model.set_default_image_feature(default_image_feature)
        print('default_image_feature was set.')
    
    os.makedirs(output_path)
    config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)


def transfer_from(src_state_dict_or_bert_name, target_model, output_loading_info=False):
    """
    Transfer the weight specified by src_state_dict_or_bert_name into the target_model.
    Arguments:
        src_state_dict_or_bert_name: source weight; str or dict
        target_model: a torch model
        output_loading_info: whether to return additional information or not
    Returns:
        the given target model (and loading_info)
    """
    
    if isinstance(src_state_dict_or_bert_name, str):
        src_model = transformers_volta.AutoModel.from_pretrained(src_state_dict_or_bert_name)
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


def main(args):
    
    print(f'called with args: {args}')
    
    image_feature = VoltaImageFeature.from_dict(load_tsv(args.tsv)[args.image_key])
    
    make_model(
        config_path=args.config,
        weight_path=args.weight,
        output_path=args.output,
        default_image_feature=image_feature,
    )
    
    if args.reinit == 1:
        make_reinit_model(
            base_model_path=args.output,
            output_path=args.output + '_reinit',
            default_image_feature=image_feature,
        )


if __name__ == '__main__':
    
    import argparse
    
    dir_name  = os.path.dirname(__file__)
    
    parser = argparse.ArgumentParser(description='Convert a volta weight to a model.')
    parser.add_argument('--config', '-c', type=str, required=True,
        help='path to a config.json file')
    parser.add_argument('--weight', '-w', type=str, required=True,
        help='path to a pretrained weight')
    parser.add_argument('--output', '-o', type=str, default=None,
        help='path to a direcctory for the new model. if omitted a directory with the same name as weight will appear in the vl_models directory')
    parser.add_argument('--image_key', '-d', type=str, default='filled_with_0',
        help='image key in the tsv file for the default image')
    parser.add_argument('--tsv', '-t', type=str, default=os.path.join(dir_name, 'test_obj36.tsv'),
        help='path to a tsv file that contains image features. the default is test_obj36.tsv in vl_models directory')
    parser.add_argument('--reinit', '-r', type=int, default=1,
        help='if set 1, a reinit model will be made in addition to main model. set 0 when turning this option off.')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(dir_name, os.path.basename(args.weight))
    
    main(args)
