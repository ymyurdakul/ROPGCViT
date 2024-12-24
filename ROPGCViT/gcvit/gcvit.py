from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    MultiHeadRelativePositionalEmbedding,
    activation_by_name,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    mhsa_with_multi_head_relative_position_embedding,
    mlp_block,
    output_block,
    se_module,
    window_attention,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "gcvit_base": {"imagenet": {224: "cad46f57087f9e0a5c3848b3af3fe144"}},
    "gcvit_large": {
        "imagenet": {224: "3a556727053be1173227b85bef8049ff"},
        "imagenet21k-ft1k": {224: "1be7d0e856d0c59b83602155e73eab16", 384: "a9efb78e8783a23c61273d518000d508", 512: "79e7d5ff6e6d6183c1e5f7682fdce619"},
    },
    "gcvit_small2": {"imagenet": {224: "64c38e556b7d72a121c0fe03a0eb5255"}},
    "gcvit_small": {"imagenet": {224: "a6d57fba8d630bd3fa6de001a06f28b6"}},
    "gcvit_tiny2": {"imagenet": {224: "4465446ef57314d8e122d88ffe64152f"}},
    "gcvit_tiny": {"imagenet": {224: "9db89519e361f096307387ac20b2fad3"}},
    "gcvit_x_tiny": {"imagenet": {224: "453477bb07c69d3f9e06eb69fc383ec3"}},
    "gcvit_xx_tiny": {"imagenet": {224: "dec2edbe2542db2ef3cfc502a018e641"}},
}

def res_mlp_block(inputs, hidden_dim, output_channel=-1, drop_rate=0, use_conv=False, use_bias=True, activation="gelu", name=None):
    channel_axis = -1 if image_data_format() == "channels_last" else (1 if use_conv else -1)
    output_channel = output_channel if output_channel > 0 else inputs.shape[channel_axis]

    # First layer (Conv2D or Dense)
    if use_conv:
        nn = layers.Conv2D(hidden_dim, kernel_size=1, use_bias=use_bias, name=name and name + "Conv_0")(inputs)
    else:
        nn = layers.Dense(hidden_dim, use_bias=use_bias, name=name and name + "Dense_0")(inputs)
    nn = activation_by_name(nn, activation, name=name)
    nn = layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn

    # Second layer (Conv2D or Dense)
    if use_conv:
        nn = layers.Conv2D(output_channel, kernel_size=1, use_bias=use_bias, name=name and name + "Conv_1")(nn)
    else:
        nn = layers.Dense(output_channel, use_bias=use_bias, name=name and name + "Dense_1")(nn)
    nn = layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn

    # Residual connection
    if use_conv:
        # If Conv2D is used, check channel compatibility
        if inputs.shape[channel_axis] != output_channel:
            shortcut = layers.Conv2D(output_channel, kernel_size=1, use_bias=use_bias, name=name and name + "Shortcut")(inputs)
        else:
            shortcut = inputs
    else:
        # If Dense is used, ensure input-output compatibility (assumes matching dimensions for Dense layers)
        shortcut = inputs
    
    # Add residual connection
    nn = layers.Add(name=name and name + "Add")([shortcut, nn])
    
    return nn


def gcvit_block(inputs, window_size, num_heads=4, global_query=None, mlp_ratio=4, layer_scale=0, drop_rate=0, activation="gelu", name=""):
    # print(global_query)
    input_channel = inputs.shape[-1]  # Channels_last only
    attn = layer_norm(inputs, axis=-1, name=name + "attn_")
    attention_block = lambda inputs, num_heads, name: mhsa_with_multi_head_relative_position_embedding(
        inputs, num_heads=num_heads, global_query=global_query, qkv_bias=True, out_bias=True, data_format="channels_last", name=name
    )
    attn = window_attention(attn, window_size=window_size, num_heads=num_heads, attention_block=attention_block, name=name + "window_mhsa_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=-1, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = layer_norm(attn_out, axis=-1, name=name + "mlp_")
    mlp = res_mlp_block(mlp, int(input_channel * mlp_ratio), use_conv=False, activation=activation, name=name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=-1, name=name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return layers.Add(name=name + "output")([attn_out, mlp])


def to_global_query(inputs, window_ratio, num_heads=4, activation="gelu", name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    query = inputs
    num_window = 1
    if window_ratio == 1:
        query = extract_feature(query, strides=1, activation=activation, name=name + "down1_")
    else:
        while num_window < window_ratio:
            num_window *= 2
            query = extract_feature(query, strides=2, activation=activation, name=name + "down{}_".format(num_window))

    # print(f"{inputs.shape = }, {query.shape = }, {num_window = }, {window_ratio = }")
    if image_data_format() == "channels_last":
        query = functional.reshape(query, [-1, query.shape[1] * query.shape[2], num_heads, input_channel // num_heads])
        query = functional.transpose(query, [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dims]
    else:
        query = functional.reshape(query, [-1, num_heads, input_channel // num_heads, query.shape[2] * query.shape[3]])
        query = functional.transpose(query, [0, 1, 3, 2])  # also [batch, num_heads, hh * ww, key_dims]
    query = functional.repeat(query, num_window * num_window, axis=0)
    # print(f"{query.shape = }")
    return query


def down_sample(inputs, out_channels=-1, activation="gelu", name=""):
    out_channels = out_channels if out_channels > 0 else inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    nn = layer_norm(inputs, name=name + "down_1_")
    nn = extract_feature(nn, strides=1, activation=activation, name=name + "down_")
    nn = conv2d_no_bias(nn, out_channels, kernel_size=3, strides=2, padding="same", name=name + "down_")
    nn = layer_norm(nn, name=name + "down_2_")
    return nn


def extract_feature(inputs, strides=2, activation="gelu", name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    nn = depthwise_conv2d_no_bias(inputs, kernel_size=3, padding="same", name=name + "extract_")
    nn = activation_by_name(nn, activation=activation, name=name + "extract_")
    nn = se_module(nn, divisor=1, use_bias=False, activation=activation, use_conv=False, name=name + "extract_se_")
    nn = conv2d_no_bias(nn, input_channel, kernel_size=1, name=name + "extract_")
    nn = inputs + nn
    return layers.MaxPool2D(pool_size=3, strides=strides, padding="same", name=name + "extract_maxpool")(nn) if strides > 1 else nn
    # if strides > 1:
    #     nn = tf.pad(nn, [[0, 0], [1, 1], [1, 1], [0, 0]])
    #     nn = layers.MaxPool2D(pool_size=3, strides=strides, padding="valid", name=name + "extract_maxpool")(nn)
    # return nn


def GCViT(
    num_blocks=[2, 2, 6, 2],
    num_heads=[2, 4, 8, 16],
    # window_size=[7, 7, 14, 7],
    window_ratios=[8, 4, 1, 1],
    embed_dim=64,
    mlp_ratio=3,
    layer_scale=-1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="gcvit",
    kwargs=None,
):
    """Patch stem"""
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    nn = conv2d_no_bias(inputs, embed_dim, kernel_size=3, strides=2, use_bias=True, padding="same", name="stem_conv")
    se_module(inputs=nn, name="se")
    nn = down_sample(nn, name="stem_")
    height_axis, width_axis = (1, 2) if image_data_format() == "channels_last" else (2, 3)

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    num_stacks = len(num_blocks)
    for stack_id, (num_block, num_head, window_ratio) in enumerate(zip(num_blocks, num_heads, window_ratios)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = down_sample(nn, out_channels=nn.shape[-1 if image_data_format() == "channels_last" else 1] * 2, name=stack_name)

        window_size = (nn.shape[height_axis] // window_ratio, nn.shape[width_axis] // window_ratio)
        # window_size = (int(tf.math.ceil(nn.shape[height_axis] / window_ratio), int(tf.math.ceil(nn.shape[width_axis] / window_ratio))
        global_query = to_global_query(nn, window_ratio, num_head, activation=activation, name=stack_name + "q_global_")

        nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            cur_global_query = None if block_id % 2 == 0 else global_query
            nn = gcvit_block(nn, window_size, num_head, cur_global_query, mlp_ratio, layer_scale, block_drop_rate, activation=activation, name=block_name)
            global_block_id += 1
        nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last -> channels_first
    nn = layer_norm(nn, name="pre_output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "gcvit", pretrained, MultiHeadRelativePositionalEmbedding)
    return model


@register_model
def GCViT_XXTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GCViT(**locals(), model_name="gcvit_xx_tiny", **kwargs)


@register_model
def GCViT_XTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 5]
    return GCViT(**locals(), model_name="gcvit_x_tiny", **kwargs)


@register_model
def GCViT_Tiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 19, 5]
    return GCViT(**locals(), model_name="gcvit_tiny", **kwargs)


@register_model
def GCViT_Tiny2(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 29, 5]
    return GCViT(**locals(), model_name="gcvit_tiny2", **kwargs)


@register_model
def GCViT_Small(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 19, 5]
    num_heads = [3, 6, 12, 24]
    embed_dim = 96
    mlp_ratio = 2
    layer_scale = 1e-5
    return GCViT(**locals(), model_name="gcvit_small", **kwargs)


@register_model
def GCViT_Small2(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 23, 5]
    num_heads = [3, 6, 12, 24]
    embed_dim = 96
    mlp_ratio = 3
    layer_scale = 1e-5
    return GCViT(**locals(), model_name="gcvit_small2", **kwargs)


@register_model
def GCViT_Base1(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 19, 5]
    num_heads = [4, 8, 16, 32]
    embed_dim = 128
    mlp_ratio = 2
    layer_scale = 1e-5
    return GCViT(**locals(), model_name="gcvit_base", **kwargs)


@register_model
def GCViT_Base(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 4, 8, 4]
    num_heads = [4, 8, 16, 32]
    embed_dim = 128
    mlp_ratio = 2
    layer_scale = 1e-5
    return GCViT(**locals(), model_name="gcvit_base", **kwargs)


@register_model
def ROPGCViT(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 19, 5]
    num_heads = [4, 8, 16, 32]
    embed_dim = 128
    mlp_ratio = 2
    layer_scale = 1e-5
    return GCViT(**locals(), model_name="gcvit_base", **kwargs)


@register_model
def GCViT_Large(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 19, 5]
    num_heads = [6, 12, 24, 48]
    embed_dim = 192
    mlp_ratio = 2
    layer_scale = 1e-5
    return GCViT(**locals(), model_name="gcvit_large", **kwargs)