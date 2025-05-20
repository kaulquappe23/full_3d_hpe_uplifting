# Models
## Input arguments for `UpliftUpsampleTransformer`

`in_feature_size` (`int`)
    Number of input channels, typically 2 for `(x, y)` coordinates of the 2D poses.

`full_output` (`bool`): If `True`, the full sequence is predicted before the strided transformer is applied, and a loss is also calculated for the full sequence.

`num_frames` (`int`): Number of frames in the input sequence. The sequence length of the input is `(num_frames - 1) * config.MASK_STRIDE + 1`.

`num_keypoints` (`int`): Number of keypoints for the pose.

`spatial_d_model` (`int`): Embedding dimension for the spatial transformer (within pose attention).

`temporal_d_model` (`int`): Embedding dimension for the temporal transformer (across pose attention).

`spatial_depth` (`int`): Number of transformer layers for the spatial transformer.

`temporal_depth` (`int`): Number of transformer layers for the temporal transformer.

`strides` (`list of int`): Stride values for the strided transformer. It is a list of stride values, one for each transformer layer.

`paddings` (`list of int`): Padding values for the strided transformer layers. The length of paddings must be equal to the length of strides.

`num_heads` (`int`): Number of heads for all transformer blocks.

`mlp_ratio` (`float`): The MLP in each transformer block after the attention consists of two layers. The hidden layer has `embed_dim * mlp_ratio` neurons.

`qkv_bias` (`bool`): Whether to use a bias for the qkv projection or not.

`attn_drop_rate` (`float`): Dropout rate used after the attention and after each fully connected layer in the MLP of the transformer blocks.

`drop_rate` (`float`): Dropout rate used after the qkv projection and after each fully connected layer in the MLP of the transformer blocks.

`drop_path_rate` (`list of float`): Path drop rate for the transformers. The first value is for the spatial transformer, the second for the temporal transformer, the third for the strided transformer. Drop path means that randomly, a block (path) of the network is disabled. The drop path rate is increased linearly from 0 to the given value for each transformer type.

`norm_layer` (`callable`): Normalization layer applied in transformer blocks.

`output_bn` (`bool`): Whether to use batch normalization before the final output head layers.

`has_strided_input` (`bool`): Indicates if the input is a sequence with a 2D pose for every frame or a strided sequence.

`first_strided_token_attention_layer` (`int`): First layer of the temporal transformer where attention is allowed from strided tokens to other tokens.

`token_mask_rate` (`float`): Additional random masking of tokens with the given rate before the temporal transformer begins.

`learnable_masked_token` (`bool`): In case of token masking, mask with a learnable layer if True, else with zero.

`return_attention` (`bool`): Whether to return the attention values additionally to the output.
