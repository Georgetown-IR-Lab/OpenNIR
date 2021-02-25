import sys
import copy
import torch
import math
from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForPreTraining, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertEmbeddings, BertLayerNorm, ACT2FN


last_attn_output = None


class Act(nn.Module):
    def __init__(self, fn, compress_fp16):
        super().__init__()
        self.fn = fn
        self.compress = {
            False: nn.Sequential(),
            True: lambda x: x.half().float()
        }[compress_fp16]
    def forward(self, input):
        return self.compress(self.fn(input))


class PrettrBertModel(BertPreTrainedModel):
    """
    Based on pytorch_pretrained_bert.BertModel, but with some extra goodies:
     - join_layer: layer to begin attention between query and document (0 for cross-attention in all layers)
     - compress_size: size of compression layer at join layer (0 for no compression)
     - compress_fp16: reduce size of floats in compression layer?
    """
    def __init__(self, config, join_layer=0, compress_size=0, compress_fp16=False):
        super(PrettrBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.join_layer = join_layer
        self.encoder = BertEncoder(config, join_layer, compress_size, compress_fp16)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        Based on pytorch_pretrained_bert.BertModel
        """
        if self.join_layer > 0:
            BAT, SEQ = attention_mask.shape
            join_mask = token_type_ids.reshape(BAT, 1, SEQ, 1) != token_type_ids.reshape(BAT, 1, 1, SEQ)
            join_mask = join_mask.float() * -10000.0
            join_mask = join_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        else:
            join_mask = None

        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoded_layers = self.forward_from_layer(embedding_output, attention_mask, from_layer=0, join_mask=join_mask)

        return [embedding_output] + encoded_layers

    def forward_from_layer(self, embedding_output, attention_mask, from_layer, join_mask=None):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=True,
                                      join_mask=join_mask,
                                      from_layer=from_layer)
        return encoded_layers

    def set_trainable(self, trainable, train_min_layer=0):
        if trainable:
            for param in self.parameters():
                param.requires_grad = trainable
            if train_min_layer > 0:
                for param in self.embeddings.parameters():
                    param.requires_grad = False
                for layer in self.encoder.layer[:train_min_layer-1]:
                    for param in layer.parameters():
                        param.requires_grad = False


class BertEncoder(nn.Module):
    def __init__(self, config, join_layer=0, compress_size=0, compress_fp16=False):
        super(BertEncoder, self).__init__()
        self.join_layer = join_layer
        self.layer = nn.ModuleList([BertLayer(config, l == (join_layer - 1), compress_size, compress_fp16) for l in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, join_mask=None, from_layer=0):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            if i < from_layer:
                continue
            hidden_states = layer_module(hidden_states, attention_mask, join_mask=(None if i >= self.join_layer else join_mask))
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertLayer(nn.Module):
    def __init__(self, config, include_compress=False, compress_size=0, compress_fp16=False):
        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        if not include_compress or compress_size == 0:
            self.selfencode = None
        else:
            self.selfencode = nn.Sequential(
                nn.Linear(config.hidden_size, compress_size),
                Act(ACT2FN[config.hidden_act], compress_fp16),
                nn.Linear(compress_size, config.hidden_size),
                BertLayerNorm(config.hidden_size) # , eps=config.layer_norm_eps
            )
        self.only_cls_output = False

    def forward(self, hidden_states, attention_mask, join_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, join_mask, only_cls_output=self.only_cls_output)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if self.selfencode is not None:
            layer_output = self.selfencode(layer_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, join_mask=None, only_cls_output=False):
        self_output = self.self(input_tensor, attention_mask, join_mask, only_cls_output=only_cls_output)
        if only_cls_output:
            attention_output = self.output(self_output, input_tensor[:, :1])
        else:
            attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, join_mask=None, only_cls_output=False):
        global last_attn_output
        if only_cls_output:
            mixed_query_layer = self.query(hidden_states[:, :1])
        else:
            mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        mixed_key_layer = self.key(hidden_states)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        mixed_value_layer = self.value(hidden_states)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        if join_mask is not None:
            attention_scores = attention_scores + join_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        last_attn_output = attention_probs

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer



class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
