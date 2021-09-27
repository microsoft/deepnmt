# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
import math

import functools

import numpy as np
encoder_ratio = 1.0
decoder_ratio = 1.0
tmp_file = 0

class TransformerEncoderLayer(nn.Module):

    def __init__(self, args, LayerNum=None):
        super().__init__()
        global tmp_file 
        
        self.args = args
        if not hasattr(self.args, 'mixed_precision'):
            self.args.mixed_precision = False
        if not hasattr(self.args, 'log_variance'):
            self.args.log_variance = False

        self.normalize_before = args.encoder_normalize_before
        self.embed_dim = args.encoder_embed_dim
        if LayerNum is not None and not self.normalize_before:

            assert 'adaptive' in args.admin_init_type

            self.self_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout, self_attention=True
            )

            self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
            self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

            if 'adaptive-profiling' == args.admin_init_type:
                if not tmp_file:
                    tmp_file = open(args.admin_init_path, 'w')
                self.attention_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.fc_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
            else:
                if not tmp_file:
                    tmp_file = open(args.admin_init_path, 'r')
                next_value = float(tmp_file.readline())
                print('encoder attn ratio: {}'.format(next_value))
                self.attention_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.attention_ratio_change.data.fill_(next_value)

                next_value = float(tmp_file.readline())
                print('encoder ffn ratio: {}'.format(next_value))
                self.fc_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.fc_ratio_change.data.fill_(next_value)

            self.self_attn_layer_norm = LayerNorm(self.embed_dim) 
            self.final_layer_norm = LayerNorm(self.embed_dim)

        else:
            assert args.admin_init_type == 'default'

            self.self_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout, self_attention=True
            )

            self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
            self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)

            self.self_attn_layer_norm = LayerNorm(self.embed_dim)
            self.final_layer_norm = LayerNorm(self.embed_dim)

        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            self.activation_dropout = getattr(args, 'relu_dropout', 0)

        if args.fp16:
            self.in_type=torch.half
        else:
            self.in_type=torch.float

    def upgrade_state_dict_named(self, state_dict, name):
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask=None):

        not_initialized = ('adaptive-profiling' == self.args.admin_init_type) and (1.0 == self.attention_ratio_change.min()) and self.training

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        if self.args.mixed_precision: 
            x = x.type(self.in_type)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), -1e8)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.args.mixed_precision:
            x = x.float()
        if self.args.log_variance:
            print('encoder attn: {}'.format(x.var()))
        if 'adaptive' in self.args.admin_init_type:
            if not_initialized:
                global encoder_ratio, tmp_file
                tmp_file.write('{}\n'.format(encoder_ratio))
                self.attention_ratio_change.data.fill_(encoder_ratio)
                print ('encoder attn ratio: {}'.format(encoder_ratio))
                input_std = np.var( (residual*self.attention_ratio_change) .clone().cpu().float().data.view(-1).numpy())
                output_std = np.var(x.clone().cpu().float().data.view(-1).numpy())
                encoder_ratio = np.sqrt(input_std + output_std)
            x = x + residual * self.attention_ratio_change
        else:
            x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)

        if self.args.mixed_precision: 
            x = x.type(self.in_type)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.args.mixed_precision:
            x = x.float()
        if self.args.log_variance:
            print('encoder ffn: {}'.format(x.var()))
        if 'adaptive' in self.args.admin_init_type:
            if not_initialized:
                tmp_file.write('{}\n'.format(encoder_ratio))
                self.fc_ratio_change.data.fill_(encoder_ratio)
                print ('encoder ffn ratio: {}'.format(encoder_ratio))
                input_std = np.var( (residual*self.fc_ratio_change) .clone().cpu().float().data.view(-1).numpy())
                output_std = np.var(x.clone().cpu().float().data.view(-1).numpy())
                encoder_ratio = np.sqrt(input_std + output_std)
            x = x + residual * self.fc_ratio_change
        else:
            x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, LayerNum=None):
        super().__init__()

        global tmp_file

        self.args = args
        if not hasattr(self.args, 'mixed_precision'):
            self.args.mixed_precision = False
        if not hasattr(self.args, 'log_variance'):
            self.args.log_variance = False

        self.normalize_before = args.decoder_normalize_before
        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        print(LayerNum)

        if LayerNum is not None and not self.normalize_before:

            assert 'adaptive' in args.admin_init_type

            self.self_attn = MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not self.cross_self_attention
            )

            assert not no_encoder_attn
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, 'encoder_embed_dim', None),
                vdim=getattr(args, 'encoder_embed_dim', None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True
            )

            self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
            self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
            
            if 'adaptive-profiling' == args.admin_init_type:
                if not tmp_file:
                    tmp_file = open(args.admin_init_path, 'w')
                self.self_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.encoder_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.fc_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
            else:
                if not tmp_file:
                    tmp_file = open(args.admin_init_path, 'r')
                next_value = float(tmp_file.readline())
                print('decoder self ratio: {}'.format(next_value))

                self.self_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.self_ratio_change.data.fill_(next_value)

                next_value = float(tmp_file.readline())
                print('decoder en ratio: {}'.format(next_value))
                self.encoder_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.encoder_ratio_change.data.fill_(next_value)

                next_value = float(tmp_file.readline())
                print('decoder ffn ratio: {}'.format(next_value))
                self.fc_ratio_change = nn.Parameter(torch.ones(self.embed_dim))
                self.fc_ratio_change.data.fill_(next_value)

            export = getattr(args, 'char_inputs', False)
            self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export) 
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export) 
            self.final_layer_norm = LayerNorm(self.embed_dim, export=export) 
        else:
            assert args.admin_init_type == 'default'

            self.self_attn = MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not self.cross_self_attention
            )

            assert not no_encoder_attn
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, 'encoder_embed_dim', None),
                vdim=getattr(args, 'encoder_embed_dim', None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True
            )
            
            self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
            self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

            export = getattr(args, 'char_inputs', False)
            self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
            if no_encoder_attn:
                self.encoder_attn = None
                self.encoder_attn_layer_norm = None
            else:
                self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
            self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            self.activation_dropout = getattr(args, 'relu_dropout', 0)


        self.need_attn = True

        self.onnx_trace = False

        if args.fp16:
            self.in_type=torch.half
        else:
            self.in_type=torch.float

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_attn=False,
        need_head_weights=False,
    ):
        not_initialized = ('adaptive-profiling' == self.args.admin_init_type) and (1.0 == self.self_ratio_change.min()) and self.training

        if need_head_weights:
            need_attn = True

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        if self.args.mixed_precision: 
            x = x.type(self.in_type)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        if self.cross_self_attention and not (incremental_state is not None and "prev_key" in self.self_attn._get_input_buffer(incremental_state)):
            if self_attn_mask is not None:
                self_attn_mask = torch.cat((x.new(x.size(0), encoder_out.size(0)).zero_(), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    encoder_padding_mask = self_attn_padding_mask.new(encoder_out.size(1), encoder_out.size(0)).zero_()
                self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x
        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.args.mixed_precision:
            x = x.float()
        if self.args.log_variance:
            print('decoder attn: {}'.format(x.var()))
        if 'adaptive' in self.args.admin_init_type:
            if not_initialized:
                global decoder_ratio, tmp_file
                tmp_file.write('{}\n'.format(decoder_ratio))
                self.self_ratio_change.data.fill_(decoder_ratio)
                print ('decoder self attn ratio: {}'.format(decoder_ratio))
                input_std = np.var( (residual*self.self_ratio_change).clone().cpu().float().data.view(-1).numpy())
                output_std = np.var(x.clone().cpu().float().data.view(-1).numpy())
                decoder_ratio = np.sqrt(input_std + output_std)
            x = x + residual * self.self_ratio_change
        else:
            x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)

            if self.args.mixed_precision: 
                x = x.type(self.in_type)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state[:2]
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.args.mixed_precision:
                x = x.float()
            if self.args.log_variance:
                print('decoder enatt: {}'.format(x.var()))
            if 'adaptive' in self.args.admin_init_type:
                if not_initialized:
                    tmp_file.write('{}\n'.format(decoder_ratio))
                    self.encoder_ratio_change.data.fill_(decoder_ratio)
                    print ('decoder encoder attn ratio: {}'.format(decoder_ratio))
                    input_std = np.var( (residual*self.encoder_ratio_change).clone().cpu().float().data.view(-1).numpy())
                    output_std = np.var(x.clone().cpu().float().data.view(-1).numpy())
                    decoder_ratio = np.sqrt(input_std + output_std)
                x = x + residual * self.encoder_ratio_change
            else:
                x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)

        if self.args.mixed_precision: 
            x = x.type(self.in_type)
        bx = self.fc1(x)
        hx = self.activation_fn(bx)
        x = F.dropout(hx, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.args.mixed_precision:
            x = x.float()
        if self.args.log_variance:
            print('decoder ffn: {}'.format(x.var()))
        if 'adaptive' in self.args.admin_init_type:
            if not_initialized:
                tmp_file.write('{}\n'.format(decoder_ratio))
                self.fc_ratio_change.data.fill_(decoder_ratio)
                print ('decoder ffn ratio: {}'.format(decoder_ratio))
                input_var = np.var( (residual * self.fc_ratio_change) .clone().cpu().float().data.view(-1).numpy())
                output_var = np.var(x.clone().cpu().float().data.view(-1).numpy())
                decoder_ratio = np.sqrt(input_var + output_var)
            x1 = x + residual * self.fc_ratio_change
        else:
            x1 = residual + x
        x2 = self.maybe_layer_norm(self.final_layer_norm, x1, after=True)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            if self_attn_padding_mask is not None:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"], saved_state["prev_key_padding_mask"]
            else:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x2, attn, self_attn_state

        return x2, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
