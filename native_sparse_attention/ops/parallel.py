# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Union

import torch
import triton
import triton.language as tl
from einops import rearrange
from fla.ops.common.utils import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_lens,
    prepare_token_indices,
)
from fla.ops.utils import mean_pooling
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous

from native_sparse_attention.ops.utils import _bitonic_merge

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None


# 压缩注意力的前向传播，返回输出o 和 softmax 分母的 log-sum-exp(lse, 用于后续的反向传播)
@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})  # 根据是否提供offsets参数来决定是否使用偏移量
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]
    ],  # 自动调优配置,测试不同的warp数量
    key=["BS", "BK", "BV"],  # 调优的关键参数
)
@triton.jit
def parallel_nsa_compression_fwd_kernel(
    # 输入参数
    q,  # Query矩阵
    k,  # Key矩阵
    v,  # Value矩阵
    o,  # 输出矩阵
    lse,  # Log Sum Exp结果
    scale,  # 缩放因子
    offsets,  # 偏移量数组
    token_indices,  # token索引
    chunk_offsets,
    T,  # 序列长度
    # 常量参数
    H: tl.constexpr,  # KV head数目
    HQ: tl.constexpr,  # Q head数目
    G: tl.constexpr,  # Group size（一个K对应多少个不同的Q）
    K: tl.constexpr,  # Query和Key维度
    V: tl.constexpr,  # Value维度
    BC: tl.constexpr,  # 压缩块大小
    BS: tl.constexpr,  # Block大小
    BK: tl.constexpr,  # Key块大小
    BV: tl.constexpr,  # Value块大小
    USE_OFFSETS: tl.constexpr,  # 是否使用偏移量的标志
):
    # 获取程序ID，定位数据块
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H  # Batch索引和Group索引

    # 根据是否使用偏移量计算序列范围
    if USE_OFFSETS:
        i_n, i_t = (
            tl.load(token_indices + i_t * 2).to(tl.int32),
            tl.load(token_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int32),
            tl.load(offsets + i_n + 1).to(tl.int32),
        )
        T = eos - bos  # 计算实际序列长度
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T  # 使用固定长度
        boc = i_b * tl.cdiv(T, BS)

    # 创建Query指针并加载Query块
    # 通过block_ptr 去理解 offset, tl.make_block_ptr(base_ptr, shape, strides, offsets, block_shape, order)
    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)  # 应用缩放

    # 计算压缩表示相关参数
    # the number of compression representations in total
    TC = tl.cdiv(T, BS)  # 总压缩块数
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS  # 需要处理的压缩块数（开头的BS个token以内无需计算compression attention）

    # 创建输出指针
    p_o = tl.make_block_ptr(
        o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)
    )
    
    # 初始化输出缓冲区和注意力计算所需的变量，这里使用online softmax算法
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)  # 输出缓冲区
    # max scores for the current block
    b_m = tl.full([G], float("-inf"), dtype=tl.float32)  # 存储最大得分
    # lse = log(acc) + m
    b_acc = tl.zeros([G], dtype=tl.float32)  # 累积器，用于计算softmax分母

    # 按压缩块进行循环处理
    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)  # 生成当前批次的索引

        # 创建Key和Value的指针
        p_k = tl.make_block_ptr(
            k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (boc * H + i_h) * V,
            (TC, V),
            (H * V, 1),
            (i_c, i_v * BV),
            (BC, BV),
            (1, 0),
        )

        # 加载Key和Value块
        b_k = tl.load(p_k, boundary_check=(0, 1))  # [BK, BC]        
        b_v = tl.load(p_v, boundary_check=(0, 1))  # [BC, BV]
        
        # 计算注意力得分
        b_s = tl.dot(b_q, b_k)  # [G, BC]
        
        # 对超出有效范围的位置设置为负无穷
        b_s = tl.where((o_c < NC)[None, :], b_s, float("-inf"))

        # 更新最大得分和计算相关系数
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m  #更新最大值[G]
        b_r = tl.exp(b_mp - b_m)  # 计算重缩放因子
        b_p = tl.exp(b_s - b_m[:, None])  # 计算注意力权重 [G, BC]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)  # 更新累积和[G]

        # 更新输出
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)  # [G, BV]

        b_mp = b_m
        
    # 处理最终结果
    if NC == 0:  # 如果没有需要处理的压缩块
        b_lse = tl.zeros([G], dtype=tl.float32)
    else:
        b_o = b_o / b_acc[:, None]  # 归一化输出
        b_lse = b_m + tl.log(b_acc)  # 计算log-sum-exp

    # 存储结果
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))  # 存储输出
    if i_v == 0:  # 只在第一个value保存lse，因为对于一个v向量，lse仅需计算一次，不需要每个block计算一次
        tl.store(
            lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G),
            b_lse.to(lse.dtype.element_ty),
        )


# 压缩注意力的反向传播，计算压缩注意力输出对q部分的梯度
@triton.heuristics({
    "USE_OFFSETS": lambda args: args["offsets"] is not None,
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_compression_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    offsets,
    token_indices,
    chunk_offsets,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = (
            tl.load(token_indices + i_t * 2).to(tl.int32),
            tl.load(token_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int32),
            tl.load(offsets + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * B * T + bos + i_t) * HQ * K

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)
        p_k = tl.make_block_ptr(
            k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (boc * H + i_h) * V,
            (V, TC),
            (1, H * V),
            (i_v * BV, i_c),
            (BV, BC),
            (0, 1),
        )
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BC]
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_p = tl.exp(b_s - b_lse[:, None])
        b_p = tl.where((o_c < NC)[None, :], b_p, 0)

        # [G, BV] @ [BV, BC] -> [G, BC]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [G, BC] @ [BC, BK] -> [G, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


# 压缩注意力的反向传播，计算压缩注意力输出对kv部分的梯度
@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_compression_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    offsets,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_c = (
            tl.load(chunk_indices + i_c * 2).to(tl.int32),
            tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int32),
            tl.load(offsets + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)

    p_k = tl.make_block_ptr(
        k + (boc * H + i_h) * K, (TC, K), (H * K, 1), (i_c * BC, 0), (BC, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + (boc * H + i_h) * V,
        (TC, V),
        (H * V, 1),
        (i_c * BC, i_v * BV),
        (BC, BV),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * B * T * H + boc * H + i_h) * K,
        (TC, K),
        (H * K, 1),
        (i_c * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (i_v * B * T * H + boc * H + i_h) * V,
        (TC, V),
        (H * V, 1),
        (i_c * BC, i_v * BV),
        (BC, BV),
        (1, 0),
    )

    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    # [BC, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)

    for i in range(i_c * BC * BS, T):
        o_c = i_c * BC + tl.arange(0, BC)

        p_q = tl.make_block_ptr(
            q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
        )
        # [G, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        p_do = tl.make_block_ptr(
            do + (bos + i) * HQ * V,
            (HQ, V),
            (V, 1),
            (i_h * G, i_v * BV),
            (G, BV),
            (1, 0),
        )
        p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [G]
        b_lse = tl.load(p_lse)
        b_delta = tl.load(p_delta)
        # [BC, G]
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_p = tl.exp(b_s - b_lse[None, :])
        b_p = tl.where((i >= max(0, (o_c + 1) * BS - 1))[:, None], b_p, 0)
        # [BC, G] @ [G, BV] -> [BC, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BC, BV] @ [BV, G] -> [BC, G]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BC, G]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BC, G] @ [G, BK] -> [BC, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# 与之前torch的实现不同，这里的triton实现将 topk KV block 选择操作独立出来，更方便优化 topk 这个本身非常耗时的算子
@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK"],
)
@triton.jit
def parallel_nsa_kernel_topk(
    q,
    k,
    lse,
    scale,
    block_indices,
    offsets,
    token_indices,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = (
            tl.load(token_indices + i_t * 2).to(tl.int32),
            tl.load(token_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int32),
            tl.load(offsets + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS
    ################################
    # 1. lse computation
    ################################
    if lse is not None:
        b_lse = tl.load(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G))
    else:
        # max scores for the current block
        b_m = tl.full([G], float("-inf"), dtype=tl.float32)
        # lse = log(acc) + m
        b_acc = tl.zeros([G], dtype=tl.float32)
        for i_c in range(0, NC, BC):
            o_c = i_c + tl.arange(0, BC)

            p_k = tl.make_block_ptr(
                k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
            )
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))

            # [G, BC]
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((o_c < NC)[None, :], b_s, float("-inf"))

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = tl.exp(b_mp - b_m)
            # [G, BC]
            b_p = tl.exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            b_mp = b_m
        if NC == 0:
            b_lse = tl.zeros([G], dtype=tl.float32)
        else:
            b_lse = b_m + tl.log(b_acc)

    ################################
    # 2. topk selection
    ################################
    # [BC]
    b_i = tl.full([BC], -1, dtype=tl.float32)
    o_i = tl.zeros([BC], dtype=tl.int32)
    m_i = tl.arange(0, BC) < BC // 2
    for i_c in range(0, i_t // BS + 1, BC):
        o_c = i_c + tl.arange(0, BC)

        p_k = tl.make_block_ptr(
            k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
        )
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((i_t // BS > o_c)[None, :], b_s, float("-inf"))
        # [G, BC]
        b_p = tl.where(
            (i_t // BS == o_c)[None, :], float(1.0), tl.exp(b_s - b_lse[:, None])
        )
        # the importance scores of the current block
        # [BC]
        b_i, b_ip = tl.sum(b_p, 0), b_i
        o_i, o_ip = tl.where(o_c <= i_t // BS, o_c + 1, 0), o_i

        n_dims: tl.constexpr = tl.standard._log2(b_i.shape[0])
        for i in tl.static_range(1, n_dims):
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)

        if i_c != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims)
            b_i_new = b_ip * m_i + b_i * (1 - m_i)
            o_i_new = o_ip * m_i + o_i * (1 - m_i)
            b_i, o_i = _bitonic_merge(
                b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)

    m_top = tl.arange(0, BC // S) == 0
    b_top = tl.sum(m_top[:, None] * tl.reshape(o_i - 1, [BC // S, S]), 0)

    p_b = tl.make_block_ptr(
        block_indices + (bos + i_t) * H * S, (H * S,), (1,), (i_h * S,), (S,), (0,)
    )
    tl.store(p_b, b_top.to(p_b.dtype.element_ty))


# 选择性注意力和滑动窗口注意力的前向传播，只对选定的块和局部窗口计算细粒度注意力，输出选择性注意力(o_slc)和滑动窗口注意力(o_swa)的输出
@triton.heuristics({
    "USE_OFFSETS": lambda args: args["offsets"] is not None,
    "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
)
@triton.jit
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    block_counts,
    offsets,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = (
            tl.load(token_indices + i_t * 2).to(tl.int32),
            tl.load(token_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int32),
            tl.load(offsets + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o = tl.make_block_ptr(
        o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)
    )
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(
                v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0)
            )
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BS, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where(
                (i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float("-inf")
            )

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = tl.exp(b_mp - b_m)
            # [G, BS]
            b_p = tl.exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            # [G, BV]
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

            b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += tl.log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


# 高效生成kv block的 attention 掩码
@triton.heuristics({
    "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor)
})
@triton.jit
def parallel_nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(
            block_counts + i_b * T * H + i_t * H + i_h
        )
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS and b_i >= 0:
        tl.store(
            block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i,
            b_m.to(block_mask.dtype.element_ty),
        )


# 预处理反向传播的辅助数据，返回 delta 值，用于后续梯度计算
@triton.jit
def parallel_nsa_bwd_kernel_preprocess(o, do, delta, B: tl.constexpr, V: tl.constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


# 选择性注意力和滑动注意力的反向传播，同时处理选择性和滑动窗口注意力对q 的梯度
@triton.heuristics({
    "USE_OFFSETS": lambda args: args["offsets"] is not None,
    "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    block_indices,
    block_counts,
    offsets,
    token_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = (
            tl.load(token_indices + i_t * 2).to(tl.int32),
            tl.load(token_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int32),
            tl.load(offsets + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * B * T + bos + i_t) * HQ * K
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(
                v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1)
            )
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BV, BS]
            b_v = tl.load(p_v, boundary_check=(0, 1))

            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p, 0)

            # [G, BV] @ [BV, BS] -> [G, BS]
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            # [G, BS] @ [BS, BK] -> [G, BK]
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


# 选择性注意力和滑动窗口注意力的反向传播，同时处理选择性和滑动窗口注意力对 kv 的梯度
@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    block_mask,
    offsets,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_s = (
            tl.load(chunk_indices + i_s * 2).to(tl.int32),
            tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int32),
            tl.load(offsets + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_s * BS, i_v * BV),
        (BS, BV),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * B * T * H + bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_s * BS, 0),
        (BS, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_s * BS, i_v * BV),
        (BS, BV),
        (1, 0),
    )

    # [BS, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H * M + i_h * M + i_s)
        if b_m:
            p_q = tl.make_block_ptr(
                q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
            )
            # [G, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)

            p_do = tl.make_block_ptr(
                do + (bos + i) * HQ * V,
                (HQ, V),
                (V, 1),
                (i_h * G, i_v * BV),
                (G, BV),
                (1, 0),
            )
            p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            # [G, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [G]
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            # [BS, G]
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = tl.exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p, 0)
            # [BS, G] @ [G, BV] -> [BS, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BS, BV] @ [BV, G] -> [BS, G]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BS, G]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BS, G] @ [G, BK] -> [BS, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def parallel_nsa_compression_fwd(
    q: torch.Tensor,  # 查询张量 [B, T, HQ, K]
    k: torch.Tensor,  # 键张量 [B, T, H, K]
    v: torch.Tensor,  # 值张量 [B, T, H, V]
    block_size: int,  # 块大小
    scale: float,  # 缩放因子
    offsets: Optional[torch.LongTensor] = None,  # 变长序列的偏移量
    token_indices: Optional[torch.LongTensor] = None,  # token的索引
):
    # 解析输入张量的维度
    B, T, HQ, K, V = (
        *q.shape,
        v.shape[-1],
    )  # batch_size, seq_len, query_heads, key_dim, value_dim
    H = k.shape[2]  # key_heads
    G = HQ // H  # 每个key head对应的query head数量(group size)

    # 设置block大小
    BC = BS = block_size  # BC: 压缩block大小, BS: 选择block大小

    # 根据GPU能力设置不同的block维度
    if torch.cuda.get_device_capability()[0] >= 9:  # 对于新型号GPU(如40系)
        BK = min(256, triton.next_power_of_2(K))  # key维度的block大小
        BV = min(256, triton.next_power_of_2(V))  # value维度的block大小
    else:  # 对于老型号GPU
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))

    # 计算在K和V维度上需要的block数量
    NK = triton.cdiv(K, BK)  # 向上取整除法
    NV = triton.cdiv(V, BV)

    # 确保key维度不超过限制
    assert NK == 1, "The key dimension can not be larger than 256"

    chunk_offsets = prepare_chunk_offsets(offsets, BS) if offsets is not None else None

    # 定义triton kernel的计算网格
    # - T: 序列长度维度，每个位置独立计算其注意力输出，可以并行
    # - NV: value维度的block数，可以并行计算不同 value block 的输出
    # - B*H: batch和Group维度，batch 内的不同样本和不同 attention head 可以完全并行
    grid = (T, NV, B * H)

    # 创建输出张量
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)  # 输出张量
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)  # log-sum-exp值

    # 调用triton kernel进行前向计算
    parallel_nsa_compression_fwd_kernel[grid](
        # 输入张量
        q=q,
        k=k,
        v=v,
        # 输出张量
        o=o,
        lse=lse,
        scale=scale,  # 缩放因子
        offsets=offsets,  # 序列偏移量
        token_indices=token_indices,  # token索引
        chunk_offsets=chunk_offsets,
        # 维度参数
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        # block相关参数
        BC=BC,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    return o, lse


def parallel_nsa_compression_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_size: int = 64,
    scale: float = None,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, HQ, K, V = *q.shape, v.shape[-1]
    H = k.shape[2]
    G = HQ // H
    BC = BS = block_size
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NV = triton.cdiv(V, BV)
    if offsets is not None:
        lens = prepare_lens(offsets)
        chunk_indices = torch.cat([
            torch.arange(n) for n in triton.cdiv(triton.cdiv(lens, BS), BC).tolist()
        ])
        chunk_indices = torch.stack(
            [chunk_indices.eq(0).cumsum(0) - 1, chunk_indices], 1
        ).to(offsets)
        chunk_offsets = prepare_chunk_offsets(offsets, BS)
        NC = len(chunk_indices)
    else:
        chunk_indices, chunk_offsets = None, None
        NC = triton.cdiv(triton.cdiv(T, BS), BC)

    delta = parallel_nsa_bwd_preprocess(o, do)

    dq = torch.empty(
        NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device
    )
    grid = (T, NV, B * H)
    parallel_nsa_compression_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        scale=scale,
        offsets=offsets,
        token_indices=token_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BC=BC,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dq = dq.sum(0)

    dk = torch.empty(
        NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device
    )
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NC, B * H)
    parallel_nsa_compression_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        offsets=offsets,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BC=BC,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dk = dk.sum(0)
    return dq, dk, dv


# 压缩注意力的 PyTorch 自动微分接口
class ParallelNSACompressionFunction(torch.autograd.Function):
    @staticmethod
    @contiguous  # 确保输入张量内存连续
    @autocast_custom_fwd  # 控制前向传播的精度
    def forward(ctx, q, k, v, block_size, scale, offsets):
        # 保存输入张量的dtype,用于后续类型转换
        ctx.dtype = q.dtype

        # 处理变长序列
        # 例如 offsets=[0,2,6] 表示:
        # - 第1个序列有2个token (0到2)
        # - 第2个序列有4个token (2到6)
        # token_indices 会被转换为:
        # [[0,0], [0,1], [1,0], [1,1], [1,2], [1,3]]
        # ======
        # 2-d sequence indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        # 调用前向传播计算函数
        # 返回输出o和log-sum-exp值lse
        o, lse = parallel_nsa_compression_fwd(
            q=q,
            k=k,
            v=v,
            block_size=block_size,
            scale=scale,
            offsets=offsets,
            token_indices=token_indices,
        )

        # 保存反向传播需要的张量和参数
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.offsets = offsets
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale

        # 返回计算结果,确保输出类型与输入一致
        return o.to(q.dtype), lse

    @staticmethod
    @contiguous  # 确保输入张量内存连续
    @autocast_custom_bwd  # 控制反向传播的精度
    def backward(ctx, do, *args):
        # 获取保存的张量
        q, k, v, o, lse = ctx.saved_tensors

        # 计算梯度
        # do是输出的梯度
        # 返回查询(dq)、键(dk)、值(dv)的梯度
        dq, dk, dv = parallel_nsa_compression_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_size=ctx.block_size,
            scale=ctx.scale,
            offsets=ctx.offsets,
            token_indices=ctx.token_indices,
        )

        # 返回每个前向传播输入参数对应的梯度
        # None表示该参数不需要梯度(如block_size等)
        return dq.to(q), dk.to(k), dv.to(v), None, None, None


def parallel_nsa_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    scale: float = None,
    offsets: Optional[torch.LongTensor] = None,
) -> torch.LongTensor:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    S = triton.next_power_of_2(S)
    # here we set BC = BS, but beware that they are actually decoupled
    BC = BS = block_size
    BK = triton.next_power_of_2(K)

    block_indices = torch.zeros(B, T, H, S, dtype=torch.int32, device=q.device)
    token_indices = prepare_token_indices(offsets) if offsets is not None else None
    chunk_offsets = prepare_chunk_offsets(offsets, BS) if offsets is not None else None
    grid = (T, B * H)
    parallel_nsa_kernel_topk[grid](
        q=q,
        k=k,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        offsets=offsets,
        token_indices=token_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        S=S,
        BC=BC,
        BS=BS,
        BK=BK,
    )
    return block_indices


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    grid = (T, NV, B * H)
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    parallel_nsa_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        block_counts=block_counts,
        offsets=offsets,
        token_indices=token_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    return o, lse


def parallel_nsa_block_mask(
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    offsets: torch.LongTensor,
    block_size: int,
):
    B, T, H, S = block_indices.shape
    BS = block_size
    if offsets is not None:
        NS = triton.cdiv(prepare_lens(offsets).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(T, B, H * S)](
        block_indices=block_indices,
        block_counts=block_counts,
        block_mask=block_mask,
        T=T,
        H=H,
        S=S,
        BS=BS,
        NS=NS,
    )
    return block_mask


def parallel_nsa_bwd_preprocess(o: torch.Tensor, do: torch.Tensor):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float32)
    parallel_nsa_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_nsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    scale: float = None,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NV = triton.cdiv(V, BV)

    delta = parallel_nsa_bwd_preprocess(o, do)

    dq = torch.empty(
        NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device
    )
    grid = (T, NV, B * H)
    parallel_nsa_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        block_indices=block_indices,
        block_counts=block_counts,
        offsets=offsets,
        token_indices=token_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dq = dq.sum(0)

    if offsets is not None:
        chunk_indices = prepare_chunk_indices(offsets, BS)
        NS = len(chunk_indices)
    else:
        chunk_indices = None
        NS = triton.cdiv(T, BS)

    # [B, T, H, M]
    block_mask = parallel_nsa_block_mask(
        block_indices, block_counts, offsets, block_size
    )
    dk = torch.empty(
        NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device
    )
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NS, B * H)
    parallel_nsa_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        block_mask=block_mask,
        offsets=offsets,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        M=block_mask.shape[-1],
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dk = dk.sum(0)
    return dq, dk, dv


# 选择性和滑动窗口注意力的 PyTorch 自动微分接口
@torch.compile
class ParallelNSAFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_counts, block_size, scale, offsets):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        o, lse = parallel_nsa_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            offsets=offsets,
            token_indices=token_indices,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        ctx.offsets = offsets
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_indices=ctx.block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            scale=ctx.scale,
            offsets=ctx.offsets,
            token_indices=ctx.token_indices,
        )
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def parallel_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    scale: float = None,
    offsets: Optional[torch.LongTensor] = None,
):
    return ParallelNSACompressionFunction.apply(q, k, v, block_size, scale, offsets)


# 选择性和滑动窗口注意力，需要预先传入每个token被选中的block_indices，没有计算 compression 注意力
def parallel_nsa(
    q: torch.Tensor,  # 查询张量 [B, T, HQ, K] 或 [B, HQ, T, K]
    k: torch.Tensor,  # 键张量 [B, T, H, K] 或 [B, H, T, K]
    v: torch.Tensor,  # 值张量 [B, T, H, V] 或 [B, H, T, V]
    g_cmp: torch.Tensor,  # 各token位置的压缩注意力的门控分数 [B, T, HQ] 或 [B, HQ, T]
    g_slc: torch.Tensor,  # 各token位置的选择注意力的门控分数 [B, T, HQ] 或 [B, HQ, T]
    g_swa: torch.Tensor,  # 各token位置的滑动注意力的门控分数 [B, T, HQ] 或 [B, HQ, T]
    block_indices: Optional[torch.LongTensor] = None,
    block_counts: Union[torch.LongTensor, int] = 16,  # 每个查询选择的块数
    block_size: int = 64,  # 选择的块大小
    window_size: int = 0,  # 滑动窗口大小
    scale: Optional[float] = None,  # 注意力分数的缩放因子
    cu_seqlens: Optional[torch.LongTensor] = None,  # 用于变长序列的累积长度
    head_first: bool = False,  # 输入张量是否为head-first格式
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_cmp (torch.Tensor):
            Gate score for compressed attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
            If `g_cmp` is provided, the passed `block_indices` will be ignored.
        block_counts (Optional[Union[torch.LongTensor, int]]):
            Number of selected blocks for each query.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=False` else `[B, H, T]`,
            each query can select the same number of blocks.
            If not provided, it will default to 16.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    # 确保提供了block_counts参数
    assert block_counts is not None, "block counts must be provided for selection"

    # 如果未提供scale，使用默认的缩放因子
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # 使用cu_seqlens时，批次大小必须为1
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    # 如果是head_first格式，转换所有输入张量为batch_first格式
    if head_first:
        q, k, v = map(lambda x: rearrange(x, "b h t d -> b t h d"), (q, k, v))
        g_cmp, g_slc, g_swa = map(
            lambda x: rearrange(x, "b h t -> b t h") if x is not None else None,
            (g_cmp, g_slc, g_swa),
        )
        if not isinstance(block_counts, int):
            block_counts = rearrange(block_counts, "b h t -> b t h")
    # 确保分组大小是16的倍数
    assert q.shape[2] % (k.shape[2] * 16) == 0, (
        "Group size must be a multiple of 16 in NSA"
    )

    # 对键值对进行压缩
    k_cmp, v_cmp = (
        mean_pooling(k, block_size, cu_seqlens),
        mean_pooling(v, block_size, cu_seqlens),
    )
    o_cmp, lse_cmp = None, None
    # 如果提供了压缩注意力的gate，就计算压缩注意力
    if g_cmp is not None:
        o_cmp, lse_cmp = parallel_nsa_compression(
            q=q,
            k=k_cmp,
            v=v_cmp,
            block_size=block_size,
            scale=scale,
            offsets=cu_seqlens,
        )
        if block_indices is not None:
            warnings.warn("`block_indices` will be ignored when `g_cmp` is provided")
        # 计算top-k块索引，选择最相关的块
        block_indices = parallel_nsa_topk(
            q=q,
            k=k_cmp,  # 使用压缩后的键
            lse=lse_cmp,  # 使用压缩注意力的softmax分母的log-sum-exp值
            block_counts=block_counts,  # 每个查询要选择的块数
            block_size=block_size,
            scale=scale,
            offsets=cu_seqlens,
        )
    # 计算选择注意力和滑动注意力
    # o_slc: 选择注意力的输出
    # o_swa: 滑动窗口注意力的输出
    o_slc = ParallelNSAFunction.apply(
        q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens
    )

    # 组合不同类型的注意力输出
    # 1. 首先使用选择注意力的门控值
    o = o_slc * g_slc.unsqueeze(-1)

    # 2. 如果有压缩注意力，添加压缩注意力的输出
    if o_cmp is not None:
        # addcmul: o += o_cmp * g_cmp.unsqueeze(-1)
        o = torch.addcmul(o, o_cmp, g_cmp.unsqueeze(-1))

    # 3. 如果使用滑动窗口，添加滑动窗口注意力的输出
    if window_size > 0:
        if cu_seqlens is not None:
            max_seqlen = q.shape[1]
            o_swa = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(window_size - 1, 0),
            ).unsqueeze(0)
        else:
            o_swa = flash_attn_func(
                q, k, v, causal=True, window_size=(window_size - 1, 0)
            )
        o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))

    # 如果输入是head_first格式，将输出转回head_first格式
    if head_first:
        o = rearrange(o, "b t h d -> b h t d")
    return o
