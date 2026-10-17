// port-lint: source utils.rs
package io.github.kotlinmania.candletransformers

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.cat

/**
 * Shared utilities: repeatKv, applyRepeatPenalty, causal mask.
 */

/**
 * Build a causal attention mask of shape `(seqLen, kvLen)` where
 * `kvLen = indexPos + seqLen`.
 *
 * `mask[i][j] = 1` means query `i` must **not** attend to key `j`.
 *
 * - `indexPos == 0`: classic square `(seqLen, seqLen)` mask.
 * - `indexPos > 0`: rectangular mask for prefix KV caching — the first
 *   `indexPos` columns are all-zero (every query attends to all cached prefix
 *   keys) and the last `seqLen` columns form the standard causal triangle.
 *
 * All models that maintain a KV cache should use this function so that
 * batched user-turn prefill works correctly after prefix restoration.
 */
public fun buildCausalMask(
    seqLen: Int,
    indexPos: Int,
    device: Device,
): Tensor {
    val kvLen = indexPos + seqLen
    val mask = UByteArray(seqLen * kvLen)
    var idx = 0
    for (i in 0 until seqLen) {
        for (j in 0 until kvLen) {
            mask[idx++] = if (j > indexPos + i) 1u else 0u
        }
    }
    return Tensor.fromSlice(mask, Shape(seqLen, kvLen), device)
}

/**
 * Build an *additive* causal attention mask of shape `(1, 1, tgt, tgt + offset)`,
 * where masked positions are `-inf` and visible ones are `0`.
 *
 * This is the counterpart to [buildCausalMask] for models that apply the mask
 * with `scores.broadcastAdd(mask)` rather than by selecting on a `UByte` mask.
 *
 * The buffer holds `tgt * (tgt + offset)` values and does not depend on the batch
 * size, so the leading dim stays `1` and is broadcast over the batch by the caller.
 * Shaping it `(b, 1, ..)` instead claims `b` times more elements than the buffer
 * holds, which `Tensor.fromSlice` does not reject; see
 * https://github.com/huggingface/candle/issues/3582.
 *
 * `sw` optionally restricts each query to a sliding window of `sw` preceding keys.
 */
public fun buildAdditiveCausalMask(
    tgt: Int,
    offset: Int,
    sw: Int?,
    device: Device,
    dtype: DType,
): Tensor {
    val minf = Float.NEGATIVE_INFINITY
    val totalCols = tgt + offset
    val mask = FloatArray(tgt * totalCols)
    var idx = 0
    for (i in 0 until tgt) {
        for (j in 0 until totalCols) {
            val pastOk = j <= i + offset
            // Within the window iff `(i + offset) - j <= w`, rearranged to
            // `j + w >= i + offset` to stay in non-negative integers (no signed casts, no
            // subtraction underflow when `j > i + offset`).
            val swOk =
                when (sw) {
                    null -> true
                    else -> j + sw >= i + offset
                }
            mask[idx++] =
                if (pastOk && swOk) {
                    0.0f
                } else {
                    minf
                }
        }
    }
    return Tensor.fromSlice(mask, Shape(1, 1, tgt, totalCols), device).toDtype(dtype)
}

public fun applyRepeatPenalty(
    logits: Tensor,
    penalty: Float,
    context: UIntArray,
): Tensor {
    val device = logits.device()
    val logitsArr = logits.toDtype(DType.F32).toVec1F32()
    val alreadySeen = HashSet<UInt>()
    for (tokenId in context) {
        if (alreadySeen.contains(tokenId)) {
            continue
        }
        alreadySeen.add(tokenId)
        val idx = tokenId.toInt()
        if (idx in logitsArr.indices) {
            val logit = logitsArr[idx]
            logitsArr[idx] =
                if (logit >= 0.0f) {
                    logit / penalty
                } else {
                    logit * penalty
                }
        }
    }
    val logitsLen = logitsArr.size
    return Tensor.fromSlice(logitsArr, Shape(logitsLen), device)
}

/**
 * Repeats a key or value tensor for grouped query attention
 * The input tensor should have a shape `(batch, numKvHeads, seqLen, headDim)`,
 */
public fun repeatKv(
    xs: Tensor,
    nRep: Int,
): Tensor =
    if (nRep == 1) {
        xs
    } else {
        val dims = xs.dims4()
        val bSz = dims.component1()
        val nKvHead = dims.component2()
        val seqLen = dims.component3()
        val headDim = dims.component4()
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor.cat(List(nRep) { xs }, 2).reshape(Shape(bSz, nKvHead * nRep, seqLen, headDim))
    }
