// port-lint: source models/quantized_t5.rs
package io.github.kotlinmania.candletransformers.models

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candlenn.Activation
import io.github.kotlinmania.candlenn.softmaxLastDim
import io.github.kotlinmania.candletransformers.QuantizedEmbedding
import io.github.kotlinmania.candletransformers.QuantizedVarBuilder
import io.github.kotlinmania.tracing.Level
import io.github.kotlinmania.tracing.Span
import io.github.kotlinmania.tracing.span
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.sqrt

/**
 * T5 model implementation with quantization support.
 *
 * T5 is an encoder-decoder model pre-trained on a multi-task mixture of supervised
 * and unsupervised tasks. This implementation provides quantization for reduced
 * memory and compute requirements.
 */

private fun defaultRelativeAttentionMaxDistance(): Int = 128

private fun defaultIsDecoder(): Boolean = false

private fun defaultUseCache(): Boolean = true

private fun defaultTieWordEmbeddings(): Boolean = true

private fun getMask(
    size: Int,
    device: Device,
): Tensor {
    val mask = UByteArray(size * size)
    var idx = 0
    for (i in 0 until size) {
        for (j in 0 until size) {
            mask[idx++] = if (j > i) 1u else 0u
        }
    }
    return Tensor.fromSlice(mask, Shape(size, size), device)
}

private fun maskedFill(
    onFalse: Tensor,
    mask: Tensor,
    onTrue: Float,
): Tensor {
    val shape = mask.shape()
    val onTrueTensor = Tensor.new(floatArrayOf(onTrue), Shape.SCALAR, onFalse.device()).broadcastAs(shape)
    return mask.whereCond(onTrueTensor, onFalse)
}

public data class QuantizedT5Config(
    public val vocabSize: Int = 32128,
    public val dModel: Int = 512,
    public val dKv: Int = 64,
    public val dFf: Int = 2048,
    public val numLayers: Int = 6,
    public val numDecoderLayers: Int? = null,
    public val numHeads: Int = 8,
    public val relativeAttentionNumBuckets: Int = 32,
    public val relativeAttentionMaxDistance: Int = defaultRelativeAttentionMaxDistance(),
    public val dropoutRate: Double = 0.1,
    public val layerNormEpsilon: Double = 1e-6,
    public val initializerFactor: Double = 1.0,
    public val feedForwardProj: T5ActivationWithOptionalGating = T5ActivationWithOptionalGating.default(),
    public val tieWordEmbeddings: Boolean = defaultTieWordEmbeddings(),
    public val isDecoder: Boolean = defaultIsDecoder(),
    public val isEncoderDecoder: Boolean = true,
    public val useCache: Boolean = defaultUseCache(),
    public val padTokenId: Int = 0,
    public val eosTokenId: Int = 1,
    public val decoderStartTokenId: Int? = 0,
) {
    public companion object {
        public fun default(): QuantizedT5Config = QuantizedT5Config()
    }
}

private class QT5LayerNorm(
    private val weight: Tensor,
    private val varianceEpsilon: Double,
    private val span: Span,
) : Module {
    companion object {
        fun load(
            h: Int,
            eps: Double,
            vb: QuantizedVarBuilder,
        ): QT5LayerNorm {
            val weight = vb.get(h, "weight").dequantize(vb.device()).unwrap()
            return QT5LayerNorm(
                weight = weight,
                varianceEpsilon = eps,
                span = span(Level.TRACE, "layer-norm"),
            )
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            val dtype = xs.dtype()
            val xsF32 = xs.toDtype(DType.F32)
            // variance = hiddenStates.to(float32).pow(2).mean(-1, keepdim=true)
            val lastIdx = D.Minus1.toIndex(xsF32.shape(), "meanKeepdim")
            val variance = xsF32.sqr().meanKeepdim(lastIdx)
            val normed = xs.broadcastDiv(variance.affine(1.0, varianceEpsilon).sqrt())
            val casted = normed.toDtype(dtype)
            casted.broadcastMul(weight)
        } finally {
            enter.exit()
        }
    }
}

private class QT5DenseActDense(
    private val wi: TracingQMatMul,
    private val wo: TracingQMatMul,
    private val act: Activation,
    private val span: Span,
) : Module {
    companion object {
        fun load(
            vb: QuantizedVarBuilder,
            cfg: QuantizedT5Config,
        ): QT5DenseActDense {
            val wi = TracingQMatMul.new(cfg.dModel, cfg.dFf, vb.pp("wi"))
            val wo = TracingQMatMul.new(cfg.dFf, cfg.dModel, vb.pp("wo"))
            return QT5DenseActDense(
                wi = wi,
                wo = wo,
                act = Activation.Relu,
                span = span(Level.TRACE, "dense-act-dense"),
            )
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            val h = wi.forward(xs)
            val a = act.forward(h)
            wo.forward(a)
        } finally {
            enter.exit()
        }
    }
}

private class QT5DenseGatedActDense(
    private val wi0: TracingQMatMul,
    private val wi1: TracingQMatMul,
    private val wo: TracingQMatMul,
    private val act: Activation,
    private val span: Span,
) : Module {
    companion object {
        fun load(
            vb: QuantizedVarBuilder,
            cfg: QuantizedT5Config,
        ): QT5DenseGatedActDense {
            val wi0 = TracingQMatMul.new(cfg.dModel, cfg.dFf, vb.pp("wi_0"))
            val wi1 = TracingQMatMul.new(cfg.dModel, cfg.dFf, vb.pp("wi_1"))
            val wo = TracingQMatMul.new(cfg.dFf, cfg.dModel, vb.pp("wo"))
            return QT5DenseGatedActDense(
                wi0 = wi0,
                wi1 = wi1,
                wo = wo,
                act = cfg.feedForwardProj.activation,
                span = span(Level.TRACE, "dense-gated-act-dense"),
            )
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            val hiddenGelu = act.forward(wi0.forward(xs))
            val hiddenLinear = wi1.forward(xs)
            val prod = hiddenGelu.broadcastMul(hiddenLinear)
            wo.forward(prod)
        } finally {
            enter.exit()
        }
    }
}

private class QT5LayerFF(
    private val denseAct: QT5DenseActDense?,
    private val gatedDenseAct: QT5DenseGatedActDense?,
    private val layerNorm: QT5LayerNorm,
    private val span: Span,
) : Module {
    companion object {
        fun load(
            vb: QuantizedVarBuilder,
            cfg: QuantizedT5Config,
        ): QT5LayerFF {
            val layerNorm = QT5LayerNorm.load(cfg.dModel, cfg.layerNormEpsilon, vb.pp("layer_norm"))
            val (denseAct, gatedDenseAct) =
                if (cfg.feedForwardProj.gated) {
                    Pair(null, QT5DenseGatedActDense.load(vb.pp("DenseReluDense"), cfg))
                } else {
                    Pair(QT5DenseActDense.load(vb.pp("DenseReluDense"), cfg), null)
                }
            return QT5LayerFF(
                denseAct = denseAct,
                gatedDenseAct = gatedDenseAct,
                layerNorm = layerNorm,
                span = span(Level.TRACE, "layer-ff"),
            )
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            val ysNorm = layerNorm.forward(xs)
            val ys =
                when {
                    denseAct != null -> denseAct.forward(ysNorm)
                    gatedDenseAct != null -> gatedDenseAct.forward(ysNorm)
                    else -> ysNorm
                }
            xs + ys
        } finally {
            enter.exit()
        }
    }
}

private class QT5Attention(
    private val q: TracingQMatMul,
    private val k: TracingQMatMul,
    private val v: TracingQMatMul,
    private val o: TracingQMatMul,
    private val nHeads: Int,
    private val dKv: Int,
    private val relativeAttentionBias: QuantizedEmbedding?,
    private val relativeAttentionNumBuckets: Int,
    private val relativeAttentionMaxDistance: Int,
    private val innerDim: Int,
    private val useCache: Boolean,
    private var kvCache: Pair<Tensor, Tensor>?,
    private val span: Span,
    private val spanCache: Span,
    private val spanMm: Span,
    private val spanSm: Span,
) {
    companion object {
        fun load(
            hasRelativeAttentionBias: Boolean,
            decoder: Boolean,
            vb: QuantizedVarBuilder,
            cfg: QuantizedT5Config,
        ): QT5Attention {
            val innerDim = cfg.numHeads * cfg.dKv
            val q = TracingQMatMul.new(cfg.dModel, innerDim, vb.pp("q"))
            val k = TracingQMatMul.new(cfg.dModel, innerDim, vb.pp("k"))
            val v = TracingQMatMul.new(cfg.dModel, innerDim, vb.pp("v"))
            val o = TracingQMatMul.new(innerDim, cfg.dModel, vb.pp("o"))
            val relativeAttentionBias =
                if (hasRelativeAttentionBias) {
                    QuantizedEmbedding.new(
                        cfg.relativeAttentionNumBuckets,
                        cfg.numHeads,
                        vb.pp("relative_attention_bias"),
                    )
                } else {
                    null
                }
            return QT5Attention(
                q = q,
                k = k,
                v = v,
                o = o,
                nHeads = cfg.numHeads,
                dKv = cfg.dKv,
                relativeAttentionBias = relativeAttentionBias,
                relativeAttentionNumBuckets = cfg.relativeAttentionNumBuckets,
                relativeAttentionMaxDistance = cfg.relativeAttentionMaxDistance,
                innerDim = innerDim,
                useCache = cfg.useCache && decoder,
                kvCache = null,
                span = span(Level.TRACE, "attention"),
                spanCache = span(Level.TRACE, "attention-cache"),
                spanMm = span(Level.TRACE, "attention-mm"),
                spanSm = span(Level.TRACE, "attention-sm"),
            )
        }
    }

    fun forward(
        xs: Tensor,
        positionBias: Tensor?,
        keyValueStates: Tensor?,
        mask: Tensor?,
    ): Pair<Tensor, Tensor?> {
        // Performs Self-attention (if keyValueStates is null) or attention
        // over source sentence (provided by keyValueStates).
        val enter = span.enter()
        try {
            val kvInput = keyValueStates ?: xs
            val bSz = xs.dim(0)
            val qLen = xs.dim(1)
            val kvLenInitial = kvInput.dim(1)
            val qProj =
                q
                    .forward(xs)
                    .reshape(Shape(bSz, qLen, nHeads, dKv))
                    .transpose(1, 2)
                    .contiguous()
            var kProj =
                k
                    .forward(kvInput)
                    .reshape(Shape(bSz, kvLenInitial, nHeads, dKv))
                    .transpose(1, 2)
            var vProj =
                v
                    .forward(kvInput)
                    .reshape(Shape(bSz, kvLenInitial, nHeads, dKv))
                    .transpose(1, 2)

            if (useCache && keyValueStates == null) {
                val enterCache = spanCache.enter()
                try {
                    val cached = kvCache
                    if (cached != null) {
                        kProj = Tensor.cat(listOf(cached.first, kProj), 2)
                        vProj = Tensor.cat(listOf(cached.second, vProj), 2)
                    }
                    kvCache = Pair(kProj, vProj)
                } finally {
                    enterCache.exit()
                }
            }
            val kCont = kProj.contiguous()
            val vCont = vProj.contiguous()
            val scoresRaw =
                run {
                    val enterMm = spanMm.enter()
                    try {
                        qProj.matmul(kCont.t())
                    } finally {
                        enterMm.exit()
                    }
                }
            val scoresMasked =
                when (mask) {
                    null -> {
                        scoresRaw
                    }

                    else -> {
                        maskedFill(
                            scoresRaw,
                            mask
                                .unsqueeze(0)
                                .unsqueeze(0)
                                .repeat(Shape(bSz, nHeads, 1, 1)),
                            Float.NEGATIVE_INFINITY,
                        )
                    }
                }

            val (scoresWithBias, resolvedPositionBias) =
                when {
                    positionBias != null -> {
                        Pair(scoresMasked.broadcastAdd(positionBias), positionBias)
                    }

                    relativeAttentionBias == null -> {
                        Pair(scoresMasked, null)
                    }

                    else -> {
                        // This only handles the bidirectional case.
                        val kvLen = kCont.dim(2)
                        val qStart = if (useCache) (kvLen - qLen) else 0
                        val qEnd = kvLen
                        val numBuckets = relativeAttentionNumBuckets / 2
                        val maxExact = numBuckets / 2
                        val rows = qEnd - qStart
                        val relativePosition = UIntArray(rows * kvLen)
                        var idx = 0
                        val logDenom = ln(relativeAttentionMaxDistance.toFloat() / maxExact.toFloat())
                        for (i in qStart until qEnd) {
                            for (j in 0 until kvLen) {
                                val bucket =
                                    if (i < j) {
                                        if (j - i < maxExact) {
                                            (j - i + numBuckets).toUInt()
                                        } else {
                                            val b =
                                                (ln((j - i).toFloat() / maxExact.toFloat()) / logDenom) *
                                                    (numBuckets - maxExact).toFloat()
                                            min(
                                                maxExact + numBuckets + b.toInt(),
                                                relativeAttentionNumBuckets - 1,
                                            ).toUInt()
                                        }
                                    } else if (i - j < maxExact) {
                                        (i - j).toUInt()
                                    } else {
                                        val b =
                                            (ln((i - j).toFloat() / maxExact.toFloat()) / logDenom) *
                                                (numBuckets - maxExact).toFloat()
                                        (maxExact + b.toInt()).toUInt()
                                    }
                                relativePosition[idx++] = bucket
                            }
                        }
                        val relativeBuckets =
                            Tensor.fromSlice(
                                relativePosition,
                                Shape(rows, kvLen),
                                qProj.device(),
                            )
                        val computedBias =
                            relativeAttentionBias
                                .forward(relativeBuckets)
                                .permute(2, 0, 1)
                                .unsqueeze(0)
                        Pair(scoresMasked.broadcastAdd(computedBias), computedBias)
                    }
                }

            val attnWeights =
                run {
                    val enterSm = spanSm.enter()
                    try {
                        softmaxLastDim(scoresWithBias)
                    } finally {
                        enterSm.exit()
                    }
                }
            val attnOutput =
                attnWeights
                    .matmul(vCont)
                    .transpose(1, 2)
                    .reshape(Shape(bSz, qLen, innerDim))
            val projected = o.forward(attnOutput)
            return Pair(projected, resolvedPositionBias)
        } finally {
            enter.exit()
        }
    }

    fun clearKvCache() {
        kvCache = null
    }
}

private class QT5LayerSelfAttention(
    private val selfAttention: QT5Attention,
    private val layerNorm: QT5LayerNorm,
    private val span: Span,
) {
    companion object {
        fun load(
            h: Boolean,
            d: Boolean,
            vb: QuantizedVarBuilder,
            cfg: QuantizedT5Config,
        ): QT5LayerSelfAttention {
            val selfAttention = QT5Attention.load(h, d, vb.pp("SelfAttention"), cfg)
            val layerNorm = QT5LayerNorm.load(cfg.dModel, cfg.layerNormEpsilon, vb.pp("layer_norm"))
            return QT5LayerSelfAttention(
                selfAttention = selfAttention,
                layerNorm = layerNorm,
                span = span(Level.TRACE, "self-attn"),
            )
        }
    }

    fun forward(
        xs: Tensor,
        positionBias: Tensor?,
        mask: Tensor?,
    ): Pair<Tensor, Tensor?> {
        val enter = span.enter()
        return try {
            val normedXs = layerNorm.forward(xs)
            val (ys, nextBias) = selfAttention.forward(normedXs, positionBias, null, mask)
            Pair(xs + ys, nextBias)
        } finally {
            enter.exit()
        }
    }

    fun clearKvCache() {
        selfAttention.clearKvCache()
    }
}

private class QT5LayerCrossAttention(
    private val crossAttention: QT5Attention,
    private val layerNorm: QT5LayerNorm,
    private val span: Span,
) {
    companion object {
        fun load(
            decoder: Boolean,
            vb: QuantizedVarBuilder,
            cfg: QuantizedT5Config,
        ): QT5LayerCrossAttention {
            val crossAttention = QT5Attention.load(false, decoder, vb.pp("EncDecAttention"), cfg)
            val layerNorm = QT5LayerNorm.load(cfg.dModel, cfg.layerNormEpsilon, vb.pp("layer_norm"))
            return QT5LayerCrossAttention(
                crossAttention = crossAttention,
                layerNorm = layerNorm,
                span = span(Level.TRACE, "cross-attn"),
            )
        }
    }

    fun forward(
        hiddenStates: Tensor,
        positionBias: Tensor?,
        keyValueStates: Tensor,
    ): Pair<Tensor, Tensor?> {
        val enter = span.enter()
        return try {
            val normedHiddenStates = layerNorm.forward(hiddenStates)
            val (ys, nextBias) =
                crossAttention.forward(
                    normedHiddenStates,
                    positionBias,
                    keyValueStates,
                    null,
                )
            Pair(hiddenStates + ys, nextBias)
        } finally {
            enter.exit()
        }
    }

    fun clearKvCache() {
        crossAttention.clearKvCache()
    }
}

private class QT5Block(
    private val selfAttn: QT5LayerSelfAttention,
    private val crossAttn: QT5LayerCrossAttention?,
    private val ff: QT5LayerFF,
    private val span: Span,
) {
    companion object {
        fun load(
            hasRelativeAttentionBias: Boolean,
            decoder: Boolean,
            vb: QuantizedVarBuilder,
            cfg: QuantizedT5Config,
        ): QT5Block {
            val layerVb = vb.pp("layer")
            val selfAttn = QT5LayerSelfAttention.load(hasRelativeAttentionBias, decoder, layerVb.pp("0"), cfg)
            val crossAttn =
                if (cfg.isDecoder) {
                    QT5LayerCrossAttention.load(decoder, layerVb.pp("1"), cfg)
                } else {
                    null
                }
            val ffI = if (crossAttn != null) 2 else 1
            val ff = QT5LayerFF.load(layerVb.pp(ffI.toString()), cfg)
            return QT5Block(
                selfAttn = selfAttn,
                crossAttn = crossAttn,
                ff = ff,
                span = span(Level.TRACE, "block"),
            )
        }
    }

    fun forward(
        xs: Tensor,
        positionBias: Tensor?,
        encoderHiddenStates: Tensor?,
    ): Pair<Tensor, Tensor?> {
        val enter = span.enter()
        return try {
            val mask =
                if (crossAttn != null) {
                    val maskLen = xs.dim(1)
                    // If the input seq length is 1, no need for a mask, this is also helpful to avoid shape
                    // issues when using the KV cache in the decoder.
                    if (maskLen <= 1) {
                        null
                    } else {
                        getMask(maskLen, xs.device())
                    }
                } else {
                    null
                }
            var (currXs, nextBias) = selfAttn.forward(xs, positionBias, mask)
            if (crossAttn != null && encoderHiddenStates != null) {
                currXs = crossAttn.forward(currXs, null, encoderHiddenStates).first
            }
            val outXs = ff.forward(currXs)
            Pair(outXs, nextBias)
        } finally {
            enter.exit()
        }
    }

    fun clearKvCache() {
        selfAttn.clearKvCache()
        crossAttn?.clearKvCache()
    }
}

private class QT5Stack(
    private val block: List<QT5Block>,
    private val shared: QuantizedEmbedding,
    private val finalLayerNorm: QT5LayerNorm,
    private val span: Span,
) {
    companion object {
        fun load(
            decoder: Boolean,
            vb: QuantizedVarBuilder,
            shared: QuantizedEmbedding,
            cfg: QuantizedT5Config,
        ): QT5Stack {
            val block =
                (0 until cfg.numLayers).map { i ->
                    QT5Block.load(i == 0, decoder, vb.pp("block.$i"), cfg)
                }
            val finalLayerNorm =
                QT5LayerNorm.load(
                    cfg.dModel,
                    cfg.layerNormEpsilon,
                    vb.pp("final_layer_norm"),
                )
            return QT5Stack(
                block = block,
                shared = shared,
                finalLayerNorm = finalLayerNorm,
                span = span(Level.TRACE, "stack"),
            )
        }
    }

    fun forward(
        inputIds: Tensor,
        encoderHiddenStates: Tensor?,
    ): Tensor {
        val enter = span.enter()
        return try {
            var hiddenStates = shared.forward(inputIds)
            var positionBias: Tensor? = null
            for (b in block) {
                val res = b.forward(hiddenStates, positionBias, encoderHiddenStates)
                hiddenStates = res.first
                positionBias = res.second
            }
            finalLayerNorm.forward(hiddenStates)
        } finally {
            enter.exit()
        }
    }

    fun clearKvCache() {
        for (b in block) {
            b.clearKvCache()
        }
    }
}

public class QuantizedT5EncoderModel private constructor(
    private val encoder: QT5Stack,
    private val device: Device,
    private val span: Span,
) {
    public companion object {
        public fun load(
            vb: QuantizedVarBuilder,
            cfg: QuantizedT5Config,
        ): QuantizedT5EncoderModel {
            val sharedVb =
                if (vb.containsKey("shared.weight")) {
                    vb.pp("shared")
                } else {
                    vb.pp("decoder").pp("embed_tokens")
                }
            val shared = QuantizedEmbedding.new(cfg.vocabSize, cfg.dModel, sharedVb)
            val encoder = QT5Stack.load(false, vb.pp("encoder"), shared, cfg)
            return QuantizedT5EncoderModel(
                encoder = encoder,
                device = vb.device(),
                span = span(Level.TRACE, "encoder"),
            )
        }
    }

    public fun forward(inputIds: Tensor): Tensor {
        val enter = span.enter()
        return try {
            encoder.forward(inputIds, null)
        } finally {
            enter.exit()
        }
    }

    public fun device(): Device = device

    public fun clearKvCache() {
        encoder.clearKvCache()
    }
}

public class QuantizedT5ForConditionalGeneration private constructor(
    private val encoder: QT5Stack,
    private val decoder: QT5Stack,
    private val dModel: Int,
    private val tieWordEmbeddings: Boolean,
    private val lmHead: TracingQMatMul?,
    private val shared: QuantizedEmbedding,
    private val device: Device,
    private val spanDecode: Span,
    private val spanDecodeHead: Span,
) {
    public companion object {
        public fun load(
            vb: QuantizedVarBuilder,
            cfg: QuantizedT5Config,
        ): QuantizedT5ForConditionalGeneration {
            require(cfg.isEncoderDecoder)
            val dModel = cfg.dModel
            val sharedVb =
                if (vb.containsKey("shared.weight")) {
                    vb.pp("shared")
                } else {
                    vb.pp("decoder").pp("embed_tokens")
                }
            val shared = QuantizedEmbedding.new(cfg.vocabSize, cfg.dModel, sharedVb)

            val encoderCfg =
                cfg.copy(
                    isDecoder = false,
                    useCache = false,
                    isEncoderDecoder = false,
                )
            val encoder = QT5Stack.load(false, vb.pp("encoder"), shared, encoderCfg)

            val decoderCfg =
                cfg.copy(
                    isDecoder = true,
                    isEncoderDecoder = false,
                    numLayers = cfg.numDecoderLayers ?: cfg.numLayers,
                )
            val decoder = QT5Stack.load(true, vb.pp("decoder"), shared, decoderCfg)

            val tieWordEmbeddings = cfg.tieWordEmbeddings
            val lmHead =
                if (tieWordEmbeddings) {
                    null
                } else {
                    TracingQMatMul.new(cfg.dModel, cfg.vocabSize, vb.pp("lm_head"))
                }

            return QuantizedT5ForConditionalGeneration(
                encoder = encoder,
                decoder = decoder,
                dModel = dModel,
                tieWordEmbeddings = tieWordEmbeddings,
                lmHead = lmHead,
                shared = shared,
                device = vb.device(),
                spanDecode = span(Level.TRACE, "decode"),
                spanDecodeHead = span(Level.TRACE, "decode-head"),
            )
        }
    }

    public fun encode(inputIds: Tensor): Tensor = encoder.forward(inputIds, null)

    public fun decode(
        decoderInputIds: Tensor,
        encoderOutput: Tensor,
    ): Tensor {
        val enter = spanDecode.enter()
        return try {
            val decoderOutput = decoder.forward(decoderInputIds, encoderOutput)
            val scalingFactor =
                if (tieWordEmbeddings) {
                    // Rescale output before projecting on vocab
                    sqrt(dModel.toDouble())
                } else {
                    1.0
                }
            val sequenceOutput =
                decoderOutput
                    .narrow(1, decoderOutput.dim(1) - 1, 1)
                    .squeeze(1)
                    .affine(scalingFactor, 0.0)
            val enterHead = spanDecodeHead.enter()
            try {
                when (lmHead) {
                    null -> sequenceOutput.matmul(shared.embeddings().t())
                    else -> lmHead.forward(sequenceOutput)
                }
            } finally {
                enterHead.exit()
            }
        } finally {
            enter.exit()
        }
    }

    public fun forward(
        inputIds: Tensor,
        decoderInputIds: Tensor,
    ): Tensor {
        val encoderOutput = encode(inputIds)
        return decode(decoderInputIds, encoderOutput)
    }

    public fun device(): Device = device

    public fun clearKvCache() {
        encoder.clearKvCache()
        decoder.clearKvCache()
    }
}
