// port-lint: source fused_moe.rs
package io.github.kotlinmania.candletransformers

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.argSortLastDim
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.sortLastDim
import io.github.kotlinmania.candlenn.Activation
import io.github.kotlinmania.candlenn.DEFAULT_KAIMING_NORMAL
import io.github.kotlinmania.candlenn.Linear
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.linearNoBias
import io.github.kotlinmania.candlenn.moeGemm
import io.github.kotlinmania.candlenn.moeGemmGguf
import io.github.kotlinmania.candlenn.softmaxLastDim

// Adapted from: https://github.com/guoqingbao/vllm.rs/blob/main/src/models/layers/moe.rs

public data class MoeCfg(
    public val hiddenSize: Int,
    public val numExperts: Int,
    public val numExpertsPerTok: Int,
    public val moeIntermediateSize: Int,
    public val normTopkProb: Boolean,
    public val act: Activation,
    public val decoderSparseStep: Int?,
)

public class FusedMoe(
    private val gate: Linear,
    private val gateUpW: Tensor,
    private val downW: Tensor,
    private val wSizeN: Int,
    private val act: Activation,
    private val normTopkProb: Boolean,
    private val numExpertsPerTok: Int,
    private val dtype: DType,
) {
    public companion object {
        public fun new(
            cfg: MoeCfg,
            vb: VarBuilder,
            dtype: DType,
        ): FusedMoe {
            val numExperts = cfg.numExperts
            val gate = linearNoBias(cfg.hiddenSize, numExperts, vb.pp("gate"))
            val expertsVb = vb.pp("experts")
            val gateUpExperts = ArrayList<Tensor>(numExperts)
            val downExperts = ArrayList<Tensor>(numExperts)

            // pack experts
            for (i in 0 until numExperts) {
                val expertVb = expertsVb.pp("$i")
                val initWs = DEFAULT_KAIMING_NORMAL
                val gateExpert =
                    expertVb.pp("gate_proj").getWithHints(
                        Shape(cfg.moeIntermediateSize, cfg.hiddenSize),
                        "weight",
                        initWs,
                    )
                val upExpert =
                    expertVb.pp("up_proj").getWithHints(
                        Shape(cfg.moeIntermediateSize, cfg.hiddenSize),
                        "weight",
                        initWs,
                    )
                val downExpert =
                    expertVb.pp("down_proj").getWithHints(
                        Shape(cfg.hiddenSize, cfg.moeIntermediateSize),
                        "weight",
                        initWs,
                    )
                // pack gateProj and upProj
                val gateUpExpert = Tensor.cat(listOf(gateExpert, upExpert), 0)
                gateUpExperts.add(gateUpExpert)
                downExperts.add(downExpert)
            }

            val gateUpW = Tensor.stack(gateUpExperts, 0)
            val downW = Tensor.stack(downExperts, 0)
            val wSizeN = gateUpW.dim(1) / 2

            return FusedMoe(
                gate = gate,
                gateUpW = gateUpW,
                downW = downW,
                wSizeN = wSizeN,
                act = cfg.act,
                normTopkProb = cfg.normTopkProb,
                numExpertsPerTok = cfg.numExpertsPerTok,
                dtype = dtype,
            )
        }
    }

    public fun forward(
        xs: Tensor,
        isPrefill: Boolean,
    ): Tensor {
        val (batch, seqLen, hiddenDim) = xs.dims3()
        val xs2d = xs.reshape(-1, hiddenDim)
        val (numTokens, _) = xs2d.dims2()

        val routerLogits = gate.forward(xs2d)
        val routingWeights = softmaxLastDim(routerLogits.toDtype(DType.F32))

        val topkIds =
            routingWeights
                .argSortLastDim(false)
                .narrow(D.Minus1, 0, numExpertsPerTok)
                .contiguous()

        var topkWeights = routingWeights.gather(topkIds, D.Minus1)

        if (normTopkProb) {
            val lastIdx = D.Minus1.toIndex(topkWeights.shape(), "sumKeepdim")
            topkWeights = topkWeights.broadcastDiv(topkWeights.sumKeepdim(lastIdx))
        }

        val (expertIds, sortedTokenIds) =
            if (isPrefill) {
                topkIds.flattenAll().sortLastDim(true)
            } else {
                topkIds.flattenAll().sortLastDim(true)
            }

        // out (M, topK, N)
        val gateUp =
            moeGemm(
                xs2d,
                gateUpW,
                null,
                sortedTokenIds,
                expertIds,
                numExpertsPerTok,
                isPrefill,
            )

        val gatePart =
            gateUp
                .narrow(D.Minus1, 0, wSizeN)
                .contiguous()
        val upPart =
            gateUp
                .narrow(D.Minus1, wSizeN, wSizeN)
                .contiguous()

        // (M * topK, N // 2)
        val downInputs = (upPart * gatePart.apply(act)).reshape(-1, wSizeN)

        // view(M, topK, K) -> sum -> (M, K)
        val ys =
            moeGemm(
                downInputs,
                downW,
                topkWeights,
                sortedTokenIds,
                expertIds,
                numExpertsPerTok,
                isPrefill,
            ).reshape(numTokens, -1, hiddenDim)
                .sum(1)

        return ys.reshape(Shape(batch, seqLen, hiddenDim))
    }
}

public class FusedMoeGguf(
    public val gate: Linear,
    public val gateExperts: QTensor,
    public val upExperts: QTensor,
    public val downExperts: QTensor,
    public val act: Activation,
    public val normTopkProb: Boolean,
    public val numExpertsPerTok: Int,
    public val dtype: DType,
) {
    public companion object {
        public fun new(
            cfg: MoeCfg,
            vb: QuantizedVarBuilder,
            dtype: DType,
        ): FusedMoeGguf {
            val numExperts = cfg.numExperts
            val gateWs =
                vb
                    .pp("ffn_gate_inp")
                    .get(Shape(numExperts, cfg.hiddenSize), "weight")
                    .dequantize(vb.device())
                    .unwrap()
                    .toDtype(DType.F32)

            val gate = Linear.new(gateWs, null)

            val gateExperts =
                vb.pp("ffn_gate_exps").get(
                    Shape(numExperts, cfg.moeIntermediateSize, cfg.hiddenSize),
                    "weight",
                )
            val upExperts =
                vb.pp("ffn_up_exps").get(
                    Shape(numExperts, cfg.moeIntermediateSize, cfg.hiddenSize),
                    "weight",
                )
            val downExperts =
                vb.pp("ffn_down_exps").get(
                    Shape(numExperts, cfg.hiddenSize, cfg.moeIntermediateSize),
                    "weight",
                )

            return FusedMoeGguf(
                gate = gate,
                gateExperts = gateExperts,
                upExperts = upExperts,
                downExperts = downExperts,
                act = cfg.act,
                normTopkProb = cfg.normTopkProb,
                numExpertsPerTok = cfg.numExpertsPerTok,
                dtype = dtype,
            )
        }
    }

    public fun forward(
        xs: Tensor,
        isPrefill: Boolean,
    ): Tensor {
        val (batch, seqLen, hiddenDim) = xs.dims3()
        val xsReshaped = xs.reshape(-1, hiddenDim)
        val (numTokens, _) = xsReshaped.dims2()
        val originalDtype = xsReshaped.dtype()
        val xsF32 =
            if (xsReshaped.dtype() != DType.F32) {
                xsReshaped.toDtype(DType.F32)
            } else {
                xsReshaped
            }

        val routerLogits = gate.forward(xsF32)
        val routingWeights = softmaxLastDim(routerLogits.toDtype(DType.F32))

        val topkIds =
            routingWeights
                .argSortLastDim(false)
                .narrow(D.Minus1, 0, numExpertsPerTok)
                .contiguous()

        var topkWeights = routingWeights.gather(topkIds, D.Minus1)

        if (normTopkProb) {
            val lastIdx = D.Minus1.toIndex(topkWeights.shape(), "sumKeepdim")
            topkWeights = topkWeights.broadcastDiv(topkWeights.sumKeepdim(lastIdx))
        }

        val (expertIds, sortedTokenIds) =
            if (isPrefill) {
                topkIds.flattenAll().sortLastDim(true)
            } else {
                topkIds.flattenAll().sortLastDim(true)
            }

        val gateOut =
            moeGemmGguf(
                xsF32,
                gateExperts,
                null,
                sortedTokenIds,
                expertIds,
                numExpertsPerTok,
                isPrefill,
                dtype,
            )
        val upOut =
            moeGemmGguf(
                xsF32,
                upExperts,
                null,
                sortedTokenIds,
                expertIds,
                numExpertsPerTok,
                isPrefill,
                dtype,
            )

        val downInputs = upOut * gateOut.apply(act)
        val moeOut =
            moeGemmGguf(
                downInputs,
                downExperts,
                topkWeights,
                sortedTokenIds,
                expertIds,
                numExpertsPerTok,
                isPrefill,
                dtype,
            )

        var ys = moeOut.reshape(numTokens, -1, hiddenDim).sum(1)
        if (ys.dtype() != originalDtype) {
            ys = ys.toDtype(originalDtype)
        }
        return ys.reshape(Shape(batch, seqLen, hiddenDim))
    }
}
