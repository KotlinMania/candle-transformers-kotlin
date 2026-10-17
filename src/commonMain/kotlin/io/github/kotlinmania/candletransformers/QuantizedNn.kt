// port-lint: source quantized_nn.rs
package io.github.kotlinmania.candletransformers

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candlenn.Embedding
import io.github.kotlinmania.candlenn.LayerNorm
import io.github.kotlinmania.candlenn.rmsNorm
import io.github.kotlinmania.candletransformers.models.TracingQMatMul
import io.github.kotlinmania.tracing.Level
import io.github.kotlinmania.tracing.Span
import io.github.kotlinmania.tracing.span

/**
 * Utilities for quantized network layers
 *
 * This module contains various implementations of standard neural network layers, modules and
 * utilities including embedding, linear layers, and various normalization techniques.
 * Most implementations provide quantized weights support.
 */

public class QuantizedEmbedding(
    private val inner: Embedding,
    private val span: Span,
) : Module {
    public companion object {
        public fun new(
            d1: Int,
            d2: Int,
            vb: QuantizedVarBuilder,
        ): QuantizedEmbedding {
            val embeddings = vb.get(Shape(d1, d2), "weight").dequantize(vb.device()).unwrap()
            val inner = Embedding.new(embeddings, d2)
            val span = span(Level.TRACE, "embedding")
            return QuantizedEmbedding(inner, span)
        }
    }

    public fun embeddings(): Tensor = inner.embeddings()

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            inner.forward(xs)
        } finally {
            enter.exit()
        }
    }
}

public class QuantizedLinear(
    private val weight: TracingQMatMul,
    private val bias: Tensor?,
) : Module {
    public companion object {
        public fun fromArc(
            weight: QTensor,
            bias: Tensor?,
        ): QuantizedLinear {
            val qm = TracingQMatMul.fromWeights(weight)
            return QuantizedLinear(qm, bias)
        }

        public fun fromWeights(
            weight: TracingQMatMul,
            bias: Tensor?,
        ): QuantizedLinear = QuantizedLinear(weight, bias)
    }

    override fun forward(xs: Tensor): Tensor {
        val x = xs.apply(weight)
        return when (bias) {
            null -> x
            else -> x.broadcastAdd(bias)
        }
    }
}

public fun quantizedLinearB(
    inDim: Int,
    outDim: Int,
    bias: Boolean,
    vb: QuantizedVarBuilder,
): QuantizedLinear {
    val biasTensor =
        if (bias) {
            vb.get(outDim, "bias").dequantize(vb.device()).unwrap()
        } else {
            null
        }
    val weight = TracingQMatMul.new(inDim, outDim, vb)
    return QuantizedLinear(weight, biasTensor)
}

public fun quantizedLinear(
    inDim: Int,
    outDim: Int,
    vb: QuantizedVarBuilder,
): QuantizedLinear {
    val bias = vb.get(outDim, "bias").dequantize(vb.device()).unwrap()
    val weight = TracingQMatMul.new(inDim, outDim, vb)
    return QuantizedLinear(weight, bias)
}

public fun quantizedLayerNorm(
    size: Int,
    eps: Double,
    vb: QuantizedVarBuilder,
): LayerNorm {
    val weight = vb.get(size, "weight").dequantize(vb.device()).unwrap()
    val bias = vb.get(size, "bias").dequantize(vb.device()).unwrap()
    return LayerNorm.new(weight, bias, eps)
}

public fun quantizedLayerNormNoBias(
    size: Int,
    eps: Double,
    vb: QuantizedVarBuilder,
): LayerNorm {
    val weight = vb.get(size, "weight").dequantize(vb.device()).unwrap()
    return LayerNorm.newNoBias(weight, eps)
}

public fun quantizedLinearNoBias(
    inDim: Int,
    outDim: Int,
    vb: QuantizedVarBuilder,
): QuantizedLinear {
    val weight = TracingQMatMul.new(inDim, outDim, vb)
    return QuantizedLinear(weight, null)
}

public class QuantizedRmsNorm(
    private val weight: Tensor,
    private val eps: Double,
    private val span: Span,
) : Module {
    public companion object {
        public fun new(
            size: Int,
            eps: Double,
            vb: QuantizedVarBuilder,
        ): QuantizedRmsNorm {
            val span = span(Level.TRACE, "rms-norm")
            val weight = vb.get(size, "weight").dequantize(vb.device()).unwrap()
            return QuantizedRmsNorm(weight, eps, span)
        }

        public fun fromQtensor(
            weight: QTensor,
            eps: Double,
        ): QuantizedRmsNorm {
            val span = span(Level.TRACE, "rms-norm")
            val dequantized = weight.dequantize(weight.device()).unwrap()
            return QuantizedRmsNorm(dequantized, eps, span)
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            rmsNorm(xs, weight, eps.toFloat())
        } finally {
            enter.exit()
        }
    }
}
