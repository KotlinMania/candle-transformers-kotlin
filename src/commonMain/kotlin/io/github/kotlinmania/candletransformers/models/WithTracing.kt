// port-lint: source models/with_tracing.rs
package io.github.kotlinmania.candletransformers.models

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.QMatMul
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candlenn.Conv2d
import io.github.kotlinmania.candlenn.Conv2dConfig
import io.github.kotlinmania.candlenn.Embedding
import io.github.kotlinmania.candlenn.LayerNorm
import io.github.kotlinmania.candlenn.LayerNormConfig
import io.github.kotlinmania.candlenn.Linear
import io.github.kotlinmania.candlenn.RmsNorm
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.conv2d
import io.github.kotlinmania.candlenn.embedding
import io.github.kotlinmania.candlenn.layerNorm
import io.github.kotlinmania.candlenn.linear
import io.github.kotlinmania.candlenn.linearB
import io.github.kotlinmania.candlenn.linearNoBias
import io.github.kotlinmania.candlenn.rmsNorm
import io.github.kotlinmania.candletransformers.QuantizedVarBuilder
import io.github.kotlinmania.tracing.Level
import io.github.kotlinmania.tracing.Span
import io.github.kotlinmania.tracing.span

public class TracingEmbedding(
    private val inner: Embedding,
    private val span: Span,
) : Module {
    public companion object {
        public fun new(
            d1: Int,
            d2: Int,
            vb: VarBuilder,
        ): TracingEmbedding {
            val inner = embedding(d1, d2, vb)
            val span = span(Level.TRACE, "embedding")
            return TracingEmbedding(inner, span)
        }

        public fun fromWeights(weights: Tensor): TracingEmbedding {
            val (_, outSize) = weights.dims2()
            val inner = Embedding.new(weights, outSize)
            val span = span(Level.TRACE, "embedding")
            return TracingEmbedding(inner, span)
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

public class TracingLinear(
    private val inner: Linear,
    private val span: Span,
) : Module {
    public companion object {
        public fun fromWeights(
            weights: Tensor,
            bias: Tensor?,
        ): TracingLinear {
            val inner = Linear.new(weights, bias)
            val span = span(Level.TRACE, "linear")
            return TracingLinear(inner, span)
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            inner.forward(xs)
        } finally {
            enter.exit()
        }
    }
}

public fun tracingLinearB(
    d1: Int,
    d2: Int,
    b: Boolean,
    vb: VarBuilder,
): TracingLinear {
    val inner = linearB(d1, d2, b, vb)
    val span = span(Level.TRACE, "linear")
    return TracingLinear(inner, span)
}

public fun tracingLinear(
    d1: Int,
    d2: Int,
    vb: VarBuilder,
): TracingLinear {
    val inner = linear(d1, d2, vb)
    val span = span(Level.TRACE, "linear")
    return TracingLinear(inner, span)
}

public fun tracingLinearNoBias(
    d1: Int,
    d2: Int,
    vb: VarBuilder,
): TracingLinear {
    val inner = linearNoBias(d1, d2, vb)
    val span = span(Level.TRACE, "linear")
    return TracingLinear(inner, span)
}

// Wrap the conv2d op to provide some tracing.
public class TracingConv2d(
    private val inner: Conv2d,
    private val span: Span,
) : Module {
    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            inner.forward(xs)
        } finally {
            enter.exit()
        }
    }
}

public fun tracingConv2d(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    cfg: Conv2dConfig,
    vs: VarBuilder,
): TracingConv2d {
    val span = span(Level.TRACE, "conv2d")
    val inner = conv2d(inChannels, outChannels, kernelSize, cfg, vs)
    return TracingConv2d(inner, span)
}

// QMatMul wrapper adding some tracing.
public class TracingQMatMul(
    private val inner: QMatMul,
    private val span: Span,
) : Module {
    public companion object {
        public fun new(
            outDim: Int,
            inDim: Int,
            vb: QuantizedVarBuilder,
        ): TracingQMatMul {
            val ws = vb.get(Shape(inDim, outDim), "weight")
            val inner = QMatMul.fromArc(ws).unwrap()
            val span = span(Level.TRACE, "qmatmul")
            return TracingQMatMul(inner, span)
        }

        public fun fromWeights(ws: QTensor): TracingQMatMul {
            val inner = QMatMul.fromArc(ws).unwrap()
            val span = span(Level.TRACE, "qmatmul")
            return TracingQMatMul(inner, span)
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            inner.forward(xs)
        } finally {
            enter.exit()
        }
    }

    override fun toString(): String = "QMatMul"
}

public class TracingLayerNorm(
    private val inner: LayerNorm,
    private val span: Span,
) : Module {
    public companion object {
        public fun new(
            weight: Tensor,
            bias: Tensor,
            eps: Double,
        ): TracingLayerNorm {
            val inner = LayerNorm.new(weight, bias, eps)
            val span = span(Level.TRACE, "layer-norm")
            return TracingLayerNorm(inner, span)
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            inner.forward(xs)
        } finally {
            enter.exit()
        }
    }
}

public fun tracingLayerNorm(
    size: Int,
    c: LayerNormConfig,
    vb: VarBuilder,
): TracingLayerNorm {
    val inner = layerNorm(size, c, vb)
    val span = span(Level.TRACE, "layer-norm")
    return TracingLayerNorm(inner, span)
}

public fun tracingLayerNorm(
    size: Int,
    eps: Double,
    vb: VarBuilder,
): TracingLayerNorm = tracingLayerNorm(size, LayerNormConfig.fromEps(eps), vb)

public class TracingRmsNorm(
    private val inner: RmsNorm,
    private val span: Span,
) : Module {
    public companion object {
        public fun new(
            size: Int,
            eps: Double,
            vb: VarBuilder,
        ): TracingRmsNorm {
            val span = span(Level.TRACE, "rms-norm")
            val inner = rmsNorm(size, eps, vb)
            return TracingRmsNorm(inner, span)
        }
    }

    public fun forwardDiff(x: Tensor): Tensor {
        val enter = span.enter()
        return try {
            inner.forwardDiff(x)
        } finally {
            enter.exit()
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val enter = span.enter()
        return try {
            inner.forward(xs)
        } finally {
            enter.exit()
        }
    }
}
