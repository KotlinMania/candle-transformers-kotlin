package io.github.kotlinmania.candletransformers.generation

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candlenn.gumbelSoftmax
import io.github.kotlinmania.candlenn.softmaxLastDim
import io.github.kotlinmania.rand.distr.weighted.WeightedIndex
import io.github.kotlinmania.rand.rngs.StdRng

/**
 * Functional interface for modifying token probability distributions in place before sampling.
 */
public fun interface LogitsModifier {
    public operator fun invoke(prs: FloatArray)
}

/**
 * Logits processor for temperature-based sampling, top-k filtering, and nucleus sampling (top-p).
 */
public class LogitsProcessor private constructor(
    private val rng: StdRng,
    private val sampling: Sampling,
) {
    public companion object {
        public fun fromSampling(
            seed: ULong,
            sampling: Sampling,
        ): LogitsProcessor {
            val rng = StdRng.seedFromU64(seed)
            return LogitsProcessor(rng, sampling)
        }

        public fun new(
            seed: ULong,
            temperature: Double?,
            topP: Double?,
        ): LogitsProcessor {
            val filteredTemp =
                if (temperature != null && temperature < 1e-7) {
                    null
                } else {
                    temperature
                }
            val sampling =
                when (filteredTemp) {
                    null -> {
                        Sampling.ArgMax
                    }

                    else -> {
                        when (topP) {
                            null -> Sampling.All(filteredTemp)
                            else -> Sampling.TopP(topP, filteredTemp)
                        }
                    }
                }
            return fromSampling(seed, sampling)
        }
    }

    private fun sampleArgmax(logits: Tensor): UInt = logits.argmax(logits.rank() - 1).toScalar<UInt>()

    private fun sampleGumbelSoftmax(
        logits: Tensor,
        temperature: Double,
    ): UInt {
        val sampled = gumbelSoftmax(logits, temperature, D.Minus1)
        return sampled.toScalar<UInt>()
    }

    private fun sampleMultinomial(prs: FloatArray): UInt {
        val distr = WeightedIndex.fromFloats(prs)
        val nextToken = distr.sample(rng)
        return nextToken.toUInt()
    }

    /**
     * top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
     * probability topP. This way we never sample tokens that have very low probabilities and are
     * less likely to go "off the rails".
     */
    private fun sampleTopp(
        prs: FloatArray,
        topP: Float,
    ): UInt {
        val argsortIndices = (0 until prs.size).sortedWith { i, j -> prs[j].compareTo(prs[i]) }

        // Clamp smaller probabilities to zero.
        var cumsum = 0.0f
        for (index in argsortIndices) {
            if (cumsum >= topP) {
                prs[index] = 0.0f
            } else {
                cumsum += prs[index]
            }
        }
        // Sample with clamped probabilities.
        return sampleMultinomial(prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    private fun sampleTopk(
        prs: FloatArray,
        topK: Int,
    ): UInt =
        if (topK >= prs.size) {
            sampleMultinomial(prs)
        } else {
            val sortedIndices = (0 until prs.size).sortedWith { i, j -> prs[j].compareTo(prs[i]) }
            val indices = sortedIndices.subList(0, topK)
            val topPrs = FloatArray(topK) { prs[indices[it]] }
            val index = sampleMultinomial(topPrs)
            indices[index.toInt()].toUInt()
        }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    private fun sampleTopkTopp(
        prs: FloatArray,
        topK: Int,
        topP: Float,
    ): UInt =
        if (topK >= prs.size) {
            sampleTopp(prs, topP)
        } else {
            val sortedIndices = (0 until prs.size).sortedWith { i, j -> prs[j].compareTo(prs[i]) }
            val indices = sortedIndices.subList(0, topK)
            val topPrs = FloatArray(topK) { prs[indices[it]] }
            var sumP = 0.0f
            for (v in topPrs) {
                sumP += v
            }
            val index =
                if (topP <= 0.0f || topP >= sumP) {
                    sampleMultinomial(topPrs)
                } else {
                    sampleTopp(topPrs, topP)
                }
            indices[index.toInt()].toUInt()
        }

    public fun sample(logits: Tensor): UInt = sampleF(logits) {}

    public fun sampleF(
        logits: Tensor,
        f: LogitsModifier,
    ): UInt {
        val logitsF32 = logits.toDtype(DType.F32)
        val computePrs = { temperature: Double ->
            val scaledLogits = logitsF32 / temperature
            val prsTensor = softmaxLastDim(scaledLogits)
            val prs = prsTensor.toVec1F32()
            f.invoke(prs)
            prs
        }

        return when (val s = sampling) {
            is Sampling.ArgMax -> {
                sampleArgmax(logitsF32)
            }

            is Sampling.GumbelSoftmax -> {
                sampleGumbelSoftmax(logitsF32, s.temperature)
            }

            is Sampling.All -> {
                val prs = computePrs(s.temperature)
                sampleMultinomial(prs)
            }

            is Sampling.TopP -> {
                val prs = computePrs(s.temperature)
                if (s.p <= 0.0 || s.p >= 1.0) {
                    // simply sample from the predicted probability distribution
                    sampleMultinomial(prs)
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    sampleTopp(prs, s.p.toFloat())
                }
            }

            is Sampling.TopK -> {
                val prs = computePrs(s.temperature)
                sampleTopk(prs, s.k)
            }

            is Sampling.TopKThenTopP -> {
                val prs = computePrs(s.temperature)
                sampleTopkTopp(prs, s.k, s.p.toFloat())
            }
        }
    }
}
