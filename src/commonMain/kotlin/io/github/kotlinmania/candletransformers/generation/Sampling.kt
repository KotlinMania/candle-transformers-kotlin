package io.github.kotlinmania.candletransformers.generation

/**
 * Sampling strategy for text generation.
 */
public sealed class Sampling {
    public data object ArgMax : Sampling()

    public data class All(
        public val temperature: Double,
    ) : Sampling()

    public data class TopK(
        public val k: Int,
        public val temperature: Double,
    ) : Sampling()

    public data class TopP(
        public val p: Double,
        public val temperature: Double,
    ) : Sampling()

    public data class TopKThenTopP(
        public val k: Int,
        public val p: Double,
        public val temperature: Double,
    ) : Sampling()

    // Note that the rng is not used for the Gumbel-Softmax sampling.
    public data class GumbelSoftmax(
        public val temperature: Double,
    ) : Sampling()
}
