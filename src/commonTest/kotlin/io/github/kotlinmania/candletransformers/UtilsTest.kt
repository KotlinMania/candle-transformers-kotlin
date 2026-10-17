// port-lint: tests utils.rs
package io.github.kotlinmania.candletransformers

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import kotlin.test.Test
import kotlin.test.assertEquals

class UtilsTest {
    // Regression test for https://github.com/huggingface/candle/issues/3582
    @Test
    fun additiveCausalMaskIsBatchIndependentAndBroadcasts() {
        val device = Device.Cpu
        val neg = Float.NEGATIVE_INFINITY

        val mask = buildAdditiveCausalMask(3, 0, null, device, DType.F32)
        assertEquals(
            listOf(1, 1, 3, 3),
            mask.dims(),
            "mask must carry a leading batch dim of 1 so broadcastAdd applies it to every row",
        )

        // Broadcasting onto a 2-sequence batch must mask both rows identically and causally.
        val scores = Tensor.zeros(Shape(2, 1, 3, 3), DType.F32, device)
        val masked =
            scores
                .broadcastAdd(mask)
                .squeeze(1)
                .toVec3<Float>()

        assertEquals(
            masked[0],
            masked[1],
            "both batch rows must receive the same causal mask",
        )
        assertEquals(listOf(0.0f, neg, neg), masked[0][0])
        assertEquals(listOf(0.0f, 0.0f, neg), masked[0][1])
        assertEquals(listOf(0.0f, 0.0f, 0.0f), masked[0][2])
    }

    @Test
    fun additiveCausalMaskHandlesKvCacheOffset() {
        // With `offset` cached keys the mask is rectangular: every query attends to
        // all cached prefix keys, then the usual causal triangle over the new tokens.
        val device = Device.Cpu
        val neg = Float.NEGATIVE_INFINITY

        val mask =
            buildAdditiveCausalMask(2, 2, null, device, DType.F32)
                .squeeze(0)
                .squeeze(0)
                .toVec2<Float>()

        assertEquals(listOf(0.0f, 0.0f, 0.0f, neg), mask[0])
        assertEquals(listOf(0.0f, 0.0f, 0.0f, 0.0f), mask[1])
    }

    @Test
    fun additiveCausalMaskRespectsSlidingWindow() {
        // With a sliding window of `w`, query i may attend to key j only when it is
        // both causal (j <= i + offset) and within the window (i + offset - j <= w).
        val device = Device.Cpu
        val neg = Float.NEGATIVE_INFINITY

        val mask =
            buildAdditiveCausalMask(4, 0, 1, device, DType.F32)
                .squeeze(0)
                .squeeze(0)
                .toVec2<Float>()

        // window = 1: each query sees itself and the single preceding position.
        assertEquals(listOf(0.0f, neg, neg, neg), mask[0])
        assertEquals(listOf(0.0f, 0.0f, neg, neg), mask[1])
        assertEquals(listOf(neg, 0.0f, 0.0f, neg), mask[2])
        assertEquals(listOf(neg, neg, 0.0f, 0.0f), mask[3])
    }
}
