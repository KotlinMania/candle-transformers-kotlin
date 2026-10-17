// port-lint: source object_detection.rs
package io.github.kotlinmania.candletransformers

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Bounding Boxes and Intersection
 *
 * This module provides functionality for handling bounding boxes and their manipulation,
 * particularly in the context of object detection. It includes tools for calculating
 * intersection over union (IoU) and non-maximum suppression (NMS).
 */

/**
 * A bounding box around an object.
 */
public data class Bbox(
    public val xmin: Float,
    public val ymin: Float,
    public val xmax: Float,
    public val ymax: Float,
    public var confidence: Float,
    public val data: List<KeyPoint> = emptyList(),
)

public data class KeyPoint(
    public val x: Float,
    public val y: Float,
    public val mask: Float,
)

/**
 * Intersection over union of two bounding boxes.
 */
public fun iou(
    b1: Bbox,
    b2: Bbox,
): Float {
    val b1Area = (b1.xmax - b1.xmin + 1.0f) * (b1.ymax - b1.ymin + 1.0f)
    val b2Area = (b2.xmax - b2.xmin + 1.0f) * (b2.ymax - b2.ymin + 1.0f)
    val iXmin = max(b1.xmin, b2.xmin)
    val iXmax = min(b1.xmax, b2.xmax)
    val iYmin = max(b1.ymin, b2.ymin)
    val iYmax = min(b1.ymax, b2.ymax)
    val iArea = max(iXmax - iXmin + 1.0f, 0.0f) * max(iYmax - iYmin + 1.0f, 0.0f)
    return iArea / (b1Area + b2Area - iArea)
}

public fun nonMaximumSuppression(
    bboxes: List<List<Bbox>>,
    threshold: Float,
): List<List<Bbox>> {
    val result = ArrayList<List<Bbox>>(bboxes.size)
    // Perform non-maximum suppression.
    for (classBoxes in bboxes) {
        val bboxesForClass = classBoxes.sortedByDescending { it.confidence }.toMutableList()
        var currentIndex = 0
        for (index in 0 until bboxesForClass.size) {
            var drop = false
            for (prevIndex in 0 until currentIndex) {
                val iouVal = iou(bboxesForClass[prevIndex], bboxesForClass[index])
                if (iouVal > threshold) {
                    drop = true
                    break
                }
            }
            if (!drop) {
                val tmp = bboxesForClass[currentIndex]
                bboxesForClass[currentIndex] = bboxesForClass[index]
                bboxesForClass[index] = tmp
                currentIndex += 1
            }
        }
        result.add(bboxesForClass.subList(0, currentIndex).toList())
    }
    return result
}

// Updates confidences starting at highest and comparing subsequent boxes.
private fun updateConfidences(
    bboxesForClass: List<Bbox>,
    updatedConfidences: FloatArray,
    iouThreshold: Float,
    sigma: Float,
) {
    val len = bboxesForClass.size
    for (currentIndex in 0 until len) {
        val currentBbox = bboxesForClass[currentIndex]
        for (index in (currentIndex + 1) until len) {
            val iouVal = iou(currentBbox, bboxesForClass[index])
            if (iouVal > iouThreshold) {
                // Decay calculation from page 4 of: https://arxiv.org/pdf/1704.04503
                val decay = exp(-iouVal * iouVal / sigma)
                val updatedConfidence = bboxesForClass[index].confidence * decay
                updatedConfidences[index] = updatedConfidence
            }
        }
    }
}

// Sorts the bounding boxes by confidence and applies soft non-maximum suppression.
// This function is based on the algorithm described in https://arxiv.org/pdf/1704.04503
public fun softNonMaximumSuppression(
    bboxes: List<List<Bbox>>,
    iouThreshold: Float? = null,
    confidenceThreshold: Float? = null,
    sigma: Float? = null,
): List<List<Bbox>> {
    val resolvedIouThreshold = iouThreshold ?: 0.5f
    val resolvedConfidenceThreshold = confidenceThreshold ?: 0.1f
    val resolvedSigma = sigma ?: 0.5f

    val result = ArrayList<List<Bbox>>(bboxes.size)
    for (classBoxes in bboxes) {
        // Sort boxes by confidence in descending order
        val bboxesForClass = classBoxes.sortedByDescending { it.confidence }.map { it.copy() }
        val updatedConfidences = FloatArray(bboxesForClass.size) { bboxesForClass[it].confidence }
        updateConfidences(
            bboxesForClass,
            updatedConfidences,
            resolvedIouThreshold,
            resolvedSigma,
        )
        // Update confidences, set to 0.0 if below threshold
        for (i in updatedConfidences.indices) {
            val confidence = updatedConfidences[i]
            bboxesForClass[i].confidence =
                if (confidence < resolvedConfidenceThreshold) {
                    0.0f
                } else {
                    confidence
                }
        }
        result.add(bboxesForClass)
    }
    return result
}
