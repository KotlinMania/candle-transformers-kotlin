// port-lint: source quantized_var_builder.rs
package io.github.kotlinmania.candletransformers

import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.NpyByteReader
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.SafeTensors
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.gguf.Content

/**
 * VarBuilder for loading GGUF files
 *
 * VarBuilder is a utility to store quantized tensors from a [GGUF model file](https://huggingface.co/docs/hub/gguf).
 * These tensors can be loaded from disk using `fromGguf` or from an in-memory
 * buffer using `fromGgufBuffer`.
 */

// VarBuilder specialized for QTensors
public class QuantizedVarBuilder internal constructor(
    private val data: Map<String, QTensor>,
    private val pathParts: List<String>,
    private val device: Device,
) {
    public companion object {
        public fun fromGguf(
            p: String,
            device: Device,
        ): QuantizedVarBuilder {
            val bytes = SafeTensors.fsRead(p).unwrap()
            return fromGgufBuffer(bytes, device)
        }

        public fun fromGgufBuffer(
            buffer: ByteArray,
            device: Device,
        ): QuantizedVarBuilder {
            val cursor = NpyByteReader(buffer)
            val content = Content.read(cursor).unwrap()
            val data = HashMap<String, QTensor>()
            for (tensorName in content.tensorInfos.keys) {
                val tensor = content.tensor(cursor, tensorName, device).unwrap()
                data[tensorName] = tensor
            }
            return QuantizedVarBuilder(
                data = data,
                pathParts = emptyList(),
                device = device,
            )
        }

        public fun fromTensors(
            data: Map<String, QTensor>,
            device: Device,
        ): QuantizedVarBuilder =
            QuantizedVarBuilder(
                data = data,
                pathParts = emptyList(),
                device = device,
            )
    }

    public fun pp(s: String): QuantizedVarBuilder {
        val nextPath = ArrayList<String>(pathParts.size + 1)
        nextPath.addAll(pathParts)
        nextPath.add(s)
        return QuantizedVarBuilder(
            data = data,
            pathParts = nextPath,
            device = device,
        )
    }

    private fun path(tensorName: String): String =
        if (pathParts.isEmpty()) {
            tensorName
        } else {
            "${pathParts.joinToString(".")}.$tensorName"
        }

    public fun get(
        s: Shape,
        name: String,
    ): QTensor {
        val fullPath = path(name)
        val qtensor = data[fullPath] ?: bail("cannot find tensor $fullPath")
        if (qtensor.shape() != s) {
            bail("shape mismatch for $name, got ${qtensor.shape()}, expected $s")
        }
        return qtensor
    }

    public fun get(
        dim: Int,
        name: String,
    ): QTensor = get(Shape(dim), name)

    public fun get(
        d1: Int,
        d2: Int,
        name: String,
    ): QTensor = get(Shape(d1, d2), name)

    public fun getNoShape(name: String): QTensor {
        val fullPath = path(name)
        return data[fullPath] ?: bail("cannot find tensor $name")
    }

    public fun device(): Device = device

    public fun containsKey(key: String): Boolean = data.containsKey(key)
}
