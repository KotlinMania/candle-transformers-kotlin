pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0" }

dependencyResolutionManagement {
    repositories {
        mavenLocal()
        google()
        mavenCentral()
    }
}

rootProject.name = "candle-transformers-kotlin"

listOf(
    "../candle-nn-kotlin",
    "../candle-kotlin",
    "../accelerate-src-kotlin",
    "../byteorder-kotlin",
    "../candle-kernels-kotlin",
    "../candle-metal-kernels-kotlin",
    "../objc2-metal-kotlin",
    "../objc2-foundation-kotlin",
    "../cudarc-kotlin",
    "../gemm-kotlin",
    "../half-kotlin",
    "../float8-kotlin",
    "../intel-mkl-src-kotlin",
    "../libm-kotlin",
    "../memmap2-kotlin",
    "../num-traits-kotlin",
    "../num-cpus-kotlin",
    "../rand-distr-kotlin",
    "../rayon-kotlin",
    "../safetensors-kotlin",
    "../yoke-kotlin",
    "../zip-kotlin",
    "../tokenizers-kotlin",
    "../envie-kotlin",
).forEach { path ->
    if (file(path).exists()) {
        includeBuild(path)
    }
}
