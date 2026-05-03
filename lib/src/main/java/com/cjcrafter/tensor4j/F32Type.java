package com.cjcrafter.tensor4j;

/**
 * 32-bit IEEE 754 float. Singleton accessed via {@link DType#F32}.
 */
public final class F32Type extends DType {

    public F32Type() {
        super(Float.BYTES, "float32");
    }
}
