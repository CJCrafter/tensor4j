package com.cjcrafter.tensor4j;

/**
 * 64-bit IEEE 754 float. Singleton accessed via {@link DType#F64}.
 */
public final class F64Type extends DType {

    public F64Type() {
        super(Double.BYTES, "float64");
    }
}
