package com.cjcrafter.tensor4j;

/**
 * 64-bit signed integer. Singleton accessed via {@link DType#I64}.
 */
public final class I64Type extends DType {

    public I64Type() {
        super(Long.BYTES, "int64");
    }
}
