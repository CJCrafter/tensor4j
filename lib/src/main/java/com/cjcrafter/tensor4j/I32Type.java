package com.cjcrafter.tensor4j;

/**
 * 32-bit signed integer. Singleton accessed via {@link DType#I32}.
 */
public final class I32Type extends DType {

    public I32Type() {
        super(Integer.BYTES, "int32");
    }
}
