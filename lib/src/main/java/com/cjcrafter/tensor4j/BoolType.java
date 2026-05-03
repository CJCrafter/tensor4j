package com.cjcrafter.tensor4j;

/**
 * Single-byte boolean. Singleton accessed via {@link DType#BOOL}.
 */
public final class BoolType extends DType {

    public BoolType() {
        super(1, "bool");
    }
}
