package com.cjcrafter.tensor4j;

import com.google.common.base.Preconditions;

/**
 * Element data type tag. Sealed hierarchy with one singleton per concrete
 * subtype (e.g. {@link #F32}). Subclasses are markers; storage and kernel
 * representations are owned by the corresponding backend, not by {@code DType}.
 */
public abstract sealed class DType
        permits F32Type, F64Type, I32Type, I64Type, BoolType {

    public static final F32Type F32 = new F32Type();
    public static final F64Type F64 = new F64Type();
    public static final I32Type I32 = new I32Type();
    public static final I64Type I64 = new I64Type();
    public static final BoolType BOOL = new BoolType();

    private final int sizeBytes;
    private final String displayName;

    protected DType(int sizeBytes, String displayName) {
        Preconditions.checkArgument(sizeBytes > 0, "sizeBytes must be positive");
        this.sizeBytes = sizeBytes;
        this.displayName = Preconditions.checkNotNull(displayName, "displayName");
    }

    /** Size of one element in bytes. */
    public final int sizeBytes() {
        return sizeBytes;
    }

    /** Human-readable name (e.g. {@code "float32"}). Matches numpy/PyTorch naming. */
    public final String displayName() {
        return displayName;
    }

    @Override
    public final String toString() {
        return displayName;
    }
}
