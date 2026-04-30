package com.cjcrafter.tensor4j;

/**
 * Element data types supported by tensor4j.
 */
public enum DType {

    F32(4, "float32"),
    F64(8, "float64"),
    //F16(2, "float16"),
    //BF16(2, "bfloat16"),
    I32(4, "int32"),
    I64(8, "int64"),
    BOOL(1, "bool");

    private final int sizeBytes;
    private final String displayName;

    DType(int sizeBytes, String displayName) {
        this.sizeBytes = sizeBytes;
        this.displayName = displayName;
    }

    /**
     * Size of one element in bytes.
     */
    public int sizeBytes() {
        return sizeBytes;
    }

    /**
     * Human-readable name (e.g. {@code "float32"}). Matches numpy/PyTorch naming.
     */
    public String displayName() {
        return displayName;
    }
}
