package com.cjcrafter.tensor4j;

/**
 * Element data type. Only 32-bit floats are supported.
 */
public final class DType {

    /** 32-bit IEEE 754 float. */
    public static final DType F32 = new DType();

    private DType() {
    }

    /**
     * Returns the size of one element in bytes.
     *
     * @return the computed value
     */
    public int sizeBytes() {
        return Float.BYTES;
    }

    /**
     * Returns the human-readable name of this type.
     *
     * @return the human-readable name
     */
    public String displayName() {
        return "float32";
    }

    @Override
    public String toString() {
        return displayName();
    }
}
