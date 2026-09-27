package com.cjcrafter.tensor4j;

import com.google.common.base.Preconditions;
import org.jetbrains.annotations.ApiStatus;

import java.util.Arrays;

/**
 * Defines the shape of a tensor as a tuple of integers, where each integer
 * represents the size of that dimension. For example, {@code (3, 4)} is a
 * shape with 3 rows and 4 columns.
 */
public final class Shape {

    private final int[] dims;
    private final int[] strides;

    /**
     * Creates a contiguous, row-major shape with the given dimensions.
     *
     * @param dims The size of each dimension; must have at least 2 entries.
     */
    public Shape(int... dims) {
        this(dims, contiguousStridesFrom(dims));
    }

    /**
     * Creates a shape with explicit (possibly non-contiguous) strides, e.g. for a view.
     *
     * @param dims The size of each dimension; must have at least 2 entries.
     * @param strides The stride of each dimension; must be the same length as {@code dims}.
     */
    public Shape(int[] dims, int[] strides) {
        if (dims.length < 2) throw new IllegalArgumentException("tensors must be at least 2D, got " + dims.length + "D");
        if (dims.length != strides.length) throw new IllegalArgumentException("dims.length must equal strides.length");

        this.dims = dims;
        this.strides = strides;
    }

    /**
     * Returns the strides for the given shape, assuming the data will be
     * stored contiguously (the default).
     *
     * @param dims The shape.
     * @return The contiguous strides of the given shape.
     */
    public static int[] contiguousStridesFrom(int[] dims) {
        int[] strides = new int[dims.length];

        int expected = 1;
        for (int i = dims.length - 1; i >= 0; i--) {
            strides[i] = expected;
            expected *= dims[i];
        }
        return strides;
    }

    /**
     * Returns the number of dimensions in this shape.
     *
     * <p>Tensors are always at least 2D, so for a matrix this returns
     * <code>2</code>, and for a batch of matrices it returns <code>3</code>, etc.
     *
     * @return The number of dimensions in this tensor.
     */
    public int dimensions() {
        return dims.length;
    }

    /**
     * The total number of elements stored by this shape.
     *
     * @return The total number of elements stored by this shape.
     */
    public int numel() {
        int sum = 1;
        for (int dim : dims) {
            sum *= dim;
        }
        return sum;
    }

    /**
     * Swaps the last two dimensions (matrix transpose). For a 2D shape this
     * is the conventional transpose; for batched ND shapes the batch dims
     * are preserved and only the last two are swapped.
     */
    public Shape transpose() {
        int n = dims.length;
        Preconditions.checkState(n >= 2, "transpose requires at least 2 dimensions, got %s", n);
        int[] newDims = dims.clone();
        int[] newStrides = strides.clone();
        newDims[n - 1] = dims[n - 2];
        newDims[n - 2] = dims[n - 1];
        newStrides[n - 1] = strides[n - 2];
        newStrides[n - 2] = strides[n - 1];
        return new Shape(newDims, newStrides);
    }

    /**
     * Reverses the order of all dimensions. Used by readers of column-major
     * (Fortran-order) data to reinterpret it as row-major. For 2D this is
     * equivalent to {@link #transpose()}.
     */
    @ApiStatus.Internal
    public Shape reverseDims() {
        int n = dims.length;
        int[] newDims = new int[n];
        int[] newStrides = new int[n];
        for (int i = 0; i < n; i++) {
            newDims[i] = dims[n - 1 - i];
            newStrides[i] = strides[n - 1 - i];
        }
        return new Shape(newDims, newStrides);
    }

    /**
     * Size along dimension i.
     */
    public int dim(int i) {
        return dims[i];
    }

    /**
     * Copy of the dims array.
     */
    public int[] dims() {
        return dims.clone();
    }

    /**
     * Copy of the strides array.
     */
    public int[] strides() {
        return strides.clone();
    }

    /**
     * Convert n-dimensional indices to a flat offset into the backing array.
     */
    public int flatIndex(int... indices) {
        int offset = 0;
        for (int i = 0; i < indices.length; i++) {
            offset += indices[i] * strides[i];
        }
        return offset;
    }

    /**
     * True if the data is laid out contiguously in row-major order.
     * This determines whether SIMD bulk operations can be used directly.
     */
    public boolean isContiguous() {
        int expected = 1;
        for (int i = dims.length - 1; i >= 0; i--) {
            if (strides[i] != expected) return false;
            expected *= dims[i];
        }
        return true;
    }

    /**
     * Reshape to new dimensions. One dimension may be -1 to infer its size
     * from the total element count.
     */
    public Shape reshape(int... newDims) {
        if (!isContiguous())
            throw new IllegalStateException("cannot reshape a non-contiguous view; make it contiguous first");

        int inferIdx = -1;
        int known = 1;
        for (int i = 0; i < newDims.length; i++) {
            if (newDims[i] == -1) {
                if (inferIdx != -1)
                    throw new IllegalArgumentException("only one dimension can be -1");
                inferIdx = i;
            } else {
                known *= newDims[i];
            }
        }

        // When we have a -1, then we should infer
        int[] resolved = newDims.clone();
        if (inferIdx != -1)
            resolved[inferIdx] = numel() / known;

        // If the new shape's numel does not match the old one... WELP.
        int total = 1;
        for (int d : resolved)
            total *= d;
        if (total != numel())
            throw new IllegalArgumentException("cannot reshape " + numel() + " elements into " + java.util.Arrays.toString(resolved));

        return new Shape(resolved);
    }

    /**
     * Compute the broadcast-compatible shape between this and other.
     *
     * <p>Examples:
     * <pre>
     *   (3,4) + (1,4)   -> (3,4)   row-vector broadcast
     *   (3,4) + (3,1)   -> (3,4)   col-vector broadcast
     *   (3,4) + (1,1)   -> (3,4)   scalar broadcast
     *   (3,4) + (2,3,4) -> (2,3,4) batch dimension added
     *   (3,4) + (2,4)   -> null    incompatible
     * </pre>
     *
     * @return the broadcast result shape, or null if incompatible.
     */
    public Shape broadcastWith(Shape other) {
        int maxNdim = Math.max(this.dimensions(), other.dimensions());
        int[] result = new int[maxNdim];

        for (int i = 0; i < maxNdim; i++) {
            int d1 = i < this.dimensions() ? this.dims[this.dimensions() - 1 - i] : 1;
            int d2 = i < other.dimensions() ? other.dims[other.dimensions() - 1 - i] : 1;

            if (d1 == d2)
                result[maxNdim - 1 - i] = d1;
            else if (d1 == 1)
                result[maxNdim - 1 - i] = d2;
            else if (d2 == 1)
                result[maxNdim - 1 - i] = d1;
            else
                return null;
        }

        return new Shape(result);
    }

    /**
     * Returns strides for broadcasting this shape into the target shape.
     * Dimensions being broadcast (size 1 -> size n) get stride 0, so the
     * same element is repeated without copying data.
     */
    public int[] broadcastStridesTo(Shape target) {
        int[] result = new int[target.dimensions()];
        int off = target.dimensions() - this.dimensions();

        for (int i = 0; i < target.dimensions(); i++) {
            int srcIdx = i - off;
            if (srcIdx < 0 || this.dims[srcIdx] == 1) result[i] = 0;
            else result[i] = this.strides[srcIdx];
        }

        return result;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Shape s)) return false;
        return Arrays.equals(dims, s.dims);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("(");

        builder.append(dims[0]);
        for (int i = 1; i < dims.length; i++) {
            builder.append(", ").append(dims[i]);
        }
        builder.append(")");
        return builder.toString();
    }
}
