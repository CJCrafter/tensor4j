package com.cjcrafter.tensor4j.backend;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.memory.Storage;
import jdk.incubator.vector.VectorOperators;

/**
 * F32 op surface for a single (Device, F32) kernel table. Implementations
 * cast incoming {@link Storage}.
 */
public interface F32KernelTable {

    /**
     * Returns the device this kernel table computes on.
     */
    Device device();

    /**
     * Returns {@link DType#F32}; this table only ever operates on 32-bit floats.
     */
    default DType dtype() {
        return DType.F32;
    }

    // ---- matmul (single 2D block) ----

    /**
     * Computes a single dense {@code out[m x n] += a[m x k] @ b[k x n]} block (no batching).
     */
    void matmul(Storage a, int aOff, int m, int k,
                Storage b, int bOff, int n,
                Storage out, int outOff);

    // ---- reductions ----

    /**
     * Returns the sum of {@code len} contiguous elements of {@code a} starting at {@code aOff}.
     */
    float sum(Storage a, int aOff, int len);

    /**
     * Returns the mean of {@code len} contiguous elements of {@code a} starting at {@code aOff}.
     */
    float mean(Storage a, int aOff, int len);

    /**
     * Sums {@code a} (shaped {@code dims}) along {@code dim}, writing the result into {@code out}.
     */
    void sumDim(Storage a, int aOff, int[] dims, int dim, Storage out);

    /**
     * Reduces {@code a} (shaped {@code dims}) to its maximum along {@code dim}, writing the result into {@code out}.
     */
    void maxDim(Storage a, int aOff, int[] dims, int dim, Storage out);

    /**
     * Reduces {@code a} (shaped {@code dims}) to the index of its maximum along {@code dim}, writing the result into {@code out}.
     */
    void argmaxDim(Storage a, int aOff, int[] dims, int dim, Storage out);

    // ---- elementwise ----

    /**
     * Applies the unary op elementwise to {@code len} elements of {@code a}, writing into {@code out}.
     */
    void unary(VectorOperators.Unary op,
               Storage a, int aOff,
               Storage out, int outOff, int len);

    /**
     * Applies the binary op elementwise between {@code a} and {@code b} (same shape/contiguous),
     * writing {@code len} results into {@code out}.
     */
    void binary(VectorOperators.Binary op,
                Storage a, int aOff,
                Storage b, int bOff,
                Storage out, int outOff, int len);

    /**
     * Applies the binary op elementwise between {@code a} and the constant {@code b}, writing
     * {@code len} results into {@code out}.
     */
    void binaryConst(VectorOperators.Binary op,
                     Storage a, int aOff, float b,
                     Storage out, int outOff, int len);

    /**
     * Applies the binary op elementwise between the constant {@code a} and {@code b} (reversed
     * operand order), writing {@code len} results into {@code out}.
     */
    void rbinaryConst(VectorOperators.Binary op,
                      float a, Storage b, int bOff,
                      Storage out, int outOff, int len);

    /**
     * Applies the binary op elementwise between {@code a} and {@code b} using explicit
     * per-axis strides (stride 0 marks a broadcast axis), writing into {@code out} shaped {@code outShape}.
     */
    void binaryBroadcast(VectorOperators.Binary op,
                         Storage a, int[] aStrides, int aOff,
                         Storage b, int[] bStrides, int bOff,
                         Storage out, Shape outShape);

    /**
     * Applies the comparison elementwise between {@code a} and {@code b} (same shape/contiguous),
     * writing {@code len} 1.0/0.0 results into {@code out}.
     */
    void compare(VectorOperators.Comparison op,
                 Storage a, int aOff,
                 Storage b, int bOff,
                 Storage out, int outOff, int len);

    /**
     * Applies the comparison elementwise between {@code a} and the constant {@code b}, writing
     * {@code len} 1.0/0.0 results into {@code out}.
     */
    void compareConst(VectorOperators.Comparison op,
                      Storage a, int aOff, float b,
                      Storage out, int outOff, int len);

    /**
     * Applies the comparison elementwise between {@code a} and {@code b} using explicit
     * per-axis strides (stride 0 marks a broadcast axis), writing 1.0/0.0 results into
     * {@code out} shaped {@code outShape}.
     */
    void compareBroadcast(VectorOperators.Comparison op,
                          Storage a, int[] aStrides, int aOff,
                          Storage b, int[] bStrides, int bOff,
                          Storage out, Shape outShape);

    /**
     * Applies the ternary op elementwise across {@code a}, {@code b}, and {@code c} (same
     * shape/contiguous), writing {@code len} results into {@code out}.
     */
    void ternary(VectorOperators.Ternary op,
                 Storage a, int aOff,
                 Storage b, int bOff,
                 Storage c, int cOff,
                 Storage out, int outOff, int len);

    // ---- optimizer fused kernel ----

    /**
     * Applies one fused AdamW update step in place to {@code len} elements of {@code param},
     * updating the {@code m}/{@code v} moment estimates from {@code grad}.
     */
    void adamw(Storage param, int paramOff,
               Storage grad, int gradOff,
               Storage m, int mOff,
               Storage v, int vOff,
               int len,
               float lr, float beta1, float beta2,
               float biasCorr1, float biasCorr2,
               float weightDecay, float eps);
}
