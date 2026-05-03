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

    Device device();

    default DType dtype() {
        return DType.F32;
    }

    // ---- matmul (single 2D block) ----

    void matmul(Storage a, int aOff, int m, int k,
                Storage b, int bOff, int n,
                Storage out, int outOff);

    // ---- reductions ----

    float sum(Storage a, int aOff, int len);

    float mean(Storage a, int aOff, int len);

    void sumDim(Storage a, int aOff, int[] dims, int dim, Storage out);

    void maxDim(Storage a, int aOff, int[] dims, int dim, Storage out);

    void argmaxDim(Storage a, int aOff, int[] dims, int dim, Storage out);

    // ---- elementwise ----

    void unary(VectorOperators.Unary op,
               Storage a, int aOff,
               Storage out, int outOff, int len);

    void binary(VectorOperators.Binary op,
                Storage a, int aOff,
                Storage b, int bOff,
                Storage out, int outOff, int len);

    void binaryConst(VectorOperators.Binary op,
                     Storage a, int aOff, float b,
                     Storage out, int outOff, int len);

    void rbinaryConst(VectorOperators.Binary op,
                      float a, Storage b, int bOff,
                      Storage out, int outOff, int len);

    void binaryBroadcast(VectorOperators.Binary op,
                         Storage a, int[] aStrides, int aOff,
                         Storage b, int[] bStrides, int bOff,
                         Storage out, Shape outShape);

    void compare(VectorOperators.Comparison op,
                 Storage a, int aOff,
                 Storage b, int bOff,
                 Storage out, int outOff, int len);

    void compareConst(VectorOperators.Comparison op,
                      Storage a, int aOff, float b,
                      Storage out, int outOff, int len);

    void compareBroadcast(VectorOperators.Comparison op,
                          Storage a, int[] aStrides, int aOff,
                          Storage b, int[] bStrides, int bOff,
                          Storage out, Shape outShape);

    void ternary(VectorOperators.Ternary op,
                 Storage a, int aOff,
                 Storage b, int bOff,
                 Storage c, int cOff,
                 Storage out, int outOff, int len);

    // ---- optimizer fused kernel ----

    void adamw(Storage param, int paramOff,
               Storage grad, int gradOff,
               Storage m, int mOff,
               Storage v, int vOff,
               int len,
               float lr, float beta1, float beta2,
               float biasCorr1, float biasCorr2,
               float weightDecay, float eps);
}
