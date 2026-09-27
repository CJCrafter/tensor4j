package com.cjcrafter.tensor4j.autograd;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;
import com.google.common.base.Preconditions;

import java.util.Arrays;

/**
 * Backward for {@code C = A @ B}, including N-D batched broadcasting (the batch dims of
 * {@code A} and {@code B} may differ and are broadcast against each other, PyTorch-style).
 *
 * <pre>{@code
 * dL/dA = gradOutput @ B^T   (summed back down over any broadcast batch dims)
 * dL/dB = A^T @ gradOutput   (summed back down over any broadcast batch dims)
 * }</pre>
 */
public class MatmulFn extends TensorFunction {

    /**
     * Matrix-multiplies {@code a} and {@code b} (see {@link com.cjcrafter.tensor4j.ops.Ops#matmul}
     * for the batching/broadcasting rules) and records the backward function.
     */
    public static Tensor apply(Tensor a, Tensor b) {
        int aDim = a.getShape().dimensions();
        int bDim = b.getShape().dimensions();

        int k1 = a.getShape().dim(aDim - 1);
        int k2 = b.getShape().dim(bDim - 2);
        Preconditions.checkArgument(k1 == k2,
                "matmul inner dimensions don't match: %s @ %s", a.getShape(), b.getShape());

        int m = a.getShape().dim(aDim - 2);
        int n = b.getShape().dim(bDim - 1);

        int[] aBatch = Arrays.copyOf(a.getShape().dims(), aDim - 2);
        int[] bBatch = Arrays.copyOf(b.getShape().dims(), bDim - 2);
        int[] outBatch = broadcastBatchDims(aBatch, bBatch, a.getShape(), b.getShape());

        int[] outDims = new int[outBatch.length + 2];
        System.arraycopy(outBatch, 0, outDims, 0, outBatch.length);
        outDims[outBatch.length] = m;
        outDims[outBatch.length + 1] = n;
        Shape resultShape = new Shape(outDims);

        Storage out = Ops.matmul(a, b, resultShape);
        return Tensor.wrapResult(out, resultShape, new MatmulFn(a, b));
    }

    private static int[] broadcastBatchDims(int[] aBatch, int[] bBatch, Shape aShape, Shape bShape) {
        int n = Math.max(aBatch.length, bBatch.length);
        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            int da = i < aBatch.length ? aBatch[aBatch.length - 1 - i] : 1;
            int db = i < bBatch.length ? bBatch[bBatch.length - 1 - i] : 1;
            int rd;
            if (da == db) rd = da;
            else if (da == 1) rd = db;
            else if (db == 1) rd = da;
            else throw new IllegalArgumentException(
                    "matmul batch dims not broadcastable: " + aShape + " @ " + bShape);
            result[n - 1 - i] = rd;
        }
        return result;
    }

    /**
     * Records the backward function for {@code a.matmul(b)}.
     */
    public MatmulFn(Tensor a, Tensor b) {
        super(a, b);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        Tensor gradA = unbroadcast(gradOutput.matmul(b.transpose()), a.getShape());
        Tensor gradB = unbroadcast(a.transpose().matmul(gradOutput), b.getShape());
        return new Tensor[]{gradA, gradB};
    }
}
