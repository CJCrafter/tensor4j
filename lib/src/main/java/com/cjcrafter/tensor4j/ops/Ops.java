package com.cjcrafter.tensor4j.ops;

import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import jdk.incubator.vector.VectorOperators;

/**
 * This utility class stores all operations that can be done on a
 * {@link com.cjcrafter.tensor4j.Tensor}.
 *
 * <p>All implementations are delegated to {@link Kernels} as a design choice
 * to allow this class to select the <b>fastest</b> way of handling an
 * operation between tensors. Think of this class as the "JIT" that calls the
 * underlying {@link Kernels} implementations based on the shape of the tensor.
 */
public final class Ops {

    private Ops() {}

    /**
     * Matrix multiply with batch support.
     *
     * <p>Supported cases:
     * <ul>
     *   <li>(m, k) @ (k, n) -> (m, n)</li>
     *   <li>(b, m, k) @ (b, k, n) -> (b, m, n)</li>
     *   <li>(b, m, k) @ (k, n) -> (b, m, n)</li>
     *   <li>(m, k) @ (b, k, n) -> (b, m, n)</li>
     * </ul>
     *
     * <p>Both inputs are forced contiguous before dispatch.
     */
    public static float[] matmul(Tensor a, Tensor b, Shape resultShape) {
        // https://github.com/pytorch/pytorch/blob/f9c0a39ad9320795fc8df77b570c317e0c2ab60e/aten/src/ATen/native/mkldnn/Matmul.cpp
        if (!a.isContiguous()) a = a.contiguous();
        if (!b.isContiguous()) b = b.contiguous();

        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        float[] out = new float[resultShape.numel()];

        if (aShape.dimensions() == 2 && bShape.dimensions() == 2) {
            // (m, k) @ (k, n) -> (m, n)
            int m = aShape.dim(0), k = aShape.dim(1), n = bShape.dim(1);
            Kernels.matmul(a.getData(), a.getOffset(), m, k,
                    b.getData(), b.getOffset(), n,
                    out, 0);

        } else {
            // batched: at least one input is 3D
            // TODO: properly batch this in special cases
            int batches = resultShape.dim(0);
            int m = resultShape.dim(1);
            int n = resultShape.dim(2);

            boolean aBatched = aShape.dimensions() == 3;
            boolean bBatched = bShape.dimensions() == 3;

            int k = aBatched ? aShape.dim(2) : aShape.dim(1);
            int aSliceSize = m * k;
            int bSliceSize = k * n;
            int cSliceSize = m * n;

            for (int batch = 0; batch < batches; batch++) {
                int aOff = a.getOffset() + (aBatched ? batch * aSliceSize : 0);
                int bOff = b.getOffset() + (bBatched ? batch * bSliceSize : 0);
                Kernels.matmul(a.getData(), aOff, m, k,
                        b.getData(), bOff, n,
                        out, batch * cSliceSize);
            }
        }

        return out;
    }

    public static float sum(Tensor a) {
        return Kernels.sum(a.getData(), a.getOffset(), a.getShape().numel());
    }

    public static float mean(Tensor a) {
        return Kernels.mean(a.getData(), a.getOffset(), a.getShape().numel());
    }

    public static float[] sumDim(Tensor a, int dim) {
        if (!a.isContiguous()) a = a.contiguous();

        int[] dims = a.getShape().dims();
        // result size = total / dims[dim]
        int outLen = a.getShape().numel() / dims[dim];
        float[] out = new float[outLen];
        Kernels.sumDim(a.getData(), a.getOffset(), dims, dim, out);
        return out;
    }

    public static float[] maxDim(Tensor a, int dim) {
        if (!a.isContiguous()) a = a.contiguous();

        int[] dims = a.getShape().dims();
        int outLen = a.getShape().numel() / dims[dim];
        float[] out = new float[outLen];
        Kernels.maxDim(a.getData(), a.getOffset(), dims, dim, out);
        return out;
    }

    public static float[] argmaxDim(Tensor a, int dim) {
        if (!a.isContiguous()) a = a.contiguous();

        int[] dims = a.getShape().dims();
        int outLen = a.getShape().numel() / dims[dim];
        float[] out = new float[outLen];
        Kernels.argmaxDim(a.getData(), a.getOffset(), dims, dim, out);
        return out;
    }

    public static void unaryOp(VectorOperators.Unary op, Tensor a, float[] out) {
        Kernels.unaryContiguous(op,
                a.getData(), a.getOffset(),
                out, 0, out.length);
    }

    public static void binaryOp(VectorOperators.Binary op, Tensor a, Tensor b, Shape resultShape, float[] out) {
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();

        if (aShape.equals(bShape) && a.isContiguous() && b.isContiguous()) {
            Kernels.binaryContiguous(op,
                a.getData(), a.getOffset(),
                b.getData(), b.getOffset(),
                out, out.length);
        } else {
            Kernels.binaryBroadcast(op,
                a.getData(), aShape.broadcastStridesTo(resultShape), a.getOffset(),
                b.getData(), bShape.broadcastStridesTo(resultShape), b.getOffset(),
                out, resultShape);
        }
    }

    public static void binaryOpConst(VectorOperators.Binary op, Tensor a, float b, float[] out) {
        Kernels.binaryConst(op,
                a.getData(), a.getOffset(),
                b,
                out, out.length);
    }

    public static void rbinaryOpConst(VectorOperators.Binary op, float a, Tensor b, float[] out) {
        Kernels.rbinaryConst(op,
                a,
                b.getData(), b.getOffset(),
                out, out.length);
    }

    public static void comparisonOp(VectorOperators.Comparison op, Tensor a, Tensor b, Shape resultShape, float[] out) {
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();

        if (aShape.equals(bShape) && a.isContiguous() && b.isContiguous()) {
            Kernels.compareContiguous(op,
                a.getData(), a.getOffset(),
                b.getData(), b.getOffset(),
                out, out.length);
        } else {
            Kernels.compareBroadcast(op,
                a.getData(), aShape.broadcastStridesTo(resultShape), a.getOffset(),
                b.getData(), bShape.broadcastStridesTo(resultShape), b.getOffset(),
                out, resultShape);
        }
    }

    public static void comparisonOpConst(VectorOperators.Comparison op, Tensor a, float b, float[] out) {
        Kernels.compareConst(op,
                a.getData(), a.getOffset(),
                b,
                out, out.length);
    }

    public static void ternaryOp(VectorOperators.Ternary op, Tensor a, Tensor b, Tensor c, Shape resultShape, float[] out) {
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        Shape cShape = c.getShape();

        if (aShape.equals(bShape) && bShape.equals(cShape) && a.isContiguous() && b.isContiguous() && c.isContiguous()) {
            Kernels.ternaryContiguous(op,
                    a.getData(), a.getOffset(),
                    b.getData(), b.getOffset(),
                    c.getData(), c.getOffset(),
                    out, 0, out.length);
        } else {
            throw new IllegalStateException("not yet impl");
        }
    }

}
