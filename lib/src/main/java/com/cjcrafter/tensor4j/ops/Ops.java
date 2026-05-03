package com.cjcrafter.tensor4j.ops;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.memory.NativeStorage;
import com.cjcrafter.tensor4j.memory.Storage;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.VectorOperators;

/**
 * Allocates result storage and dispatches tensor operations to {@link SegmentKernels}.
 * Non-inplace ops return freshly allocated {@link NativeStorage}; inplace ops
 * write into the supplied destination {@link Storage}.
 */
public final class Ops {

    private Ops() {}

    private static Storage alloc(int numel) {
        return NativeStorage.allocate(DType.F32, numel);
    }

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
    public static Storage matmul(Tensor a, Tensor b, Shape resultShape) {
        if (!a.isContiguous()) a = a.contiguous();
        if (!b.isContiguous()) b = b.contiguous();

        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        Storage out = alloc(resultShape.numel());
        // matmul accumulates into c; native blocks are not zero-initialized
        out.segment().fill((byte) 0);

        if (aShape.dimensions() == 2 && bShape.dimensions() == 2) {
            int m = aShape.dim(0), k = aShape.dim(1), n = bShape.dim(1);
            SegmentKernels.matmul(a.getStorage().segment(), a.getOffset(), m, k,
                    b.getStorage().segment(), b.getOffset(), n,
                    out.segment(), 0);
        } else {
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
                SegmentKernels.matmul(a.getStorage().segment(), aOff, m, k,
                        b.getStorage().segment(), bOff, n,
                        out.segment(), batch * cSliceSize);
            }
        }
        return out;
    }

    public static float sum(Tensor a) {
        return SegmentKernels.sum(a.getStorage().segment(), a.getOffset(), a.getShape().numel());
    }

    public static float mean(Tensor a) {
        return SegmentKernels.mean(a.getStorage().segment(), a.getOffset(), a.getShape().numel());
    }

    public static Storage sumDim(Tensor a, int dim) {
        if (!a.isContiguous()) a = a.contiguous();
        int[] dims = a.getShape().dims();
        int outLen = a.getShape().numel() / dims[dim];
        Storage out = alloc(outLen);
        SegmentKernels.sumDim(a.getStorage().segment(), a.getOffset(), dims, dim, out.segment());
        return out;
    }

    public static Storage maxDim(Tensor a, int dim) {
        if (!a.isContiguous()) a = a.contiguous();
        int[] dims = a.getShape().dims();
        int outLen = a.getShape().numel() / dims[dim];
        Storage out = alloc(outLen);
        SegmentKernels.maxDim(a.getStorage().segment(), a.getOffset(), dims, dim, out.segment());
        return out;
    }

    public static Storage argmaxDim(Tensor a, int dim) {
        if (!a.isContiguous()) a = a.contiguous();
        int[] dims = a.getShape().dims();
        int outLen = a.getShape().numel() / dims[dim];
        Storage out = alloc(outLen);
        SegmentKernels.argmaxDim(a.getStorage().segment(), a.getOffset(), dims, dim, out.segment());
        return out;
    }

    public static Storage unaryOp(VectorOperators.Unary op, Tensor a) {
        Storage out = alloc(a.getShape().numel());
        unaryOp(op, a, out);
        return out;
    }

    public static void unaryOp(VectorOperators.Unary op, Tensor a, Storage out) {
        SegmentKernels.unaryContiguous(op,
                a.getStorage().segment(), a.getOffset(),
                out.segment(), 0, a.getShape().numel());
    }

    public static Storage binaryOp(VectorOperators.Binary op, Tensor a, Tensor b, Shape resultShape) {
        Storage out = alloc(resultShape.numel());
        binaryOp(op, a, b, resultShape, out);
        return out;
    }

    public static void binaryOp(VectorOperators.Binary op, Tensor a, Tensor b, Shape resultShape, Storage out) {
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        int n = resultShape.numel();

        if (aShape.equals(bShape) && a.isContiguous() && b.isContiguous()) {
            SegmentKernels.binaryContiguous(op,
                    a.getStorage().segment(), a.getOffset(),
                    b.getStorage().segment(), b.getOffset(),
                    out.segment(), 0, n);
        } else {
            SegmentKernels.binaryBroadcast(op,
                    a.getStorage().segment(), aShape.broadcastStridesTo(resultShape), a.getOffset(),
                    b.getStorage().segment(), bShape.broadcastStridesTo(resultShape), b.getOffset(),
                    out.segment(), resultShape);
        }
    }

    public static Storage binaryOpConst(VectorOperators.Binary op, Tensor a, float b) {
        Storage out = alloc(a.getShape().numel());
        binaryOpConst(op, a, b, out);
        return out;
    }

    public static void binaryOpConst(VectorOperators.Binary op, Tensor a, float b, Storage out) {
        SegmentKernels.binaryConst(op,
                a.getStorage().segment(), a.getOffset(),
                b, out.segment(), a.getShape().numel());
    }

    public static Storage rbinaryOpConst(VectorOperators.Binary op, float a, Tensor b) {
        Storage out = alloc(b.getShape().numel());
        rbinaryOpConst(op, a, b, out);
        return out;
    }

    public static void rbinaryOpConst(VectorOperators.Binary op, float a, Tensor b, Storage out) {
        SegmentKernels.rbinaryConst(op,
                a, b.getStorage().segment(), b.getOffset(),
                out.segment(), b.getShape().numel());
    }

    public static Storage comparisonOp(VectorOperators.Comparison op, Tensor a, Tensor b, Shape resultShape) {
        Storage out = alloc(resultShape.numel());
        comparisonOp(op, a, b, resultShape, out);
        return out;
    }

    public static void comparisonOp(VectorOperators.Comparison op, Tensor a, Tensor b, Shape resultShape, Storage out) {
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        int n = resultShape.numel();

        if (aShape.equals(bShape) && a.isContiguous() && b.isContiguous()) {
            SegmentKernels.compareContiguous(op,
                    a.getStorage().segment(), a.getOffset(),
                    b.getStorage().segment(), b.getOffset(),
                    out.segment(), n);
        } else {
            SegmentKernels.compareBroadcast(op,
                    a.getStorage().segment(), aShape.broadcastStridesTo(resultShape), a.getOffset(),
                    b.getStorage().segment(), bShape.broadcastStridesTo(resultShape), b.getOffset(),
                    out.segment(), resultShape);
        }
    }

    public static Storage comparisonOpConst(VectorOperators.Comparison op, Tensor a, float b) {
        Storage out = alloc(a.getShape().numel());
        comparisonOpConst(op, a, b, out);
        return out;
    }

    public static void comparisonOpConst(VectorOperators.Comparison op, Tensor a, float b, Storage out) {
        SegmentKernels.compareConst(op,
                a.getStorage().segment(), a.getOffset(),
                b, out.segment(), a.getShape().numel());
    }

    public static Storage ternaryOp(VectorOperators.Ternary op, Tensor a, Tensor b, Tensor c, Shape resultShape) {
        Storage out = alloc(resultShape.numel());
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        Shape cShape = c.getShape();
        Preconditions.checkState(aShape.equals(bShape) && bShape.equals(cShape)
                && a.isContiguous() && b.isContiguous() && c.isContiguous(),
                "ternary broadcast not yet supported");

        SegmentKernels.ternaryContiguous(op,
                a.getStorage().segment(), a.getOffset(),
                b.getStorage().segment(), b.getOffset(),
                c.getStorage().segment(), c.getOffset(),
                out.segment(), 0, resultShape.numel());
        return out;
    }
}
