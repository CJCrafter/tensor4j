package com.cjcrafter.tensor4j.ops;

import com.cjcrafter.tensor4j.DType;
import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.backend.Backend;
import com.cjcrafter.tensor4j.backend.Backends;
import com.cjcrafter.tensor4j.backend.F32KernelTable;
import com.cjcrafter.tensor4j.memory.NativeStorage;
import com.cjcrafter.tensor4j.memory.Storage;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.VectorOperators;

/**
 * Tensor-level dispatcher. Resolves the right {@link Backend} for the inputs,
 * pulls the appropriate kernel table, allocates result {@link Storage}, and
 * handles tensor-shape concerns (broadcast, batched matmul, contiguous).
 */
public final class Ops {

    private Ops() {}

    private static F32KernelTable f32(Tensor t) {
        return Backends.forDevice(t.device()).f32Kernels();
    }

    private static Storage alloc(Tensor t, int numel) {
        Backend backend = Backends.forDevice(t.device());
        Preconditions.checkArgument(backend.supports(t.dtype()),
                "backend %s does not support dtype %s", backend.device(), t.dtype());
        return NativeStorage.allocate(backend.allocator(), DType.F32, numel);
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
        Storage out = alloc(a, resultShape.numel());
        // matmul accumulates into c; native blocks are not zero-initialized
        out.segment().fill((byte) 0);

        F32KernelTable k = f32(a);

        if (aShape.dimensions() == 2 && bShape.dimensions() == 2) {
            int m = aShape.dim(0), kDim = aShape.dim(1), n = bShape.dim(1);
            k.matmul(a.getStorage(), a.getOffset(), m, kDim,
                    b.getStorage(), b.getOffset(), n,
                    out, 0);
        } else {
            int batches = resultShape.dim(0);
            int m = resultShape.dim(1);
            int n = resultShape.dim(2);

            boolean aBatched = aShape.dimensions() == 3;
            boolean bBatched = bShape.dimensions() == 3;

            int kDim = aBatched ? aShape.dim(2) : aShape.dim(1);
            int aSliceSize = m * kDim;
            int bSliceSize = kDim * n;
            int cSliceSize = m * n;

            for (int batch = 0; batch < batches; batch++) {
                int aOff = a.getOffset() + (aBatched ? batch * aSliceSize : 0);
                int bOff = b.getOffset() + (bBatched ? batch * bSliceSize : 0);
                k.matmul(a.getStorage(), aOff, m, kDim,
                        b.getStorage(), bOff, n,
                        out, batch * cSliceSize);
            }
        }
        return out;
    }

    public static float sum(Tensor a) {
        return f32(a).sum(a.getStorage(), a.getOffset(), a.getShape().numel());
    }

    public static float mean(Tensor a) {
        return f32(a).mean(a.getStorage(), a.getOffset(), a.getShape().numel());
    }

    public static Storage sumDim(Tensor a, int dim) {
        if (!a.isContiguous()) a = a.contiguous();
        int[] dims = a.getShape().dims();
        int outLen = a.getShape().numel() / dims[dim];
        Storage out = alloc(a, outLen);
        f32(a).sumDim(a.getStorage(), a.getOffset(), dims, dim, out);
        return out;
    }

    public static Storage maxDim(Tensor a, int dim) {
        if (!a.isContiguous()) a = a.contiguous();
        int[] dims = a.getShape().dims();
        int outLen = a.getShape().numel() / dims[dim];
        Storage out = alloc(a, outLen);
        f32(a).maxDim(a.getStorage(), a.getOffset(), dims, dim, out);
        return out;
    }

    public static Storage argmaxDim(Tensor a, int dim) {
        if (!a.isContiguous()) a = a.contiguous();
        int[] dims = a.getShape().dims();
        int outLen = a.getShape().numel() / dims[dim];
        Storage out = alloc(a, outLen);
        f32(a).argmaxDim(a.getStorage(), a.getOffset(), dims, dim, out);
        return out;
    }

    public static Storage unaryOp(VectorOperators.Unary op, Tensor a) {
        Storage out = alloc(a, a.getShape().numel());
        unaryOp(op, a, out);
        return out;
    }

    public static void unaryOp(VectorOperators.Unary op, Tensor a, Storage out) {
        f32(a).unary(op, a.getStorage(), a.getOffset(), out, 0, a.getShape().numel());
    }

    public static Storage binaryOp(VectorOperators.Binary op, Tensor a, Tensor b, Shape resultShape) {
        Storage out = alloc(a, resultShape.numel());
        binaryOp(op, a, b, resultShape, out);
        return out;
    }

    public static void binaryOp(VectorOperators.Binary op, Tensor a, Tensor b, Shape resultShape, Storage out) {
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        int n = resultShape.numel();
        F32KernelTable k = f32(a);

        if (aShape.equals(bShape) && a.isContiguous() && b.isContiguous()) {
            k.binary(op,
                    a.getStorage(), a.getOffset(),
                    b.getStorage(), b.getOffset(),
                    out, 0, n);
        } else {
            k.binaryBroadcast(op,
                    a.getStorage(), aShape.broadcastStridesTo(resultShape), a.getOffset(),
                    b.getStorage(), bShape.broadcastStridesTo(resultShape), b.getOffset(),
                    out, resultShape);
        }
    }

    public static Storage binaryOpConst(VectorOperators.Binary op, Tensor a, float b) {
        Storage out = alloc(a, a.getShape().numel());
        binaryOpConst(op, a, b, out);
        return out;
    }

    public static void binaryOpConst(VectorOperators.Binary op, Tensor a, float b, Storage out) {
        f32(a).binaryConst(op, a.getStorage(), a.getOffset(), b, out, 0, a.getShape().numel());
    }

    public static Storage rbinaryOpConst(VectorOperators.Binary op, float a, Tensor b) {
        Storage out = alloc(b, b.getShape().numel());
        rbinaryOpConst(op, a, b, out);
        return out;
    }

    public static void rbinaryOpConst(VectorOperators.Binary op, float a, Tensor b, Storage out) {
        f32(b).rbinaryConst(op, a, b.getStorage(), b.getOffset(), out, 0, b.getShape().numel());
    }

    public static Storage comparisonOp(VectorOperators.Comparison op, Tensor a, Tensor b, Shape resultShape) {
        Storage out = alloc(a, resultShape.numel());
        comparisonOp(op, a, b, resultShape, out);
        return out;
    }

    public static void comparisonOp(VectorOperators.Comparison op, Tensor a, Tensor b, Shape resultShape, Storage out) {
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        int n = resultShape.numel();
        F32KernelTable k = f32(a);

        if (aShape.equals(bShape) && a.isContiguous() && b.isContiguous()) {
            k.compare(op,
                    a.getStorage(), a.getOffset(),
                    b.getStorage(), b.getOffset(),
                    out, 0, n);
        } else {
            k.compareBroadcast(op,
                    a.getStorage(), aShape.broadcastStridesTo(resultShape), a.getOffset(),
                    b.getStorage(), bShape.broadcastStridesTo(resultShape), b.getOffset(),
                    out, resultShape);
        }
    }

    public static Storage comparisonOpConst(VectorOperators.Comparison op, Tensor a, float b) {
        Storage out = alloc(a, a.getShape().numel());
        comparisonOpConst(op, a, b, out);
        return out;
    }

    public static void comparisonOpConst(VectorOperators.Comparison op, Tensor a, float b, Storage out) {
        f32(a).compareConst(op, a.getStorage(), a.getOffset(), b, out, 0, a.getShape().numel());
    }

    public static Storage ternaryOp(VectorOperators.Ternary op, Tensor a, Tensor b, Tensor c, Shape resultShape) {
        Storage out = alloc(a, resultShape.numel());
        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        Shape cShape = c.getShape();
        Preconditions.checkState(aShape.equals(bShape) && bShape.equals(cShape)
                && a.isContiguous() && b.isContiguous() && c.isContiguous(),
                "ternary broadcast not yet supported");

        f32(a).ternary(op,
                a.getStorage(), a.getOffset(),
                b.getStorage(), b.getOffset(),
                c.getStorage(), c.getOffset(),
                out, 0, resultShape.numel());
        return out;
    }
}
