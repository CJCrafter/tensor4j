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

import java.util.Arrays;

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
     * Matrix multiply with N-D batched broadcasting (PyTorch semantics):
     * the last two dims of each input are the matrix dims; everything before
     * is a batch shape, broadcast against the other input's batch shape.
     * Both inputs are forced contiguous before dispatch.
     */
    public static Storage matmul(Tensor a, Tensor b, Shape resultShape) {
        if (!a.isContiguous()) a = a.contiguous();
        if (!b.isContiguous()) b = b.contiguous();

        Shape aShape = a.getShape();
        Shape bShape = b.getShape();
        int aDim = aShape.dimensions();
        int bDim = bShape.dimensions();

        int m = aShape.dim(aDim - 2);
        int kDim = aShape.dim(aDim - 1);
        int n = bShape.dim(bDim - 1);

        Storage out = alloc(a, resultShape.numel());
        // matmul accumulates into c; native blocks are not zero-initialized
        out.segment().fill((byte) 0);

        F32KernelTable k = f32(a);

        // Fast path: plain 2D @ 2D, no batching.
        if (aDim == 2 && bDim == 2) {
            k.matmul(a.getStorage(), a.getOffset(), m, kDim,
                    b.getStorage(), b.getOffset(), n,
                    out, 0);
            return out;
        }

        // Batched path. Output batch shape is everything before the last 2 dims.
        int outNdim = resultShape.dimensions();
        int batchNdim = outNdim - 2;
        int[] outBatch = Arrays.copyOf(resultShape.dims(), batchNdim);

        int[] aBatchPadded = padLeftWithOnes(Arrays.copyOf(aShape.dims(), aDim - 2), batchNdim);
        int[] bBatchPadded = padLeftWithOnes(Arrays.copyOf(bShape.dims(), bDim - 2), batchNdim);

        int aSliceSize = m * kDim;
        int bSliceSize = kDim * n;
        int cSliceSize = m * n;

        // Per-batch-axis stride into the source; 0 for axes that are broadcast (size 1).
        int[] aStrides = sliceStridesFor(aBatchPadded, aSliceSize);
        int[] bStrides = sliceStridesFor(bBatchPadded, bSliceSize);

        int totalBatches = 1;
        for (int dim : outBatch) totalBatches *= dim;

        int[] indices = new int[batchNdim];
        int cOff = 0;
        for (int batch = 0; batch < totalBatches; batch++) {
            int aOff = a.getOffset();
            int bOff = b.getOffset();
            for (int d = 0; d < batchNdim; d++) {
                aOff += indices[d] * aStrides[d];
                bOff += indices[d] * bStrides[d];
            }
            k.matmul(a.getStorage(), aOff, m, kDim,
                    b.getStorage(), bOff, n,
                    out, cOff);
            cOff += cSliceSize;

            // increment multi-dim batch index, rightmost first
            for (int d = batchNdim - 1; d >= 0; d--) {
                if (++indices[d] < outBatch[d]) break;
                indices[d] = 0;
            }
        }

        return out;
    }

    private static int[] padLeftWithOnes(int[] batch, int targetLen) {
        if (batch.length == targetLen) return batch;
        int[] result = new int[targetLen];
        Arrays.fill(result, 0, targetLen - batch.length, 1);
        System.arraycopy(batch, 0, result, targetLen - batch.length, batch.length);
        return result;
    }

    private static int[] sliceStridesFor(int[] paddedBatch, int sliceSize) {
        int n = paddedBatch.length;
        int[] strides = new int[n];
        int s = sliceSize;
        for (int d = n - 1; d >= 0; d--) {
            if (paddedBatch[d] == 1) {
                strides[d] = 0;
            } else {
                strides[d] = s;
                s *= paddedBatch[d];
            }
        }
        return strides;
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
