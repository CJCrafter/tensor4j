package com.cjcrafter.tensor4j.ops;

import com.cjcrafter.tensor4j.Shape;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import static jdk.incubator.vector.VectorOperators.*;

/**
 * Compute kernels operating on a {@link MemorySegment} of floats.
 */
public final class SegmentKernels {

    private static final int REGISTER_BLOCKS = 8;
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final FloatVector ZERO = FloatVector.zero(SPECIES);
    private static final FloatVector ONE = FloatVector.broadcast(SPECIES, 1f);
    private static final ByteOrder NATIVE = ByteOrder.nativeOrder();
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT;
    private static final long F32_BYTES = 4L;

    private SegmentKernels() {
    }

    private static float applyScalarUnary(VectorOperators.Unary op, float a) {
        if (op == ABS) return Math.abs(a);
        else if (op == NEG) return -a;
        else if (op == SIN) return (float) Math.sin(a);
        else if (op == COS) return (float) Math.cos(a);
        else if (op == TAN) return (float) Math.tan(a);
        else if (op == ASIN) return (float) Math.asin(a);
        else if (op == ACOS) return (float) Math.acos(a);
        else if (op == ATAN) return (float) Math.atan(a);
        else if (op == EXP) return (float) Math.exp(a);
        else if (op == LOG) return (float) Math.log(a);
        else if (op == LOG10) return (float) Math.log10(a);
        else if (op == SQRT) return (float) Math.sqrt(a);
        else if (op == CBRT) return (float) Math.cbrt(a);
        else if (op == SINH) return (float) Math.sinh(a);
        else if (op == COSH) return (float) Math.cosh(a);
        else if (op == TANH) return (float) Math.tanh(a);
        else if (op == EXPM1) return (float) Math.expm1(a);
        else if (op == LOG1P) return (float) Math.log1p(a);
        else throw new UnsupportedOperationException("unary op: " + op.name());
    }

    private static float applyScalarBinary(VectorOperators.Binary op, float a, float b) {
        if (op == ADD) return a + b;
        else if (op == SUB) return a - b;
        else if (op == MUL) return a * b;
        else if (op == DIV) return a / b;
        else if (op == MIN) return Math.min(a, b);
        else if (op == MAX) return Math.max(a, b);
        else if (op == ATAN2) return (float) Math.atan2(a, b);
        else if (op == POW) return (float) Math.pow(a, b);
        else if (op == HYPOT) return (float) Math.hypot(a, b);
        else throw new UnsupportedOperationException("binary op: " + op.name());
    }

    private static float applyScalarComparison(VectorOperators.Comparison op, float a, float b) {
        if (op == EQ) return a == b ? 1f : 0f;
        else if (op == NE) return a != b ? 1f : 0f;
        else if (op == LT) return a < b ? 1f : 0f;
        else if (op == LE) return a <= b ? 1f : 0f;
        else if (op == GT) return a > b ? 1f : 0f;
        else if (op == GE) return a >= b ? 1f : 0f;
        else throw new UnsupportedOperationException("comparison op: " + op.name());
    }

    // -------------- //
    // --- MATMUL --- //
    // -------------- //

    private static final int K_TILE = 64;
    private static final int J_TILE = 128;
    private static final int I_TILE = 256;

    public static void matmul(MemorySegment a, long aOff, int m, int k,
                              MemorySegment b, long bOff, int n,
                              MemorySegment c, long cOff) {
        float[] bPacked = new float[K_TILE * J_TILE];
        MemorySegment bPackedSeg = MemorySegment.ofArray(bPacked);

        for (int it = 0; it < m; it += I_TILE) {
            int iEnd = Math.min(m, it + I_TILE);

        for (int jt = 0; jt < n; jt += J_TILE) {
            int jEnd = Math.min(n, jt + J_TILE);
            int jBound = SPECIES.loopBound(jEnd - jt) + jt;

        for (int pt = 0; pt < k; pt += K_TILE) {
            int pEnd = Math.min(k, pt + K_TILE);
            int jTileWidth = jEnd - jt;

            for (int p = pt; p < pEnd; p++) {
                MemorySegment.copy(b, (bOff + p * n + jt) * F32_BYTES,
                        bPackedSeg, (long)(p - pt) * jTileWidth * F32_BYTES,
                        (long) jTileWidth * F32_BYTES);
            }

            int i = it;
            for (; i <= iEnd - REGISTER_BLOCKS; i += REGISTER_BLOCKS) {
                long aRowOff0 = (aOff + i * k) * F32_BYTES;
                long aRowOff1 = (aOff + (i + 1) * k) * F32_BYTES;
                long aRowOff2 = (aOff + (i + 2) * k) * F32_BYTES;
                long aRowOff3 = (aOff + (i + 3) * k) * F32_BYTES;
                long aRowOff4 = (aOff + (i + 4) * k) * F32_BYTES;
                long aRowOff5 = (aOff + (i + 5) * k) * F32_BYTES;
                long aRowOff6 = (aOff + (i + 6) * k) * F32_BYTES;
                long aRowOff7 = (aOff + (i + 7) * k) * F32_BYTES;
                long cRowOff0 = (cOff + i * n) * F32_BYTES;
                long cRowOff1 = (cOff + (i + 1) * n) * F32_BYTES;
                long cRowOff2 = (cOff + (i + 2) * n) * F32_BYTES;
                long cRowOff3 = (cOff + (i + 3) * n) * F32_BYTES;
                long cRowOff4 = (cOff + (i + 4) * n) * F32_BYTES;
                long cRowOff5 = (cOff + (i + 5) * n) * F32_BYTES;
                long cRowOff6 = (cOff + (i + 6) * n) * F32_BYTES;
                long cRowOff7 = (cOff + (i + 7) * n) * F32_BYTES;

                int j = jt;
                for (; j < jBound; j += SPECIES.length()) {
                    long jBytes = (long) j * F32_BYTES;
                    var vc0 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff0 + jBytes, NATIVE);
                    var vc1 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff1 + jBytes, NATIVE);
                    var vc2 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff2 + jBytes, NATIVE);
                    var vc3 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff3 + jBytes, NATIVE);
                    var vc4 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff4 + jBytes, NATIVE);
                    var vc5 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff5 + jBytes, NATIVE);
                    var vc6 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff6 + jBytes, NATIVE);
                    var vc7 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff7 + jBytes, NATIVE);

                    for (int p = pt; p < pEnd; p++) {
                        long bPackedOffBytes = ((long)(p - pt) * jTileWidth + (j - jt)) * F32_BYTES;
                        var vb = FloatVector.fromMemorySegment(SPECIES, bPackedSeg, bPackedOffBytes, NATIVE);
                        long pBytes = (long) p * F32_BYTES;
                        vc0 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff0 + pBytes)).fma(vb, vc0);
                        vc1 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff1 + pBytes)).fma(vb, vc1);
                        vc2 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff2 + pBytes)).fma(vb, vc2);
                        vc3 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff3 + pBytes)).fma(vb, vc3);
                        vc4 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff4 + pBytes)).fma(vb, vc4);
                        vc5 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff5 + pBytes)).fma(vb, vc5);
                        vc6 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff6 + pBytes)).fma(vb, vc6);
                        vc7 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff7 + pBytes)).fma(vb, vc7);
                    }

                    vc0.intoMemorySegment(c, cRowOff0 + jBytes, NATIVE);
                    vc1.intoMemorySegment(c, cRowOff1 + jBytes, NATIVE);
                    vc2.intoMemorySegment(c, cRowOff2 + jBytes, NATIVE);
                    vc3.intoMemorySegment(c, cRowOff3 + jBytes, NATIVE);
                    vc4.intoMemorySegment(c, cRowOff4 + jBytes, NATIVE);
                    vc5.intoMemorySegment(c, cRowOff5 + jBytes, NATIVE);
                    vc6.intoMemorySegment(c, cRowOff6 + jBytes, NATIVE);
                    vc7.intoMemorySegment(c, cRowOff7 + jBytes, NATIVE);
                }

                if (j < jEnd) {
                    var mask = SPECIES.indexInRange(j - jt, jEnd - jt);
                    long jBytes = (long) j * F32_BYTES;
                    var vc0 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff0 + jBytes, NATIVE, mask);
                    var vc1 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff1 + jBytes, NATIVE, mask);
                    var vc2 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff2 + jBytes, NATIVE, mask);
                    var vc3 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff3 + jBytes, NATIVE, mask);
                    var vc4 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff4 + jBytes, NATIVE, mask);
                    var vc5 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff5 + jBytes, NATIVE, mask);
                    var vc6 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff6 + jBytes, NATIVE, mask);
                    var vc7 = FloatVector.fromMemorySegment(SPECIES, c, cRowOff7 + jBytes, NATIVE, mask);

                    for (int p = pt; p < pEnd; p++) {
                        long bPackedOffBytes = ((long)(p - pt) * jTileWidth + (j - jt)) * F32_BYTES;
                        var vb = FloatVector.fromMemorySegment(SPECIES, bPackedSeg, bPackedOffBytes, NATIVE, mask);
                        long pBytes = (long) p * F32_BYTES;
                        vc0 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff0 + pBytes)).fma(vb, vc0);
                        vc1 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff1 + pBytes)).fma(vb, vc1);
                        vc2 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff2 + pBytes)).fma(vb, vc2);
                        vc3 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff3 + pBytes)).fma(vb, vc3);
                        vc4 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff4 + pBytes)).fma(vb, vc4);
                        vc5 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff5 + pBytes)).fma(vb, vc5);
                        vc6 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff6 + pBytes)).fma(vb, vc6);
                        vc7 = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff7 + pBytes)).fma(vb, vc7);
                    }

                    vc0.intoMemorySegment(c, cRowOff0 + jBytes, NATIVE, mask);
                    vc1.intoMemorySegment(c, cRowOff1 + jBytes, NATIVE, mask);
                    vc2.intoMemorySegment(c, cRowOff2 + jBytes, NATIVE, mask);
                    vc3.intoMemorySegment(c, cRowOff3 + jBytes, NATIVE, mask);
                    vc4.intoMemorySegment(c, cRowOff4 + jBytes, NATIVE, mask);
                    vc5.intoMemorySegment(c, cRowOff5 + jBytes, NATIVE, mask);
                    vc6.intoMemorySegment(c, cRowOff6 + jBytes, NATIVE, mask);
                    vc7.intoMemorySegment(c, cRowOff7 + jBytes, NATIVE, mask);
                }
            }

            for (; i < iEnd; i++) {
                long aRowOff = (aOff + i * k) * F32_BYTES;
                long cRowOff = (cOff + i * n) * F32_BYTES;
                int j = jt;
                for (; j < jBound; j += SPECIES.length()) {
                    long jBytes = (long) j * F32_BYTES;
                    var vc = FloatVector.fromMemorySegment(SPECIES, c, cRowOff + jBytes, NATIVE);
                    for (int p = pt; p < pEnd; p++) {
                        long bPackedOffBytes = ((long)(p - pt) * jTileWidth + (j - jt)) * F32_BYTES;
                        var aVal = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff + (long) p * F32_BYTES));
                        var vb = FloatVector.fromMemorySegment(SPECIES, bPackedSeg, bPackedOffBytes, NATIVE);
                        vc = aVal.fma(vb, vc);
                    }
                    vc.intoMemorySegment(c, cRowOff + jBytes, NATIVE);
                }
                if (j < jEnd) {
                    var mask = SPECIES.indexInRange(j - jt, jEnd - jt);
                    long jBytes = (long) j * F32_BYTES;
                    var vc = FloatVector.fromMemorySegment(SPECIES, c, cRowOff + jBytes, NATIVE, mask);
                    for (int p = pt; p < pEnd; p++) {
                        long bPackedOffBytes = ((long)(p - pt) * jTileWidth + (j - jt)) * F32_BYTES;
                        var aVal = FloatVector.broadcast(SPECIES, a.get(F32, aRowOff + (long) p * F32_BYTES));
                        var vb = FloatVector.fromMemorySegment(SPECIES, bPackedSeg, bPackedOffBytes, NATIVE, mask);
                        vc = aVal.fma(vb, vc);
                    }
                    vc.intoMemorySegment(c, cRowOff + jBytes, NATIVE, mask);
                }
            }
        }
        }
        }
    }

    // ------------------ //
    // --- REDUCTIONS --- //
    // ------------------ //

    public static float sum(MemorySegment a, long aOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        var vs0 = ZERO;
        var vs1 = ZERO;
        var vs2 = ZERO;
        var vs3 = ZERO;
        var vs4 = ZERO;
        var vs5 = ZERO;
        var vs6 = ZERO;
        var vs7 = ZERO;

        int laneBytes = SPECIES.length() * 4;
        for (; i < blockBound; i += step) {
            long base = (aOff + i) * F32_BYTES;
            vs0 = vs0.add(FloatVector.fromMemorySegment(SPECIES, a, base, NATIVE));
            vs1 = vs1.add(FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes, NATIVE));
            vs2 = vs2.add(FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 2L, NATIVE));
            vs3 = vs3.add(FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 3L, NATIVE));
            vs4 = vs4.add(FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 4L, NATIVE));
            vs5 = vs5.add(FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 5L, NATIVE));
            vs6 = vs6.add(FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 6L, NATIVE));
            vs7 = vs7.add(FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 7L, NATIVE));
        }

        var vsum = vs0.add(vs1).add(vs2).add(vs3)
                .add(vs4).add(vs5).add(vs6).add(vs7);

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            vsum = vsum.add(FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE));
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);

        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE, mask);
            sum += va.reduceLanes(VectorOperators.ADD, mask);
        }
        return sum;
    }

    public static float mean(MemorySegment a, long aOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        var vs0 = ZERO; var vc0 = ZERO;
        var vs1 = ZERO; var vc1 = ZERO;
        var vs2 = ZERO; var vc2 = ZERO;
        var vs3 = ZERO; var vc3 = ZERO;
        var vs4 = ZERO; var vc4 = ZERO;
        var vs5 = ZERO; var vc5 = ZERO;
        var vs6 = ZERO; var vc6 = ZERO;
        var vs7 = ZERO; var vc7 = ZERO;

        int laneBytes = SPECIES.length() * 4;
        for (; i < blockBound; i += step) {
            long base = (aOff + i) * F32_BYTES;
            var va0 = FloatVector.fromMemorySegment(SPECIES, a, base, NATIVE);
            var va1 = FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes, NATIVE);
            var va2 = FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 2L, NATIVE);
            var va3 = FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 3L, NATIVE);
            var va4 = FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 4L, NATIVE);
            var va5 = FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 5L, NATIVE);
            var va6 = FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 6L, NATIVE);
            var va7 = FloatVector.fromMemorySegment(SPECIES, a, base + laneBytes * 7L, NATIVE);

            var y0 = va0.sub(vc0); var t0 = vs0.add(y0);
            var y1 = va1.sub(vc1); var t1 = vs1.add(y1);
            var y2 = va2.sub(vc2); var t2 = vs2.add(y2);
            var y3 = va3.sub(vc3); var t3 = vs3.add(y3);
            var y4 = va4.sub(vc4); var t4 = vs4.add(y4);
            var y5 = va5.sub(vc5); var t5 = vs5.add(y5);
            var y6 = va6.sub(vc6); var t6 = vs6.add(y6);
            var y7 = va7.sub(vc7); var t7 = vs7.add(y7);

            vc0 = t0.sub(vs0).sub(y0);
            vc1 = t1.sub(vs1).sub(y1);
            vc2 = t2.sub(vs2).sub(y2);
            vc3 = t3.sub(vs3).sub(y3);
            vc4 = t4.sub(vs4).sub(y4);
            vc5 = t5.sub(vs5).sub(y5);
            vc6 = t6.sub(vs6).sub(y6);
            vc7 = t7.sub(vs7).sub(y7);

            vs0 = t0; vs1 = t1; vs2 = t2; vs3 = t3;
            vs4 = t4; vs5 = t5; vs6 = t6; vs7 = t7;
        }

        var vsum = vs0.add(vs1).add(vs2).add(vs3)
                .add(vs4).add(vs5).add(vs6).add(vs7);
        var vcomp = vc0.add(vc1).add(vc2).add(vc3)
                .add(vc4).add(vc5).add(vc6).add(vc7);

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE);
            var y = va.sub(vcomp);
            var t = vsum.add(y);
            vcomp = t.sub(vsum).sub(y);
            vsum = t;
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float comp = vcomp.reduceLanes(VectorOperators.ADD);

        for (; i < len; i++) {
            float val = a.get(F32, (aOff + i) * F32_BYTES);
            float y = val - comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }

        return (sum - comp) / len;
    }

    public static void sumDim(MemorySegment a, long aOff, int[] dims, int dim, MemorySegment out) {
        int ndim = dims.length;

        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dims[i];
        int reduceDim = dims[dim];
        int inner = 1;
        for (int i = dim + 1; i < ndim; i++) inner *= dims[i];

        for (int o = 0; o < outer; o++) {
            int outBase = o * inner;
            int aBase = o * reduceDim * inner;

            for (int i = 0; i < inner; i++)
                out.set(F32, (long) (outBase + i) * F32_BYTES, 0f);

            for (int r = 0; r < reduceDim; r++) {
                long aSlice = aOff + aBase + r * inner;

                int i = 0;
                int bound = SPECIES.loopBound(inner);
                for (; i < bound; i += SPECIES.length()) {
                    var va = FloatVector.fromMemorySegment(SPECIES, a, (aSlice + i) * F32_BYTES, NATIVE);
                    var vo = FloatVector.fromMemorySegment(SPECIES, out, (long) (outBase + i) * F32_BYTES, NATIVE);
                    vo.add(va).intoMemorySegment(out, (long) (outBase + i) * F32_BYTES, NATIVE);
                }
                for (; i < inner; i++) {
                    float prev = out.get(F32, (long) (outBase + i) * F32_BYTES);
                    float add = a.get(F32, (aSlice + i) * F32_BYTES);
                    out.set(F32, (long) (outBase + i) * F32_BYTES, prev + add);
                }
            }
        }
    }

    public static void maxDim(MemorySegment a, long aOff, int[] dims, int dim, MemorySegment out) {
        int ndim = dims.length;

        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dims[i];
        int reduceDim = dims[dim];
        int inner = 1;
        for (int i = dim + 1; i < ndim; i++) inner *= dims[i];

        for (int o = 0; o < outer; o++) {
            int outBase = o * inner;
            long aBase = aOff + (long) o * reduceDim * inner;

            for (int i = 0; i < inner; i++)
                out.set(F32, (long) (outBase + i) * F32_BYTES, a.get(F32, (aBase + i) * F32_BYTES));

            for (int r = 1; r < reduceDim; r++) {
                long aSlice = aBase + (long) r * inner;
                for (int i = 0; i < inner; i++) {
                    float v = a.get(F32, (aSlice + i) * F32_BYTES);
                    float cur = out.get(F32, (long) (outBase + i) * F32_BYTES);
                    if (v > cur)
                        out.set(F32, (long) (outBase + i) * F32_BYTES, v);
                }
            }
        }
    }

    public static void argmaxDim(MemorySegment a, long aOff, int[] dims, int dim, MemorySegment out) {
        int ndim = dims.length;

        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dims[i];
        int reduceDim = dims[dim];
        int inner = 1;
        for (int i = dim + 1; i < ndim; i++) inner *= dims[i];

        for (int o = 0; o < outer; o++) {
            int outBase = o * inner;
            long aBase = aOff + (long) o * reduceDim * inner;

            for (int i = 0; i < inner; i++) {
                float maxVal = a.get(F32, (aBase + i) * F32_BYTES);
                int maxIdx = 0;
                for (int r = 1; r < reduceDim; r++) {
                    float val = a.get(F32, (aBase + (long) r * inner + i) * F32_BYTES);
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = r;
                    }
                }
                out.set(F32, (long) (outBase + i) * F32_BYTES, maxIdx);
            }
        }
    }

    // -------------------------- //
    // --- UNARY ELEMENT-WISE --- //
    // -------------------------- //

    public static void unaryContiguous(VectorOperators.Unary op,
                                       MemorySegment a, long aOff,
                                       MemorySegment out, long outOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int laneBytes = SPECIES.length() * 4;

        for (; i < blockBound; i += step) {
            long aBase = (aOff + i) * F32_BYTES;
            long oBase = (outOff + i) * F32_BYTES;
            var va0 = FloatVector.fromMemorySegment(SPECIES, a, aBase, NATIVE);
            var va1 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes, NATIVE);
            var va2 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 2L, NATIVE);
            var va3 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 3L, NATIVE);
            var va4 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 4L, NATIVE);
            var va5 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 5L, NATIVE);
            var va6 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 6L, NATIVE);
            var va7 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 7L, NATIVE);
            va0.lanewise(op).intoMemorySegment(out, oBase, NATIVE);
            va1.lanewise(op).intoMemorySegment(out, oBase + laneBytes, NATIVE);
            va2.lanewise(op).intoMemorySegment(out, oBase + laneBytes * 2L, NATIVE);
            va3.lanewise(op).intoMemorySegment(out, oBase + laneBytes * 3L, NATIVE);
            va4.lanewise(op).intoMemorySegment(out, oBase + laneBytes * 4L, NATIVE);
            va5.lanewise(op).intoMemorySegment(out, oBase + laneBytes * 5L, NATIVE);
            va6.lanewise(op).intoMemorySegment(out, oBase + laneBytes * 6L, NATIVE);
            va7.lanewise(op).intoMemorySegment(out, oBase + laneBytes * 7L, NATIVE);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE);
            va.lanewise(op).intoMemorySegment(out, (outOff + i) * F32_BYTES, NATIVE);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE, mask);
            va.lanewise(op).intoMemorySegment(out, (outOff + i) * F32_BYTES, NATIVE, mask);
        }
    }

    // --------------------------- //
    // --- BINARY ELEMENT-WISE --- //
    // --------------------------- //

    public static void binaryContiguous(VectorOperators.Binary op,
                                        MemorySegment a, long aOff,
                                        MemorySegment b, long bOff,
                                        MemorySegment out, long outOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int laneBytes = SPECIES.length() * 4;

        for (; i < blockBound; i += step) {
            long aBase = (aOff + i) * F32_BYTES;
            long bBase = (bOff + i) * F32_BYTES;
            long oBase = (outOff + i) * F32_BYTES;
            var va0 = FloatVector.fromMemorySegment(SPECIES, a, aBase, NATIVE);
            var va1 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes, NATIVE);
            var va2 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 2L, NATIVE);
            var va3 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 3L, NATIVE);
            var va4 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 4L, NATIVE);
            var va5 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 5L, NATIVE);
            var va6 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 6L, NATIVE);
            var va7 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 7L, NATIVE);
            var vb0 = FloatVector.fromMemorySegment(SPECIES, b, bBase, NATIVE);
            var vb1 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes, NATIVE);
            var vb2 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 2L, NATIVE);
            var vb3 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 3L, NATIVE);
            var vb4 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 4L, NATIVE);
            var vb5 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 5L, NATIVE);
            var vb6 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 6L, NATIVE);
            var vb7 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 7L, NATIVE);
            va0.lanewise(op, vb0).intoMemorySegment(out, oBase, NATIVE);
            va1.lanewise(op, vb1).intoMemorySegment(out, oBase + laneBytes, NATIVE);
            va2.lanewise(op, vb2).intoMemorySegment(out, oBase + laneBytes * 2L, NATIVE);
            va3.lanewise(op, vb3).intoMemorySegment(out, oBase + laneBytes * 3L, NATIVE);
            va4.lanewise(op, vb4).intoMemorySegment(out, oBase + laneBytes * 4L, NATIVE);
            va5.lanewise(op, vb5).intoMemorySegment(out, oBase + laneBytes * 5L, NATIVE);
            va6.lanewise(op, vb6).intoMemorySegment(out, oBase + laneBytes * 6L, NATIVE);
            va7.lanewise(op, vb7).intoMemorySegment(out, oBase + laneBytes * 7L, NATIVE);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, (bOff + i) * F32_BYTES, NATIVE);
            va.lanewise(op, vb).intoMemorySegment(out, (outOff + i) * F32_BYTES, NATIVE);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE, mask);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, (bOff + i) * F32_BYTES, NATIVE, mask);
            va.lanewise(op, vb).intoMemorySegment(out, (outOff + i) * F32_BYTES, NATIVE, mask);
        }
    }

    public static void binaryContiguous(VectorOperators.Binary op,
                                        MemorySegment a, long aOff,
                                        MemorySegment b, long bOff,
                                        MemorySegment out, int len) {
        binaryContiguous(op, a, aOff, b, bOff, out, 0, len);
    }

    public static void binaryConst(VectorOperators.Binary op,
                                   MemorySegment a, long aOff,
                                   float b,
                                   MemorySegment out, int len) {
        var vb = FloatVector.broadcast(SPECIES, b);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int laneBytes = SPECIES.length() * 4;

        for (; i < blockBound; i += step) {
            long aBase = (aOff + i) * F32_BYTES;
            long oBase = (long) i * F32_BYTES;
            var va0 = FloatVector.fromMemorySegment(SPECIES, a, aBase, NATIVE);
            var va1 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes, NATIVE);
            var va2 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 2L, NATIVE);
            var va3 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 3L, NATIVE);
            var va4 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 4L, NATIVE);
            var va5 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 5L, NATIVE);
            var va6 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 6L, NATIVE);
            var va7 = FloatVector.fromMemorySegment(SPECIES, a, aBase + laneBytes * 7L, NATIVE);
            va0.lanewise(op, vb).intoMemorySegment(out, oBase, NATIVE);
            va1.lanewise(op, vb).intoMemorySegment(out, oBase + laneBytes, NATIVE);
            va2.lanewise(op, vb).intoMemorySegment(out, oBase + laneBytes * 2L, NATIVE);
            va3.lanewise(op, vb).intoMemorySegment(out, oBase + laneBytes * 3L, NATIVE);
            va4.lanewise(op, vb).intoMemorySegment(out, oBase + laneBytes * 4L, NATIVE);
            va5.lanewise(op, vb).intoMemorySegment(out, oBase + laneBytes * 5L, NATIVE);
            va6.lanewise(op, vb).intoMemorySegment(out, oBase + laneBytes * 6L, NATIVE);
            va7.lanewise(op, vb).intoMemorySegment(out, oBase + laneBytes * 7L, NATIVE);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE);
            va.lanewise(op, vb).intoMemorySegment(out, (long) i * F32_BYTES, NATIVE);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE, mask);
            va.lanewise(op, vb).intoMemorySegment(out, (long) i * F32_BYTES, NATIVE, mask);
        }
    }

    public static void rbinaryConst(VectorOperators.Binary op,
                                    float a,
                                    MemorySegment b, long bOff,
                                    MemorySegment out, int len) {
        var va = FloatVector.broadcast(SPECIES, a);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int laneBytes = SPECIES.length() * 4;

        for (; i < blockBound; i += step) {
            long bBase = (bOff + i) * F32_BYTES;
            long oBase = (long) i * F32_BYTES;
            var vb0 = FloatVector.fromMemorySegment(SPECIES, b, bBase, NATIVE);
            var vb1 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes, NATIVE);
            var vb2 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 2L, NATIVE);
            var vb3 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 3L, NATIVE);
            var vb4 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 4L, NATIVE);
            var vb5 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 5L, NATIVE);
            var vb6 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 6L, NATIVE);
            var vb7 = FloatVector.fromMemorySegment(SPECIES, b, bBase + laneBytes * 7L, NATIVE);
            va.lanewise(op, vb0).intoMemorySegment(out, oBase, NATIVE);
            va.lanewise(op, vb1).intoMemorySegment(out, oBase + laneBytes, NATIVE);
            va.lanewise(op, vb2).intoMemorySegment(out, oBase + laneBytes * 2L, NATIVE);
            va.lanewise(op, vb3).intoMemorySegment(out, oBase + laneBytes * 3L, NATIVE);
            va.lanewise(op, vb4).intoMemorySegment(out, oBase + laneBytes * 4L, NATIVE);
            va.lanewise(op, vb5).intoMemorySegment(out, oBase + laneBytes * 5L, NATIVE);
            va.lanewise(op, vb6).intoMemorySegment(out, oBase + laneBytes * 6L, NATIVE);
            va.lanewise(op, vb7).intoMemorySegment(out, oBase + laneBytes * 7L, NATIVE);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var vb = FloatVector.fromMemorySegment(SPECIES, b, (bOff + i) * F32_BYTES, NATIVE);
            va.lanewise(op, vb).intoMemorySegment(out, (long) i * F32_BYTES, NATIVE);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, (bOff + i) * F32_BYTES, NATIVE, mask);
            va.lanewise(op, vb).intoMemorySegment(out, (long) i * F32_BYTES, NATIVE, mask);
        }
    }

    public static void binaryBroadcast(VectorOperators.Binary op,
                                       MemorySegment a, int[] aStrides, long aOff,
                                       MemorySegment b, int[] bStrides, long bOff,
                                       MemorySegment out, Shape resultShape) {
        int ndim = resultShape.dimensions();
        int numel = resultShape.numel();
        int[] indices = new int[ndim];

        for (int flat = 0; flat < numel; flat++) {
            long aIdx = aOff;
            long bIdx = bOff;
            for (int d = 0; d < ndim; d++) {
                aIdx += (long) indices[d] * aStrides[d];
                bIdx += (long) indices[d] * bStrides[d];
            }
            float av = a.get(F32, aIdx * F32_BYTES);
            float bv = b.get(F32, bIdx * F32_BYTES);
            out.set(F32, (long) flat * F32_BYTES, applyScalarBinary(op, av, bv));

            for (int d = ndim - 1; d >= 0; d--) {
                if (++indices[d] < resultShape.dim(d)) break;
                indices[d] = 0;
            }
        }
    }

    // ------------------------------- //
    // --- COMPARISON ELEMENT-WISE --- //
    // ------------------------------- //

    public static void compareContiguous(VectorOperators.Comparison op,
                                         MemorySegment a, long aOff,
                                         MemorySegment b, long bOff,
                                         MemorySegment out, long outOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            for (int r = 0; r < REGISTER_BLOCKS; r++) {
                int off = i + SPECIES.length() * r;
                var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + off) * F32_BYTES, NATIVE);
                var vb = FloatVector.fromMemorySegment(SPECIES, b, (bOff + off) * F32_BYTES, NATIVE);
                ZERO.blend(ONE, va.compare(op, vb)).intoMemorySegment(out, (outOff + off) * F32_BYTES, NATIVE);
            }
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, (bOff + i) * F32_BYTES, NATIVE);
            ZERO.blend(ONE, va.compare(op, vb)).intoMemorySegment(out, (outOff + i) * F32_BYTES, NATIVE);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE, mask);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, (bOff + i) * F32_BYTES, NATIVE, mask);
            ZERO.blend(ONE, va.compare(op, vb)).intoMemorySegment(out, (outOff + i) * F32_BYTES, NATIVE, mask);
        }
    }

    public static void compareContiguous(VectorOperators.Comparison op,
                                         MemorySegment a, long aOff,
                                         MemorySegment b, long bOff,
                                         MemorySegment out, int len) {
        compareContiguous(op, a, aOff, b, bOff, out, 0, len);
    }

    public static void compareConst(VectorOperators.Comparison op,
                                    MemorySegment a, long aOff,
                                    float b,
                                    MemorySegment out, int len) {
        var vb = FloatVector.broadcast(SPECIES, b);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            for (int r = 0; r < REGISTER_BLOCKS; r++) {
                int off = i + SPECIES.length() * r;
                var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + off) * F32_BYTES, NATIVE);
                ZERO.blend(ONE, va.compare(op, vb)).intoMemorySegment(out, (long) off * F32_BYTES, NATIVE);
            }
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE);
            ZERO.blend(ONE, va.compare(op, vb)).intoMemorySegment(out, (long) i * F32_BYTES, NATIVE);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE, mask);
            ZERO.blend(ONE, va.compare(op, vb)).intoMemorySegment(out, (long) i * F32_BYTES, NATIVE, mask);
        }
    }

    public static void compareBroadcast(VectorOperators.Comparison op,
                                        MemorySegment a, int[] aStrides, long aOff,
                                        MemorySegment b, int[] bStrides, long bOff,
                                        MemorySegment out, Shape resultShape) {
        int ndim = resultShape.dimensions();
        int numel = resultShape.numel();
        int[] indices = new int[ndim];

        for (int flat = 0; flat < numel; flat++) {
            long aIdx = aOff;
            long bIdx = bOff;
            for (int d = 0; d < ndim; d++) {
                aIdx += (long) indices[d] * aStrides[d];
                bIdx += (long) indices[d] * bStrides[d];
            }
            float av = a.get(F32, aIdx * F32_BYTES);
            float bv = b.get(F32, bIdx * F32_BYTES);
            out.set(F32, (long) flat * F32_BYTES, applyScalarComparison(op, av, bv));

            for (int d = ndim - 1; d >= 0; d--) {
                if (++indices[d] < resultShape.dim(d)) break;
                indices[d] = 0;
            }
        }
    }

    // ---------------------------- //
    // --- TERNARY ELEMENT-WISE --- //
    // ---------------------------- //

    public static void ternaryContiguous(VectorOperators.Ternary op,
                                         MemorySegment a, long aOff,
                                         MemorySegment b, long bOff,
                                         MemorySegment c, long cOff,
                                         MemorySegment out, long outOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int laneBytes = SPECIES.length() * 4;

        for (; i < blockBound; i += step) {
            long aBase = (aOff + i) * F32_BYTES;
            long bBase = (bOff + i) * F32_BYTES;
            long cBase = (cOff + i) * F32_BYTES;
            long oBase = (outOff + i) * F32_BYTES;
            for (int r = 0; r < REGISTER_BLOCKS; r++) {
                long lane = (long) laneBytes * r;
                var va = FloatVector.fromMemorySegment(SPECIES, a, aBase + lane, NATIVE);
                var vb = FloatVector.fromMemorySegment(SPECIES, b, bBase + lane, NATIVE);
                var vc = FloatVector.fromMemorySegment(SPECIES, c, cBase + lane, NATIVE);
                va.lanewise(op, vb, vc).intoMemorySegment(out, oBase + lane, NATIVE);
            }
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, (bOff + i) * F32_BYTES, NATIVE);
            var vc = FloatVector.fromMemorySegment(SPECIES, c, (cOff + i) * F32_BYTES, NATIVE);
            va.lanewise(op, vb, vc).intoMemorySegment(out, (outOff + i) * F32_BYTES, NATIVE);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromMemorySegment(SPECIES, a, (aOff + i) * F32_BYTES, NATIVE, mask);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, (bOff + i) * F32_BYTES, NATIVE, mask);
            var vc = FloatVector.fromMemorySegment(SPECIES, c, (cOff + i) * F32_BYTES, NATIVE, mask);
            va.lanewise(op, vb, vc).intoMemorySegment(out, (outOff + i) * F32_BYTES, NATIVE, mask);
        }
    }

    // -------------------- //
    // --- FUSED ADAMW  --- //
    // -------------------- //

    public static void adamw(MemorySegment p, long pOff,
                             MemorySegment g, long gOff,
                             MemorySegment m, long mOff,
                             MemorySegment v, long vOff,
                             int len,
                             float lr, float beta1, float beta2,
                             float biasCorr1, float biasCorr2,
                             float weightDecay, float eps) {

        var vBeta1      = FloatVector.broadcast(SPECIES, beta1);
        var vOneMinusB1 = FloatVector.broadcast(SPECIES, 1f - beta1);
        var vBeta2      = FloatVector.broadcast(SPECIES, beta2);
        var vOneMinusB2 = FloatVector.broadcast(SPECIES, 1f - beta2);
        var vLr         = FloatVector.broadcast(SPECIES, lr);
        var vInvBC1     = FloatVector.broadcast(SPECIES, 1f / biasCorr1);
        var vInvBC2     = FloatVector.broadcast(SPECIES, 1f / biasCorr2);
        var vEps        = FloatVector.broadcast(SPECIES, eps);
        var vDecay      = FloatVector.broadcast(SPECIES, lr * weightDecay);

        int i = 0;
        int bound = SPECIES.loopBound(len);

        for (; i < bound; i += SPECIES.length()) {
            long gB = (gOff + i) * F32_BYTES;
            long mB = (mOff + i) * F32_BYTES;
            long vB = (vOff + i) * F32_BYTES;
            long pB = (pOff + i) * F32_BYTES;

            var vg = FloatVector.fromMemorySegment(SPECIES, g, gB, NATIVE);
            var vm = FloatVector.fromMemorySegment(SPECIES, m, mB, NATIVE);
            var vv = FloatVector.fromMemorySegment(SPECIES, v, vB, NATIVE);
            var vp = FloatVector.fromMemorySegment(SPECIES, p, pB, NATIVE);

            vm = vBeta1.fma(vm, vOneMinusB1.mul(vg));
            vv = vBeta2.fma(vv, vOneMinusB2.mul(vg.mul(vg)));

            var mHat = vm.mul(vInvBC1);
            var vHat = vv.mul(vInvBC2);

            vp = vp.sub(vDecay.mul(vp));
            vp = vp.sub(vLr.mul(mHat.div(vHat.lanewise(SQRT).add(vEps))));

            vm.intoMemorySegment(m, mB, NATIVE);
            vv.intoMemorySegment(v, vB, NATIVE);
            vp.intoMemorySegment(p, pB, NATIVE);
        }

        float invBC1 = 1f / biasCorr1;
        float invBC2 = 1f / biasCorr2;
        float decay = lr * weightDecay;
        for (; i < len; i++) {
            long gB = (gOff + i) * F32_BYTES;
            long mB = (mOff + i) * F32_BYTES;
            long vB = (vOff + i) * F32_BYTES;
            long pB = (pOff + i) * F32_BYTES;
            float gi = g.get(F32, gB);
            float mi = beta1 * m.get(F32, mB) + (1f - beta1) * gi;
            float vi = beta2 * v.get(F32, vB) + (1f - beta2) * gi * gi;
            m.set(F32, mB, mi);
            v.set(F32, vB, vi);

            float mHat = mi * invBC1;
            float vHat = vi * invBC2;
            float pi = p.get(F32, pB);
            pi -= decay * pi;
            pi -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
            p.set(F32, pB, pi);
        }
    }
}
