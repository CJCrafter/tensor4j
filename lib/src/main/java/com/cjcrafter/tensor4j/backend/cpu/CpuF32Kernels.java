package com.cjcrafter.tensor4j.backend.cpu;

import com.cjcrafter.tensor4j.Device;
import com.cjcrafter.tensor4j.Shape;
import com.cjcrafter.tensor4j.backend.F32KernelTable;
import com.cjcrafter.tensor4j.memory.Storage;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static jdk.incubator.vector.VectorOperators.*;

/**
 * F32 CPU kernels backed by plain heap {@code float[]} arrays.
 */
public final class CpuF32Kernels implements F32KernelTable {

    private static final CpuF32Kernels INSTANCE = new CpuF32Kernels();

    /**
     * Returns the shared singleton instance.
     */
    public static CpuF32Kernels instance() {
        return INSTANCE;
    }

    private CpuF32Kernels() {
    }

    private static final int REGISTER_BLOCKS = 8;
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final FloatVector ZERO = FloatVector.zero(SPECIES);
    private static final FloatVector ONE = FloatVector.broadcast(SPECIES, 1f);

    private static float[] arr(Storage s) {
        return s.array();
    }

    @Override
    public Device device() {
        return Device.cpu();
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

    @Override
    public void matmul(Storage aS, int aOff, int m, int k,
                       Storage bS, int bOff, int n,
                       Storage cS, int cOff) {
        float[] a = arr(aS);
        float[] b = arr(bS);
        float[] c = arr(cS);

        float[] bPacked = new float[K_TILE * J_TILE];

        for (int it = 0; it < m; it += I_TILE) {
            int iEnd = Math.min(m, it + I_TILE);

        for (int jt = 0; jt < n; jt += J_TILE) {
            int jEnd = Math.min(n, jt + J_TILE);
            int jBound = SPECIES.loopBound(jEnd - jt) + jt;

        for (int pt = 0; pt < k; pt += K_TILE) {
            int pEnd = Math.min(k, pt + K_TILE);
            int jTileWidth = jEnd - jt;

            for (int p = pt; p < pEnd; p++) {
                System.arraycopy(b, bOff + p * n + jt, bPacked, (p - pt) * jTileWidth, jTileWidth);
            }

            int i = it;
            for (; i <= iEnd - REGISTER_BLOCKS; i += REGISTER_BLOCKS) {
                int aRowOff0 = aOff + i * k;
                int aRowOff1 = aOff + (i + 1) * k;
                int aRowOff2 = aOff + (i + 2) * k;
                int aRowOff3 = aOff + (i + 3) * k;
                int aRowOff4 = aOff + (i + 4) * k;
                int aRowOff5 = aOff + (i + 5) * k;
                int aRowOff6 = aOff + (i + 6) * k;
                int aRowOff7 = aOff + (i + 7) * k;
                int cRowOff0 = cOff + i * n;
                int cRowOff1 = cOff + (i + 1) * n;
                int cRowOff2 = cOff + (i + 2) * n;
                int cRowOff3 = cOff + (i + 3) * n;
                int cRowOff4 = cOff + (i + 4) * n;
                int cRowOff5 = cOff + (i + 5) * n;
                int cRowOff6 = cOff + (i + 6) * n;
                int cRowOff7 = cOff + (i + 7) * n;

                int j = jt;
                for (; j < jBound; j += SPECIES.length()) {
                    var vc0 = FloatVector.fromArray(SPECIES, c, cRowOff0 + j);
                    var vc1 = FloatVector.fromArray(SPECIES, c, cRowOff1 + j);
                    var vc2 = FloatVector.fromArray(SPECIES, c, cRowOff2 + j);
                    var vc3 = FloatVector.fromArray(SPECIES, c, cRowOff3 + j);
                    var vc4 = FloatVector.fromArray(SPECIES, c, cRowOff4 + j);
                    var vc5 = FloatVector.fromArray(SPECIES, c, cRowOff5 + j);
                    var vc6 = FloatVector.fromArray(SPECIES, c, cRowOff6 + j);
                    var vc7 = FloatVector.fromArray(SPECIES, c, cRowOff7 + j);

                    for (int p = pt; p < pEnd; p++) {
                        int bPackedOff = (p - pt) * jTileWidth + (j - jt);
                        var vb = FloatVector.fromArray(SPECIES, bPacked, bPackedOff);
                        vc0 = FloatVector.broadcast(SPECIES, a[aRowOff0 + p]).fma(vb, vc0);
                        vc1 = FloatVector.broadcast(SPECIES, a[aRowOff1 + p]).fma(vb, vc1);
                        vc2 = FloatVector.broadcast(SPECIES, a[aRowOff2 + p]).fma(vb, vc2);
                        vc3 = FloatVector.broadcast(SPECIES, a[aRowOff3 + p]).fma(vb, vc3);
                        vc4 = FloatVector.broadcast(SPECIES, a[aRowOff4 + p]).fma(vb, vc4);
                        vc5 = FloatVector.broadcast(SPECIES, a[aRowOff5 + p]).fma(vb, vc5);
                        vc6 = FloatVector.broadcast(SPECIES, a[aRowOff6 + p]).fma(vb, vc6);
                        vc7 = FloatVector.broadcast(SPECIES, a[aRowOff7 + p]).fma(vb, vc7);
                    }

                    vc0.intoArray(c, cRowOff0 + j);
                    vc1.intoArray(c, cRowOff1 + j);
                    vc2.intoArray(c, cRowOff2 + j);
                    vc3.intoArray(c, cRowOff3 + j);
                    vc4.intoArray(c, cRowOff4 + j);
                    vc5.intoArray(c, cRowOff5 + j);
                    vc6.intoArray(c, cRowOff6 + j);
                    vc7.intoArray(c, cRowOff7 + j);
                }

                if (j < jEnd) {
                    var mask = SPECIES.indexInRange(j - jt, jEnd - jt);
                    var vc0 = FloatVector.fromArray(SPECIES, c, cRowOff0 + j, mask);
                    var vc1 = FloatVector.fromArray(SPECIES, c, cRowOff1 + j, mask);
                    var vc2 = FloatVector.fromArray(SPECIES, c, cRowOff2 + j, mask);
                    var vc3 = FloatVector.fromArray(SPECIES, c, cRowOff3 + j, mask);
                    var vc4 = FloatVector.fromArray(SPECIES, c, cRowOff4 + j, mask);
                    var vc5 = FloatVector.fromArray(SPECIES, c, cRowOff5 + j, mask);
                    var vc6 = FloatVector.fromArray(SPECIES, c, cRowOff6 + j, mask);
                    var vc7 = FloatVector.fromArray(SPECIES, c, cRowOff7 + j, mask);

                    for (int p = pt; p < pEnd; p++) {
                        int bPackedOff = (p - pt) * jTileWidth + (j - jt);
                        var vb = FloatVector.fromArray(SPECIES, bPacked, bPackedOff, mask);
                        vc0 = FloatVector.broadcast(SPECIES, a[aRowOff0 + p]).fma(vb, vc0);
                        vc1 = FloatVector.broadcast(SPECIES, a[aRowOff1 + p]).fma(vb, vc1);
                        vc2 = FloatVector.broadcast(SPECIES, a[aRowOff2 + p]).fma(vb, vc2);
                        vc3 = FloatVector.broadcast(SPECIES, a[aRowOff3 + p]).fma(vb, vc3);
                        vc4 = FloatVector.broadcast(SPECIES, a[aRowOff4 + p]).fma(vb, vc4);
                        vc5 = FloatVector.broadcast(SPECIES, a[aRowOff5 + p]).fma(vb, vc5);
                        vc6 = FloatVector.broadcast(SPECIES, a[aRowOff6 + p]).fma(vb, vc6);
                        vc7 = FloatVector.broadcast(SPECIES, a[aRowOff7 + p]).fma(vb, vc7);
                    }

                    vc0.intoArray(c, cRowOff0 + j, mask);
                    vc1.intoArray(c, cRowOff1 + j, mask);
                    vc2.intoArray(c, cRowOff2 + j, mask);
                    vc3.intoArray(c, cRowOff3 + j, mask);
                    vc4.intoArray(c, cRowOff4 + j, mask);
                    vc5.intoArray(c, cRowOff5 + j, mask);
                    vc6.intoArray(c, cRowOff6 + j, mask);
                    vc7.intoArray(c, cRowOff7 + j, mask);
                }
            }

            for (; i < iEnd; i++) {
                int aRowOff = aOff + i * k;
                int cRowOff = cOff + i * n;
                int j = jt;
                for (; j < jBound; j += SPECIES.length()) {
                    var vc = FloatVector.fromArray(SPECIES, c, cRowOff + j);
                    for (int p = pt; p < pEnd; p++) {
                        int bPackedOff = (p - pt) * jTileWidth + (j - jt);
                        var aVal = FloatVector.broadcast(SPECIES, a[aRowOff + p]);
                        var vb = FloatVector.fromArray(SPECIES, bPacked, bPackedOff);
                        vc = aVal.fma(vb, vc);
                    }
                    vc.intoArray(c, cRowOff + j);
                }
                if (j < jEnd) {
                    var mask = SPECIES.indexInRange(j - jt, jEnd - jt);
                    var vc = FloatVector.fromArray(SPECIES, c, cRowOff + j, mask);
                    for (int p = pt; p < pEnd; p++) {
                        int bPackedOff = (p - pt) * jTileWidth + (j - jt);
                        var aVal = FloatVector.broadcast(SPECIES, a[aRowOff + p]);
                        var vb = FloatVector.fromArray(SPECIES, bPacked, bPackedOff, mask);
                        vc = aVal.fma(vb, vc);
                    }
                    vc.intoArray(c, cRowOff + j, mask);
                }
            }
        }
        }
        }
    }

    // ------------------ //
    // --- REDUCTIONS --- //
    // ------------------ //

    @Override
    public float sum(Storage aS, int aOff, int len) {
        float[] a = arr(aS);
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

        int lane = SPECIES.length();
        for (; i < blockBound; i += step) {
            int base = aOff + i;
            vs0 = vs0.add(FloatVector.fromArray(SPECIES, a, base));
            vs1 = vs1.add(FloatVector.fromArray(SPECIES, a, base + lane));
            vs2 = vs2.add(FloatVector.fromArray(SPECIES, a, base + lane * 2));
            vs3 = vs3.add(FloatVector.fromArray(SPECIES, a, base + lane * 3));
            vs4 = vs4.add(FloatVector.fromArray(SPECIES, a, base + lane * 4));
            vs5 = vs5.add(FloatVector.fromArray(SPECIES, a, base + lane * 5));
            vs6 = vs6.add(FloatVector.fromArray(SPECIES, a, base + lane * 6));
            vs7 = vs7.add(FloatVector.fromArray(SPECIES, a, base + lane * 7));
        }

        var vsum = vs0.add(vs1).add(vs2).add(vs3)
                .add(vs4).add(vs5).add(vs6).add(vs7);

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            vsum = vsum.add(FloatVector.fromArray(SPECIES, a, aOff + i));
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);

        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromArray(SPECIES, a, aOff + i, mask);
            sum += va.reduceLanes(VectorOperators.ADD, mask);
        }
        return sum;
    }

    @Override
    public float mean(Storage aS, int aOff, int len) {
        float[] a = arr(aS);
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

        int lane = SPECIES.length();
        for (; i < blockBound; i += step) {
            int base = aOff + i;
            var va0 = FloatVector.fromArray(SPECIES, a, base);
            var va1 = FloatVector.fromArray(SPECIES, a, base + lane);
            var va2 = FloatVector.fromArray(SPECIES, a, base + lane * 2);
            var va3 = FloatVector.fromArray(SPECIES, a, base + lane * 3);
            var va4 = FloatVector.fromArray(SPECIES, a, base + lane * 4);
            var va5 = FloatVector.fromArray(SPECIES, a, base + lane * 5);
            var va6 = FloatVector.fromArray(SPECIES, a, base + lane * 6);
            var va7 = FloatVector.fromArray(SPECIES, a, base + lane * 7);

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
            var va = FloatVector.fromArray(SPECIES, a, aOff + i);
            var y = va.sub(vcomp);
            var t = vsum.add(y);
            vcomp = t.sub(vsum).sub(y);
            vsum = t;
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float comp = vcomp.reduceLanes(VectorOperators.ADD);

        for (; i < len; i++) {
            float val = a[aOff + i];
            float y = val - comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }

        return (sum - comp) / len;
    }

    @Override
    public void sumDim(Storage aS, int aOff, int[] dims, int dim, Storage outS) {
        float[] a = arr(aS);
        float[] out = arr(outS);
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
                out[outBase + i] = 0f;

            for (int r = 0; r < reduceDim; r++) {
                int aSlice = aOff + aBase + r * inner;

                int i = 0;
                int bound = SPECIES.loopBound(inner);
                for (; i < bound; i += SPECIES.length()) {
                    var va = FloatVector.fromArray(SPECIES, a, aSlice + i);
                    var vo = FloatVector.fromArray(SPECIES, out, outBase + i);
                    vo.add(va).intoArray(out, outBase + i);
                }
                for (; i < inner; i++) {
                    out[outBase + i] += a[aSlice + i];
                }
            }
        }
    }

    @Override
    public void maxDim(Storage aS, int aOff, int[] dims, int dim, Storage outS) {
        float[] a = arr(aS);
        float[] out = arr(outS);
        int ndim = dims.length;

        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dims[i];
        int reduceDim = dims[dim];
        int inner = 1;
        for (int i = dim + 1; i < ndim; i++) inner *= dims[i];

        for (int o = 0; o < outer; o++) {
            int outBase = o * inner;
            int aBase = aOff + o * reduceDim * inner;

            for (int i = 0; i < inner; i++)
                out[outBase + i] = a[aBase + i];

            for (int r = 1; r < reduceDim; r++) {
                int aSlice = aBase + r * inner;
                for (int i = 0; i < inner; i++) {
                    float v = a[aSlice + i];
                    float cur = out[outBase + i];
                    if (v > cur)
                        out[outBase + i] = v;
                }
            }
        }
    }

    @Override
    public void argmaxDim(Storage aS, int aOff, int[] dims, int dim, Storage outS) {
        float[] a = arr(aS);
        float[] out = arr(outS);
        int ndim = dims.length;

        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dims[i];
        int reduceDim = dims[dim];
        int inner = 1;
        for (int i = dim + 1; i < ndim; i++) inner *= dims[i];

        for (int o = 0; o < outer; o++) {
            int outBase = o * inner;
            int aBase = aOff + o * reduceDim * inner;

            for (int i = 0; i < inner; i++) {
                float maxVal = a[aBase + i];
                int maxIdx = 0;
                for (int r = 1; r < reduceDim; r++) {
                    float val = a[aBase + r * inner + i];
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = r;
                    }
                }
                out[outBase + i] = maxIdx;
            }
        }
    }

    // -------------------------- //
    // --- UNARY ELEMENT-WISE --- //
    // -------------------------- //

    @Override
    public void unary(VectorOperators.Unary op,
                      Storage aS, int aOff,
                      Storage outS, int outOff, int len) {
        float[] a = arr(aS);
        float[] out = arr(outS);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int lane = SPECIES.length();

        for (; i < blockBound; i += step) {
            int aBase = aOff + i;
            int oBase = outOff + i;
            var va0 = FloatVector.fromArray(SPECIES, a, aBase);
            var va1 = FloatVector.fromArray(SPECIES, a, aBase + lane);
            var va2 = FloatVector.fromArray(SPECIES, a, aBase + lane * 2);
            var va3 = FloatVector.fromArray(SPECIES, a, aBase + lane * 3);
            var va4 = FloatVector.fromArray(SPECIES, a, aBase + lane * 4);
            var va5 = FloatVector.fromArray(SPECIES, a, aBase + lane * 5);
            var va6 = FloatVector.fromArray(SPECIES, a, aBase + lane * 6);
            var va7 = FloatVector.fromArray(SPECIES, a, aBase + lane * 7);
            va0.lanewise(op).intoArray(out, oBase);
            va1.lanewise(op).intoArray(out, oBase + lane);
            va2.lanewise(op).intoArray(out, oBase + lane * 2);
            va3.lanewise(op).intoArray(out, oBase + lane * 3);
            va4.lanewise(op).intoArray(out, oBase + lane * 4);
            va5.lanewise(op).intoArray(out, oBase + lane * 5);
            va6.lanewise(op).intoArray(out, oBase + lane * 6);
            va7.lanewise(op).intoArray(out, oBase + lane * 7);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, aOff + i);
            va.lanewise(op).intoArray(out, outOff + i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromArray(SPECIES, a, aOff + i, mask);
            va.lanewise(op).intoArray(out, outOff + i, mask);
        }
    }

    // --------------------------- //
    // --- BINARY ELEMENT-WISE --- //
    // --------------------------- //

    @Override
    public void binary(VectorOperators.Binary op,
                       Storage aS, int aOff,
                       Storage bS, int bOff,
                       Storage outS, int outOff, int len) {
        float[] a = arr(aS);
        float[] b = arr(bS);
        float[] out = arr(outS);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int lane = SPECIES.length();

        for (; i < blockBound; i += step) {
            int aBase = aOff + i;
            int bBase = bOff + i;
            int oBase = outOff + i;
            var va0 = FloatVector.fromArray(SPECIES, a, aBase);
            var va1 = FloatVector.fromArray(SPECIES, a, aBase + lane);
            var va2 = FloatVector.fromArray(SPECIES, a, aBase + lane * 2);
            var va3 = FloatVector.fromArray(SPECIES, a, aBase + lane * 3);
            var va4 = FloatVector.fromArray(SPECIES, a, aBase + lane * 4);
            var va5 = FloatVector.fromArray(SPECIES, a, aBase + lane * 5);
            var va6 = FloatVector.fromArray(SPECIES, a, aBase + lane * 6);
            var va7 = FloatVector.fromArray(SPECIES, a, aBase + lane * 7);
            var vb0 = FloatVector.fromArray(SPECIES, b, bBase);
            var vb1 = FloatVector.fromArray(SPECIES, b, bBase + lane);
            var vb2 = FloatVector.fromArray(SPECIES, b, bBase + lane * 2);
            var vb3 = FloatVector.fromArray(SPECIES, b, bBase + lane * 3);
            var vb4 = FloatVector.fromArray(SPECIES, b, bBase + lane * 4);
            var vb5 = FloatVector.fromArray(SPECIES, b, bBase + lane * 5);
            var vb6 = FloatVector.fromArray(SPECIES, b, bBase + lane * 6);
            var vb7 = FloatVector.fromArray(SPECIES, b, bBase + lane * 7);
            va0.lanewise(op, vb0).intoArray(out, oBase);
            va1.lanewise(op, vb1).intoArray(out, oBase + lane);
            va2.lanewise(op, vb2).intoArray(out, oBase + lane * 2);
            va3.lanewise(op, vb3).intoArray(out, oBase + lane * 3);
            va4.lanewise(op, vb4).intoArray(out, oBase + lane * 4);
            va5.lanewise(op, vb5).intoArray(out, oBase + lane * 5);
            va6.lanewise(op, vb6).intoArray(out, oBase + lane * 6);
            va7.lanewise(op, vb7).intoArray(out, oBase + lane * 7);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, aOff + i);
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i);
            va.lanewise(op, vb).intoArray(out, outOff + i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromArray(SPECIES, a, aOff + i, mask);
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i, mask);
            va.lanewise(op, vb).intoArray(out, outOff + i, mask);
        }
    }

    @Override
    public void binaryConst(VectorOperators.Binary op,
                            Storage aS, int aOff, float b,
                            Storage outS, int outOff, int len) {
        float[] a = arr(aS);
        float[] out = arr(outS);
        var vb = FloatVector.broadcast(SPECIES, b);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int lane = SPECIES.length();

        for (; i < blockBound; i += step) {
            int aBase = aOff + i;
            int oBase = outOff + i;
            var va0 = FloatVector.fromArray(SPECIES, a, aBase);
            var va1 = FloatVector.fromArray(SPECIES, a, aBase + lane);
            var va2 = FloatVector.fromArray(SPECIES, a, aBase + lane * 2);
            var va3 = FloatVector.fromArray(SPECIES, a, aBase + lane * 3);
            var va4 = FloatVector.fromArray(SPECIES, a, aBase + lane * 4);
            var va5 = FloatVector.fromArray(SPECIES, a, aBase + lane * 5);
            var va6 = FloatVector.fromArray(SPECIES, a, aBase + lane * 6);
            var va7 = FloatVector.fromArray(SPECIES, a, aBase + lane * 7);
            va0.lanewise(op, vb).intoArray(out, oBase);
            va1.lanewise(op, vb).intoArray(out, oBase + lane);
            va2.lanewise(op, vb).intoArray(out, oBase + lane * 2);
            va3.lanewise(op, vb).intoArray(out, oBase + lane * 3);
            va4.lanewise(op, vb).intoArray(out, oBase + lane * 4);
            va5.lanewise(op, vb).intoArray(out, oBase + lane * 5);
            va6.lanewise(op, vb).intoArray(out, oBase + lane * 6);
            va7.lanewise(op, vb).intoArray(out, oBase + lane * 7);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, aOff + i);
            va.lanewise(op, vb).intoArray(out, outOff + i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromArray(SPECIES, a, aOff + i, mask);
            va.lanewise(op, vb).intoArray(out, outOff + i, mask);
        }
    }

    @Override
    public void rbinaryConst(VectorOperators.Binary op,
                             float a, Storage bS, int bOff,
                             Storage outS, int outOff, int len) {
        float[] b = arr(bS);
        float[] out = arr(outS);
        var va = FloatVector.broadcast(SPECIES, a);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int lane = SPECIES.length();

        for (; i < blockBound; i += step) {
            int bBase = bOff + i;
            int oBase = outOff + i;
            var vb0 = FloatVector.fromArray(SPECIES, b, bBase);
            var vb1 = FloatVector.fromArray(SPECIES, b, bBase + lane);
            var vb2 = FloatVector.fromArray(SPECIES, b, bBase + lane * 2);
            var vb3 = FloatVector.fromArray(SPECIES, b, bBase + lane * 3);
            var vb4 = FloatVector.fromArray(SPECIES, b, bBase + lane * 4);
            var vb5 = FloatVector.fromArray(SPECIES, b, bBase + lane * 5);
            var vb6 = FloatVector.fromArray(SPECIES, b, bBase + lane * 6);
            var vb7 = FloatVector.fromArray(SPECIES, b, bBase + lane * 7);
            va.lanewise(op, vb0).intoArray(out, oBase);
            va.lanewise(op, vb1).intoArray(out, oBase + lane);
            va.lanewise(op, vb2).intoArray(out, oBase + lane * 2);
            va.lanewise(op, vb3).intoArray(out, oBase + lane * 3);
            va.lanewise(op, vb4).intoArray(out, oBase + lane * 4);
            va.lanewise(op, vb5).intoArray(out, oBase + lane * 5);
            va.lanewise(op, vb6).intoArray(out, oBase + lane * 6);
            va.lanewise(op, vb7).intoArray(out, oBase + lane * 7);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i);
            va.lanewise(op, vb).intoArray(out, outOff + i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i, mask);
            va.lanewise(op, vb).intoArray(out, outOff + i, mask);
        }
    }

    @Override
    public void binaryBroadcast(VectorOperators.Binary op,
                                Storage aS, int[] aStrides, int aOff,
                                Storage bS, int[] bStrides, int bOff,
                                Storage outS, Shape resultShape) {
        float[] a = arr(aS);
        float[] b = arr(bS);
        float[] out = arr(outS);
        int ndim = resultShape.dimensions();
        int numel = resultShape.numel();
        int[] indices = new int[ndim];

        for (int flat = 0; flat < numel; flat++) {
            int aIdx = aOff;
            int bIdx = bOff;
            for (int d = 0; d < ndim; d++) {
                aIdx += indices[d] * aStrides[d];
                bIdx += indices[d] * bStrides[d];
            }
            float av = a[aIdx];
            float bv = b[bIdx];
            out[flat] = applyScalarBinary(op, av, bv);

            for (int d = ndim - 1; d >= 0; d--) {
                if (++indices[d] < resultShape.dim(d)) break;
                indices[d] = 0;
            }
        }
    }

    // ------------------------------- //
    // --- COMPARISON ELEMENT-WISE --- //
    // ------------------------------- //

    @Override
    public void compare(VectorOperators.Comparison op,
                        Storage aS, int aOff,
                        Storage bS, int bOff,
                        Storage outS, int outOff, int len) {
        float[] a = arr(aS);
        float[] b = arr(bS);
        float[] out = arr(outS);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            for (int r = 0; r < REGISTER_BLOCKS; r++) {
                int off = i + SPECIES.length() * r;
                var va = FloatVector.fromArray(SPECIES, a, aOff + off);
                var vb = FloatVector.fromArray(SPECIES, b, bOff + off);
                ZERO.blend(ONE, va.compare(op, vb)).intoArray(out, outOff + off);
            }
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, aOff + i);
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i);
            ZERO.blend(ONE, va.compare(op, vb)).intoArray(out, outOff + i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromArray(SPECIES, a, aOff + i, mask);
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i, mask);
            ZERO.blend(ONE, va.compare(op, vb)).intoArray(out, outOff + i, mask);
        }
    }

    @Override
    public void compareConst(VectorOperators.Comparison op,
                             Storage aS, int aOff, float b,
                             Storage outS, int outOff, int len) {
        float[] a = arr(aS);
        float[] out = arr(outS);
        var vb = FloatVector.broadcast(SPECIES, b);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            for (int r = 0; r < REGISTER_BLOCKS; r++) {
                int off = i + SPECIES.length() * r;
                var va = FloatVector.fromArray(SPECIES, a, aOff + off);
                ZERO.blend(ONE, va.compare(op, vb)).intoArray(out, outOff + off);
            }
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, aOff + i);
            ZERO.blend(ONE, va.compare(op, vb)).intoArray(out, outOff + i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromArray(SPECIES, a, aOff + i, mask);
            ZERO.blend(ONE, va.compare(op, vb)).intoArray(out, outOff + i, mask);
        }
    }

    @Override
    public void compareBroadcast(VectorOperators.Comparison op,
                                 Storage aS, int[] aStrides, int aOff,
                                 Storage bS, int[] bStrides, int bOff,
                                 Storage outS, Shape resultShape) {
        float[] a = arr(aS);
        float[] b = arr(bS);
        float[] out = arr(outS);
        int ndim = resultShape.dimensions();
        int numel = resultShape.numel();
        int[] indices = new int[ndim];

        for (int flat = 0; flat < numel; flat++) {
            int aIdx = aOff;
            int bIdx = bOff;
            for (int d = 0; d < ndim; d++) {
                aIdx += indices[d] * aStrides[d];
                bIdx += indices[d] * bStrides[d];
            }
            float av = a[aIdx];
            float bv = b[bIdx];
            out[flat] = applyScalarComparison(op, av, bv);

            for (int d = ndim - 1; d >= 0; d--) {
                if (++indices[d] < resultShape.dim(d)) break;
                indices[d] = 0;
            }
        }
    }

    // ---------------------------- //
    // --- TERNARY ELEMENT-WISE --- //
    // ---------------------------- //

    @Override
    public void ternary(VectorOperators.Ternary op,
                        Storage aS, int aOff,
                        Storage bS, int bOff,
                        Storage cS, int cOff,
                        Storage outS, int outOff, int len) {
        float[] a = arr(aS);
        float[] b = arr(bS);
        float[] c = arr(cS);
        float[] out = arr(outS);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;
        int lane = SPECIES.length();

        for (; i < blockBound; i += step) {
            int aBase = aOff + i;
            int bBase = bOff + i;
            int cBase = cOff + i;
            int oBase = outOff + i;
            for (int r = 0; r < REGISTER_BLOCKS; r++) {
                int off = lane * r;
                var va = FloatVector.fromArray(SPECIES, a, aBase + off);
                var vb = FloatVector.fromArray(SPECIES, b, bBase + off);
                var vc = FloatVector.fromArray(SPECIES, c, cBase + off);
                va.lanewise(op, vb, vc).intoArray(out, oBase + off);
            }
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, aOff + i);
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i);
            var vc = FloatVector.fromArray(SPECIES, c, cOff + i);
            va.lanewise(op, vb, vc).intoArray(out, outOff + i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromArray(SPECIES, a, aOff + i, mask);
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i, mask);
            var vc = FloatVector.fromArray(SPECIES, c, cOff + i, mask);
            va.lanewise(op, vb, vc).intoArray(out, outOff + i, mask);
        }
    }

    // -------------------- //
    // --- FUSED ADAMW  --- //
    // -------------------- //

    @Override
    public void adamw(Storage pS, int pOff,
                      Storage gS, int gOff,
                      Storage mS, int mOff,
                      Storage vS, int vOff,
                      int len,
                      float lr, float beta1, float beta2,
                      float biasCorr1, float biasCorr2,
                      float weightDecay, float eps) {
        float[] p = arr(pS);
        float[] g = arr(gS);
        float[] m = arr(mS);
        float[] v = arr(vS);

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
            int gB = gOff + i;
            int mB = mOff + i;
            int vB = vOff + i;
            int pB = pOff + i;

            var vg = FloatVector.fromArray(SPECIES, g, gB);
            var vm = FloatVector.fromArray(SPECIES, m, mB);
            var vv = FloatVector.fromArray(SPECIES, v, vB);
            var vp = FloatVector.fromArray(SPECIES, p, pB);

            vm = vBeta1.fma(vm, vOneMinusB1.mul(vg));
            vv = vBeta2.fma(vv, vOneMinusB2.mul(vg.mul(vg)));

            var mHat = vm.mul(vInvBC1);
            var vHat = vv.mul(vInvBC2);

            vp = vp.sub(vDecay.mul(vp));
            vp = vp.sub(vLr.mul(mHat.div(vHat.lanewise(SQRT).add(vEps))));

            vm.intoArray(m, mB);
            vv.intoArray(v, vB);
            vp.intoArray(p, pB);
        }

        float invBC1 = 1f / biasCorr1;
        float invBC2 = 1f / biasCorr2;
        float decay = lr * weightDecay;
        for (; i < len; i++) {
            int gB = gOff + i;
            int mB = mOff + i;
            int vB = vOff + i;
            int pB = pOff + i;
            float gi = g[gB];
            float mi = beta1 * m[mB] + (1f - beta1) * gi;
            float vi = beta2 * v[vB] + (1f - beta2) * gi * gi;
            m[mB] = mi;
            v[vB] = vi;

            float mHat = mi * invBC1;
            float vHat = vi * invBC2;
            float pi = p[pB];
            pi -= decay * pi;
            pi -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
            p[pB] = pi;
        }
    }
}
