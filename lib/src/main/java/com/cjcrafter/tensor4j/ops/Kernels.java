package com.cjcrafter.tensor4j.ops;

import com.cjcrafter.tensor4j.Shape;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static jdk.incubator.vector.VectorOperators.*;

/**
 * Raw compute kernels for internal use. This should not be treated as stable
 * API. Instead, prefer using {@link com.cjcrafter.tensor4j.Tensor Tensors} directly.
 */
public final class Kernels {

    /**
     * How many register blocks to have.
     *
     * <p>All modern CPUs are heavily pipelined. Consecutive instructions with
     * data dependencies (e.g. 1 instruction depends on the previous
     * instruction) cannot take advantage of the pipelining. By loading
     * <code>n</code> independent instructions sequentially <b>BEFORE</b>
     * executing the instructions, we can take advantaging of this pipelining
     * and prevent the CPU from loading no-ops to pad.
     *
     * <p>This cannot be configurable, this needs to be a static constant in
     * order for the compiler to optimize.
     *
     * <p><code>2,4,8</code> are all good values. <code>1</code> effectively
     * disables the optimization.
     */
    private static final int REGISTER_BLOCKS = 8;


    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final FloatVector ZERO = FloatVector.zero(SPECIES);
    private static final FloatVector ONE = FloatVector.broadcast(SPECIES, 1f);


    /**
     * Don't let anyone instantiate this class
     */
    private Kernels() {
    }

    // ----------------- //
    // --- Utilities --- //
    // ----------------- //

    private static float applyScalarUnary(VectorOperators.Unary op, float a) {
        if (op == ABS)
            return Math.abs(a);
        else if (op == NEG)
            return -a;
        else if (op == SIN)
            return (float) Math.sin(a);
        else if (op == COS)
            return (float) Math.cos(a);
        else if (op == TAN)
            return (float) Math.tan(a);
        else if (op == ASIN)
            return (float) Math.asin(a);
        else if (op == ACOS)
            return (float) Math.acos(a);
        else if (op == ATAN)
            return (float) Math.atan(a);
        else if (op == EXP)
            return (float) Math.exp(a);
        else if (op == LOG)  // natural log
            return (float) Math.log(a);
        else if (op == LOG10)
            return (float) Math.log10(a);
        else if (op == SQRT)
            return (float) Math.sqrt(a);
        else if (op == CBRT)
            return (float) Math.cbrt(a);
        else if (op == SINH)
            return (float) Math.sinh(a);
        else if (op == COSH)
            return (float) Math.cosh(a);
        else if (op == TANH)
            return (float) Math.tanh(a);
        else if (op == EXPM1)
            return (float) Math.expm1(a);
        else if (op == LOG1P)
            return (float) Math.log1p(a);
        else
            throw new UnsupportedOperationException("Oops! Forgot to add unary op for " + op.name());
    }

    private static float applyScalarBinary(VectorOperators.Binary op, float a, float b) {
        // JIT should optimize this... https://shipilev.net/blog/2015/black-magic-method-dispatch/
        if (op == ADD)
            return a + b;
        else if (op == SUB)
            return a - b;
        else if (op == MUL)
            return a * b;
        else if (op == DIV)
            return a / b;
        else if (op == MIN)
            return Math.min(a, b);
        else if (op == MAX)
            return Math.max(a, b);
        else if (op == ATAN2)
            return (float) Math.atan2(a, b);
        else if (op == POW)
            return (float) Math.pow(a, b);
        else if (op == HYPOT)
            return (float) Math.hypot(a, b);
        else
            throw new UnsupportedOperationException("Oops! Forgot to add binary op for " + op.name());
    }

    private static float applyScalarComparison(VectorOperators.Comparison op, float a, float b) {
        if (op == EQ)
            return a == b ? 1f : 0f;
        else if (op == NE)
            return a != b ? 1f : 0f;
        else if (op == LT)
            return a < b ? 1f : 0f;
        else if (op == LE)
            return a <= b ? 1f : 0f;
        else if (op == GT)
            return a > b ? 1f : 0f;
        else if (op == GE)
            return a >= b ? 1f : 0f;
        else
            throw new UnsupportedOperationException("Oops! Forgot to add comparison op for " + op.name());
    }

    private static float applyScalarTernary(VectorOperators.Ternary op, float a, float b, float c) {
        if (op == FMA)
            return Math.fma(a, b, c);
        else
            throw new UnsupportedOperationException("Oops! Forgot to add ternary op for " + op.name());
    }

    // -------------- //
    // --- MATMUL --- //
    // -------------- //

    /**
     * The "partial sum" tile. Each pass adds {@code K_TILE} terms.
     *
     * <p>So we have a few constraints:
     * <ul>
     *     <li>{@code B = K_TILE * J_TILE * 4 bytes}</li>
     *     <li>{@code C = REGISTER_BLOCKS * J_TILE * 4 bytes}</li>
     *     <li>{@code A = REGISTER_BLOCKS * K_TILE * 4 bytes}</li>
     *     <li>{@code B + C + A <= L1}</li>
     * </ul>
     */
    private static final int K_TILE = 64;

    /**
     * The "column" tile. Instead of computing all n columns of <b>C</b> before
     * moving on, only compute {@code J_TILE} columns at once. This means the
     * active slice of <b>B</b> is {@code K_TILE * J_TILE <= L1}.
     */
    private static final int J_TILE = 128;

    /**
     * The "row" tile. Instead of computing all m rows of <b>A</b> before
     * moving on, only compute {@code I_TILE} rows. This number should be small
     * enough to fit into the L2 cache... should never hit, but important to
     * keep that limit for massive matrices.
     */
    private static final int I_TILE = 256;

    public static void matmul(float[] a, int aOff, int m, int k,
                              float[] b, int bOff, int n,
                              float[] c, int cOff) {
        // https://siboehm.com/articles/22/Fast-MMM-on-CPU

        // pack B tile into contiguous memory so the prefetcher sees a
        // single sequential stream instead of n-strided scattered rows
        float[] bPacked = new float[K_TILE * J_TILE];

        // which rows of C/A are we working on?
        for (int it = 0; it < m; it += I_TILE) {
            int iEnd = Math.min(m, it + I_TILE);

        // which cols of C/A are we working on?
        for (int jt = 0; jt < n; jt += J_TILE) {
            int jEnd = Math.min(n, jt + J_TILE);
            int jBound = SPECIES.loopBound(jEnd - jt) + jt;

        // which slice of CA are we summing?
        for (int pt = 0; pt < k; pt += K_TILE) {
            int pEnd = Math.min(k, pt + K_TILE);
            int jTileWidth = jEnd - jt;

            // copy the active B tile into a contiguous buffer... this is worth it
            // since B is "strided wrong"... TODO: see if A @ B^T deserves its own path...
            for (int p = pt; p < pEnd; p++) {
                System.arraycopy(b, bOff + p * n + jt, bPacked, (p - pt) * jTileWidth, jTileWidth);
            }

            // any decent cpu should be able to fit at least 8 whole vectors
            // into their registry
            int i = it;
            for (; i <= iEnd - REGISTER_BLOCKS; i += REGISTER_BLOCKS) {
                // By loading these 8 "instructions" sequentially, we have
                // fewer data dependencies in our cpu pipeline. So these
                // 8 operations are much more likely to be run one after the
                // other instead of having no-ops loaded into the cpu.
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
                    // Since we are accumulating values, we cannot start at 0
                    // anymore... load whatever we have already calculated partially
                    var vc0 = FloatVector.fromArray(SPECIES, c, cRowOff0 + j);
                    var vc1 = FloatVector.fromArray(SPECIES, c, cRowOff1 + j);
                    var vc2 = FloatVector.fromArray(SPECIES, c, cRowOff2 + j);
                    var vc3 = FloatVector.fromArray(SPECIES, c, cRowOff3 + j);
                    var vc4 = FloatVector.fromArray(SPECIES, c, cRowOff4 + j);
                    var vc5 = FloatVector.fromArray(SPECIES, c, cRowOff5 + j);
                    var vc6 = FloatVector.fromArray(SPECIES, c, cRowOff6 + j);
                    var vc7 = FloatVector.fromArray(SPECIES, c, cRowOff7 + j);

                    // By loading our "c" vectors first, we can do many
                    // iterations without having to reload the c values.
                    for (int p = pt; p < pEnd; p++) {
                        var vb = FloatVector.fromArray(SPECIES, bPacked, (p - pt) * jTileWidth + (j - jt));
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

                // masked vector tail for this column tile
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
                        var vb = FloatVector.fromArray(SPECIES, bPacked, (p - pt) * jTileWidth + (j - jt), mask);
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

            // Since we are doing register blocking, we might end up with some
            // number NOT divisible by REGISTER_BLOCKS...
            for (; i < iEnd; i++) {
                int aRowOff = aOff + i * k;
                int cRowOff = cOff + i * n;
                int j = jt;
                for (; j < jBound; j += SPECIES.length()) {
                    var vc = FloatVector.fromArray(SPECIES, c, cRowOff + j);
                    for (int p = pt; p < pEnd; p++) {
                        var aVal = FloatVector.broadcast(SPECIES, a[aRowOff + p]);
                        var vb = FloatVector.fromArray(SPECIES, bPacked, (p - pt) * jTileWidth + (j - jt));
                        vc = aVal.fma(vb, vc);
                    }
                    vc.intoArray(c, cRowOff + j);
                }
                if (j < jEnd) {
                    var mask = SPECIES.indexInRange(j - jt, jEnd - jt);
                    var vc = FloatVector.fromArray(SPECIES, c, cRowOff + j, mask);
                    for (int p = pt; p < pEnd; p++) {
                        var aVal = FloatVector.broadcast(SPECIES, a[aRowOff + p]);
                        var vb = FloatVector.fromArray(SPECIES, bPacked, (p - pt) * jTileWidth + (j - jt), mask);
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

    public static float sum(float[] a, int aOff, int len) {
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

        for (; i < blockBound; i += step) {
            vs0 = vs0.add(FloatVector.fromArray(SPECIES, a, aOff + i));
            vs1 = vs1.add(FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length()));
            vs2 = vs2.add(FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 2));
            vs3 = vs3.add(FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 3));
            vs4 = vs4.add(FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 4));
            vs5 = vs5.add(FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 5));
            vs6 = vs6.add(FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 6));
            vs7 = vs7.add(FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 7));
        }

        // merge the 8 accumulators, then do a single horizontal reduce
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

    public static float mean(float[] a, int aOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        // each accumulator gets its own compensation vector
        var vs0 = ZERO; var vc0 = ZERO;
        var vs1 = ZERO; var vc1 = ZERO;
        var vs2 = ZERO; var vc2 = ZERO;
        var vs3 = ZERO; var vc3 = ZERO;
        var vs4 = ZERO; var vc4 = ZERO;
        var vs5 = ZERO; var vc5 = ZERO;
        var vs6 = ZERO; var vc6 = ZERO;
        var vs7 = ZERO; var vc7 = ZERO;

        for (; i < blockBound; i += step) {
            var va0 = FloatVector.fromArray(SPECIES, a, aOff + i);
            var va1 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length());
            var va2 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 2);
            var va3 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 3);
            var va4 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 4);
            var va5 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 5);
            var va6 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 6);
            var va7 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 7);

            // Kahan step for each block
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

            vs0 = t0;
            vs1 = t1;
            vs2 = t2;
            vs3 = t3;
            vs4 = t4;
            vs5 = t5;
            vs6 = t6;
            vs7 = t7;
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

        // scalar tail
        for (; i < len; i++) {
            float y = a[aOff + i] - comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }

        return (sum - comp) / len;
    }

    public static void sumDim(float[] a, int aOff, int[] dims, int dim, float[] out) {
        int ndim = dims.length;

        // outer = product of dims before `dim`, inner = product of dims after `dim`
        int outer = 1;
        for (int i = 0; i < dim; i++)
            outer *= dims[i];
        int reduceDim = dims[dim];
        int inner = 1;
        for (int i = dim + 1; i < ndim; i++)
            inner *= dims[i];

        // For each (outer, inner) pair, sum across the reduce dimension.
        // Memory layout: a[o * reduceDim * inner + r * inner + i]
        for (int o = 0; o < outer; o++) {
            int outBase = o * inner;
            int aBase = o * reduceDim * inner;

            // zero the output slice
            for (int i = 0; i < inner; i++)
                out[outBase + i] = 0f;

            for (int r = 0; r < reduceDim; r++) {
                int aSlice = aOff + aBase + r * inner;

                // SIMD for the inner loop
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

    public static void maxDim(float[] a, int aOff, int[] dims, int dim, float[] out) {
        int ndim = dims.length;

        int outer = 1;
        for (int i = 0; i < dim; i++)
            outer *= dims[i];
        int reduceDim = dims[dim];
        int inner = 1;
        for (int i = dim + 1; i < ndim; i++)
            inner *= dims[i];

        for (int o = 0; o < outer; o++) {
            int outBase = o * inner;
            int aBase = aOff + o * reduceDim * inner;

            // seed with the r=0 slice
            for (int i = 0; i < inner; i++) {
                out[outBase + i] = a[aBase + i];
            }
            // fold in the remaining slices with elementwise max
            for (int r = 1; r < reduceDim; r++) {
                int aSlice = aBase + r * inner;
                for (int i = 0; i < inner; i++) {
                    float v = a[aSlice + i];
                    if (v > out[outBase + i]) {
                        out[outBase + i] = v;
                    }
                }
            }
        }
    }

    public static void argmaxDim(float[] a, int aOff, int[] dims, int dim, float[] out) {
        int ndim = dims.length;

        int outer = 1;
        for (int i = 0; i < dim; i++)
            outer *= dims[i];
        int reduceDim = dims[dim];
        int inner = 1;
        for (int i = dim + 1; i < ndim; i++)
            inner *= dims[i];

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

    public static void unaryContiguous(VectorOperators.Unary op,
                                        float[] a, int aOff,
                                        float[] out, int outOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            var va0 = FloatVector.fromArray(SPECIES, a, aOff + i);
            var va1 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length());
            var va2 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 2);
            var va3 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 3);
            var va4 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 4);
            var va5 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 5);
            var va6 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 6);
            var va7 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 7);
            va0.lanewise(op).intoArray(out, outOff + i);
            va1.lanewise(op).intoArray(out, outOff + i + SPECIES.length());
            va2.lanewise(op).intoArray(out, outOff + i + SPECIES.length() * 2);
            va3.lanewise(op).intoArray(out, outOff + i + SPECIES.length() * 3);
            va4.lanewise(op).intoArray(out, outOff + i + SPECIES.length() * 4);
            va5.lanewise(op).intoArray(out, outOff + i + SPECIES.length() * 5);
            va6.lanewise(op).intoArray(out, outOff + i + SPECIES.length() * 6);
            va7.lanewise(op).intoArray(out, outOff + i + SPECIES.length() * 7);
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

    public static void binaryContiguous(VectorOperators.Binary op,
                                        float[] a, int aOff,
                                        float[] b, int bOff,
                                        float[] out, int outOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            var va0 = FloatVector.fromArray(SPECIES, a, aOff + i);
            var va1 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length());
            var va2 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 2);
            var va3 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 3);
            var va4 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 4);
            var va5 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 5);
            var va6 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 6);
            var va7 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 7);
            var vb0 = FloatVector.fromArray(SPECIES, b, bOff + i);
            var vb1 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length());
            var vb2 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 2);
            var vb3 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 3);
            var vb4 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 4);
            var vb5 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 5);
            var vb6 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 6);
            var vb7 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 7);
            va0.lanewise(op, vb0).intoArray(out, outOff + i);
            va1.lanewise(op, vb1).intoArray(out, outOff + i + SPECIES.length());
            va2.lanewise(op, vb2).intoArray(out, outOff + i + SPECIES.length() * 2);
            va3.lanewise(op, vb3).intoArray(out, outOff + i + SPECIES.length() * 3);
            va4.lanewise(op, vb4).intoArray(out, outOff + i + SPECIES.length() * 4);
            va5.lanewise(op, vb5).intoArray(out, outOff + i + SPECIES.length() * 5);
            va6.lanewise(op, vb6).intoArray(out, outOff + i + SPECIES.length() * 6);
            va7.lanewise(op, vb7).intoArray(out, outOff + i + SPECIES.length() * 7);
        }

        // remaining full vectors
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

    public static void binaryConst(VectorOperators.Binary op,
                                   float[] a, int aOff,
                                   float b,
                                   float[] out, int len) {
        var vb = FloatVector.broadcast(SPECIES, b);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            var va0 = FloatVector.fromArray(SPECIES, a, aOff + i);
            var va1 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length());
            var va2 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 2);
            var va3 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 3);
            var va4 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 4);
            var va5 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 5);
            var va6 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 6);
            var va7 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 7);
            va0.lanewise(op, vb).intoArray(out, i);
            va1.lanewise(op, vb).intoArray(out, i + SPECIES.length());
            va2.lanewise(op, vb).intoArray(out, i + SPECIES.length() * 2);
            va3.lanewise(op, vb).intoArray(out, i + SPECIES.length() * 3);
            va4.lanewise(op, vb).intoArray(out, i + SPECIES.length() * 4);
            va5.lanewise(op, vb).intoArray(out, i + SPECIES.length() * 5);
            va6.lanewise(op, vb).intoArray(out, i + SPECIES.length() * 6);
            va7.lanewise(op, vb).intoArray(out, i + SPECIES.length() * 7);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, aOff + i);
            va.lanewise(op, vb).intoArray(out, i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromArray(SPECIES, a, aOff + i, mask);
            va.lanewise(op, vb).intoArray(out, i, mask);
        }
    }

    public static void rbinaryConst(VectorOperators.Binary op,
                                    float a,
                                    float[] b, int bOff,
                                    float[] out, int len) {
        // Same as binaryConst, but for non-associative binary ops so we can
        // swap the order... e.g. tensor-1 and 1-tensor.
        var va = FloatVector.broadcast(SPECIES, a);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            var vb0 = FloatVector.fromArray(SPECIES, b, bOff + i);
            var vb1 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length());
            var vb2 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 2);
            var vb3 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 3);
            var vb4 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 4);
            var vb5 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 5);
            var vb6 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 6);
            var vb7 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 7);
            va.lanewise(op, vb0).intoArray(out, i);
            va.lanewise(op, vb1).intoArray(out, i + SPECIES.length());
            va.lanewise(op, vb2).intoArray(out, i + SPECIES.length() * 2);
            va.lanewise(op, vb3).intoArray(out, i + SPECIES.length() * 3);
            va.lanewise(op, vb4).intoArray(out, i + SPECIES.length() * 4);
            va.lanewise(op, vb5).intoArray(out, i + SPECIES.length() * 5);
            va.lanewise(op, vb6).intoArray(out, i + SPECIES.length() * 6);
            va.lanewise(op, vb7).intoArray(out, i + SPECIES.length() * 7);
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i);
            va.lanewise(op, vb).intoArray(out, i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var vb = FloatVector.fromArray(SPECIES, b, bOff + i, mask);
            va.lanewise(op, vb).intoArray(out, i, mask);
        }
    }

    public static void binaryContiguous(VectorOperators.Binary op,
                                        float[] a, int aOff,
                                        float[] b, int bOff,
                                        float[] out, int len) {
        binaryContiguous(op, a, aOff, b, bOff, out, 0, len);
    }

    public static void binaryBroadcast(VectorOperators.Binary op,
                                       float[] a, int[] aStrides, int aOff,
                                       float[] b, int[] bStrides, int bOff,
                                       float[] out, Shape resultShape) {
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
            out[flat] = applyScalarBinary(op, a[aIdx], b[bIdx]);

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
                                         float[] a, int aOff,
                                         float[] b, int bOff,
                                         float[] out, int outOff, int len) {
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

    public static void compareContiguous(VectorOperators.Comparison op,
                                         float[] a, int aOff,
                                         float[] b, int bOff,
                                         float[] out, int len) {
        compareContiguous(op, a, aOff, b, bOff, out, 0, len);
    }

    public static void compareConst(VectorOperators.Comparison op,
                                    float[] a, int aOff,
                                    float b,
                                    float[] out, int len) {
        var vb = FloatVector.broadcast(SPECIES, b);
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            for (int r = 0; r < REGISTER_BLOCKS; r++) {
                int off = i + SPECIES.length() * r;
                var va = FloatVector.fromArray(SPECIES, a, aOff + off);
                ZERO.blend(ONE, va.compare(op, vb)).intoArray(out, off);
            }
        }

        int bound = SPECIES.loopBound(len);
        for (; i < bound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, aOff + i);
            ZERO.blend(ONE, va.compare(op, vb)).intoArray(out, i);
        }
        if (i < len) {
            var mask = SPECIES.indexInRange(i, len);
            var va = FloatVector.fromArray(SPECIES, a, aOff + i, mask);
            ZERO.blend(ONE, va.compare(op, vb)).intoArray(out, i, mask);
        }
    }

    public static void compareBroadcast(VectorOperators.Comparison op,
                                        float[] a, int[] aStrides, int aOff,
                                        float[] b, int[] bStrides, int bOff,
                                        float[] out, Shape resultShape) {
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
            out[flat] = applyScalarComparison(op, a[aIdx], b[bIdx]);

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
                                         float[] a, int aOff,
                                         float[] b, int bOff,
                                         float[] c, int cOff,
                                         float[] out, int outOff, int len) {
        int i = 0;
        int step = SPECIES.length() * REGISTER_BLOCKS;
        int blockBound = (len / step) * step;

        for (; i < blockBound; i += step) {
            var va0 = FloatVector.fromArray(SPECIES, a, aOff + i);
            var va1 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length());
            var va2 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 2);
            var va3 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 3);
            var va4 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 4);
            var va5 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 5);
            var va6 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 6);
            var va7 = FloatVector.fromArray(SPECIES, a, aOff + i + SPECIES.length() * 7);
            var vb0 = FloatVector.fromArray(SPECIES, b, bOff + i);
            var vb1 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length());
            var vb2 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 2);
            var vb3 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 3);
            var vb4 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 4);
            var vb5 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 5);
            var vb6 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 6);
            var vb7 = FloatVector.fromArray(SPECIES, b, bOff + i + SPECIES.length() * 7);
            var vc0 = FloatVector.fromArray(SPECIES, c, cOff + i);
            var vc1 = FloatVector.fromArray(SPECIES, c, cOff + i + SPECIES.length());
            var vc2 = FloatVector.fromArray(SPECIES, c, cOff + i + SPECIES.length() * 2);
            var vc3 = FloatVector.fromArray(SPECIES, c, cOff + i + SPECIES.length() * 3);
            var vc4 = FloatVector.fromArray(SPECIES, c, cOff + i + SPECIES.length() * 4);
            var vc5 = FloatVector.fromArray(SPECIES, c, cOff + i + SPECIES.length() * 5);
            var vc6 = FloatVector.fromArray(SPECIES, c, cOff + i + SPECIES.length() * 6);
            var vc7 = FloatVector.fromArray(SPECIES, c, cOff + i + SPECIES.length() * 7);
            va0.lanewise(op, vb0, vc0).intoArray(out, outOff + i);
            va1.lanewise(op, vb1, vc1).intoArray(out, outOff + i + SPECIES.length());
            va2.lanewise(op, vb2, vc2).intoArray(out, outOff + i + SPECIES.length() * 2);
            va3.lanewise(op, vb3, vc3).intoArray(out, outOff + i + SPECIES.length() * 3);
            va4.lanewise(op, vb4, vc4).intoArray(out, outOff + i + SPECIES.length() * 4);
            va5.lanewise(op, vb5, vc5).intoArray(out, outOff + i + SPECIES.length() * 5);
            va6.lanewise(op, vb6, vc6).intoArray(out, outOff + i + SPECIES.length() * 6);
            va7.lanewise(op, vb7, vc7).intoArray(out, outOff + i + SPECIES.length() * 7);
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

    public static void adamw(float[] p, int pOff,
                             float[] g, int gOff,
                             float[] m, int mOff,
                             float[] v, int vOff,
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
            var vg = FloatVector.fromArray(SPECIES, g, gOff + i);
            var vm = FloatVector.fromArray(SPECIES, m, mOff + i);
            var vv = FloatVector.fromArray(SPECIES, v, vOff + i);
            var vp = FloatVector.fromArray(SPECIES, p, pOff + i);

            vm = vBeta1.fma(vm, vOneMinusB1.mul(vg));
            vv = vBeta2.fma(vv, vOneMinusB2.mul(vg.mul(vg)));

            var mHat = vm.mul(vInvBC1);
            var vHat = vv.mul(vInvBC2);

            vp = vp.sub(vDecay.mul(vp));
            vp = vp.sub(vLr.mul(mHat.div(vHat.lanewise(SQRT).add(vEps))));

            vm.intoArray(m, mOff + i);
            vv.intoArray(v, vOff + i);
            vp.intoArray(p, pOff + i);
        }

        float invBC1 = 1f / biasCorr1;
        float invBC2 = 1f / biasCorr2;
        float decay = lr * weightDecay;
        for (; i < len; i++) {
            float gi = g[gOff + i];
            float mi = beta1 * m[mOff + i] + (1f - beta1) * gi;
            float vi = beta2 * v[vOff + i] + (1f - beta2) * gi * gi;
            m[mOff + i] = mi;
            v[vOff + i] = vi;

            float mHat = mi * invBC1;
            float vHat = vi * invBC2;
            p[pOff + i] -= decay * p[pOff + i];
            p[pOff + i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
        }
    }
}
