package com.cjcrafter.tensor4j;

import com.cjcrafter.tensor4j.autograd.*;
import com.cjcrafter.tensor4j.memory.NativeStorage;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;
import com.google.common.base.Preconditions;
import org.jetbrains.annotations.ApiStatus;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;

import static jdk.incubator.vector.VectorOperators.*;

/**
 * A multi-dimensional float32 matrix backed by a refcounted {@link Storage}.
 *
 * <p>Most ops are out-of-place; {@link #transpose()}, {@link #view(int...)},
 * and {@link #detach()} share the underlying storage.
 */
public class Tensor implements Cloneable {

    private final Storage storage;
    private final Shape shape;
    private final int offset;

    private final boolean requiresGrad;
    private AutogradMeta meta; // lazily allocated; null when no grad has been touched

    Tensor(Storage storage, Shape shape, int offset, boolean requiresGrad) {
        Preconditions.checkNotNull(storage, "storage");
        Preconditions.checkNotNull(shape, "shape");
        this.storage = storage;
        this.shape = shape;
        this.offset = offset;
        this.requiresGrad = requiresGrad;

        // Increment the number of pointers to the storage, and decrement the
        // pointer if it gets Garbage Collected
        storage.incRef();
        Tensor4j.onClean(this, storage::decRef);
    }

    private static Storage alloc(int numel) {
        return NativeStorage.allocate(DType.F32, numel);
    }

    // ---- accessors ----

    /**
     * Wrap an op result in a {@link Tensor} and attach {@code fn} as its gradFn
     * iff any input requires grad and grad tracking is enabled. The new tensor
     * is at offset 0.
     */
    public static Tensor wrapResult(Storage storage, Shape shape, TensorFunction fn) {
        return wrapResult(storage, shape, 0, fn);
    }

    /**
     * Variant of {@link #wrapResult(Storage, Shape, TensorFunction)} that
     * allows a non-zero offset (for view-like ops that share storage).
     */
    public static Tensor wrapResult(Storage storage, Shape shape, int offset, TensorFunction fn) {
        boolean grad = false;
        for (Tensor input : fn.getInputs()) {
            if (input.requiresGrad) {
                grad = true;
                break;
            }
        }
        Tensor result = new Tensor(storage, shape, offset, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.setGradFn(fn);
        }
        return result;
    }

    /**
     * Like {@link #wrapResult(Storage, Shape, TensorFunction)} but defers
     * Function construction until after the result tensor exists, passing the
     * detached result back to the factory. Used by ops that cache their own
     * output for the backward pass (e.g. exp, sqrt, tanh).
     */
    public static Tensor wrapResultDeferred(Storage storage, Shape shape,
                                            Tensor[] inputs,
                                            java.util.function.Function<Tensor, TensorFunction> fnFactory) {
        boolean grad = false;
        for (Tensor input : inputs) {
            if (input.requiresGrad) {
                grad = true;
                break;
            }
        }
        Tensor result = new Tensor(storage, shape, 0, grad);
        if (grad && Tensor4j.isGradEnabled()) {
            result.setGradFn(fnFactory.apply(result.detach()));
        }
        return result;
    }

    private static void buildTopo(Tensor node, Set<Tensor> visited, List<Tensor> out) {
        if (!visited.add(node)) return;
        TensorFunction fn = node.meta == null ? null : node.meta.gradFn;
        if (fn != null) {
            for (Tensor input : fn.getInputs()) {
                buildTopo(input, visited, out);
            }
        }
        out.add(node);
    }

    public static Tensor cat(int dim, Tensor... tensors) {
        Preconditions.checkArgument(tensors != null && tensors.length > 0,
                "cat requires at least one tensor");

        Shape first = tensors[0].shape;
        int ndim = first.dimensions();
        Preconditions.checkArgument(dim >= 0 && dim < ndim,
                "dim %s out of range for %sD tensor", dim, ndim);

        int[] outDims = first.dims();
        int[] sizesAlongDim = new int[tensors.length];
        sizesAlongDim[0] = first.dim(dim);
        int catTotal = first.dim(dim);

        for (int i = 1; i < tensors.length; i++) {
            Shape s = tensors[i].shape;
            Preconditions.checkArgument(s.dimensions() == ndim,
                    "cat: mismatched ndim at input %s", i);
            for (int d = 0; d < ndim; d++) {
                if (d == dim) continue;
                Preconditions.checkArgument(s.dim(d) == first.dim(d),
                        "cat: shapes differ at dim %s: %s vs %s", d, first, s);
            }
            sizesAlongDim[i] = s.dim(dim);
            catTotal += s.dim(dim);
        }
        outDims[dim] = catTotal;
        Shape outShape = new Shape(outDims);

        int outer = 1;
        for (int d = 0; d < dim; d++) outer *= first.dim(d);
        int inner = 1;
        for (int d = dim + 1; d < ndim; d++) inner *= first.dim(d);
        int outDimStride = catTotal * inner;

        Storage out = alloc(outShape.numel());
        MemorySegment outSeg = out.segment();

        int cursor = 0;
        for (int k = 0; k < tensors.length; k++) {
            Tensor t = tensors[k].isContiguous() ? tensors[k] : tensors[k].contiguous();
            int sizeK = sizesAlongDim[k];
            int inDimStride = sizeK * inner;
            MemorySegment tSeg = t.storage.segment();
            int tOff = t.offset;
            for (int o = 0; o < outer; o++) {
                int outBase = o * outDimStride + cursor * inner;
                int inBase = tOff + o * inDimStride;
                MemorySegment.copy(tSeg, (long) inBase * Float.BYTES,
                        outSeg, (long) outBase * Float.BYTES,
                        (long) sizeK * inner * Float.BYTES);
            }
            cursor += sizeK;
        }

        return wrapResult(out, outShape, new CatFn(dim, sizesAlongDim, tensors));
    }

    private AutogradMeta meta() {
        if (meta == null)
            meta = new AutogradMeta();
        return meta;
    }

    public Storage getStorage() {
        return storage;
    }

    public DType dtype() {
        return storage.dtype();
    }

    public Device device() {
        return storage.device();
    }

    /**
     * Returns a fresh {@code float[]} containing this tensor's logical
     * elements in row-major order. Always copies; mutations to the returned
     * array do not affect the tensor.
     */
    public float[] toFloatArray() {
        Tensor src = isContiguous() ? this : contiguous();
        int n = shape.numel();
        float[] out = new float[n];
        MemorySegment.copy(src.storage.segment(), ValueLayout.JAVA_FLOAT,
                (long) src.offset * Float.BYTES, out, 0, n);
        return out;
    }

    private float read(int idx) {
        return storage.segment().get(ValueLayout.JAVA_FLOAT, (long) idx * Float.BYTES);
    }

    private void write(int idx, float val) {
        storage.segment().set(ValueLayout.JAVA_FLOAT, (long) idx * Float.BYTES, val);
    }

    public Shape getShape() {
        return shape;
    }

    public int getOffset() {
        return offset;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public Tensor getGrad() {
        return meta == null ? null : meta.grad;
    }

    public TensorFunction getGradFn() {
        return meta == null ? null : meta.gradFn;
    }

    void setGradFn(TensorFunction fn) {
        meta().gradFn = fn;
    }

    public void zeroGrad() {
        if (meta != null)
            meta.grad = null;
    }

    private void checkInPlaceSafe() {
        if (this.requiresGrad && meta != null && meta.gradFn != null)
            throw new IllegalStateException(
                    "in-place operation on a tensor that is part of the computation graph");
        if (offset != 0 || storage.capacity() != shape.numel() || !isContiguous())
            throw new UnsupportedOperationException(
                    "in-place operation not supported on views or non-contiguous tensors");
    }

    public void backward() {
        if (shape.numel() != 1)
            throw new IllegalStateException("backward() only supported on scalar tensors, got " + shape);

        // seed with dL/dL = 1
        Storage seedStorage = alloc(1);
        seedStorage.segment().set(ValueLayout.JAVA_FLOAT, 0L, 1f);
        meta().grad = new Tensor(seedStorage, new Shape(1, 1), 0, false);

        // build reverse-topological order of the forward graph so gradient is
        // fully accumulated from all paths before propagating further back.
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        buildTopo(this, visited, topo);

        for (int idx = topo.size() - 1; idx >= 0; idx--) {
            Tensor node = topo.get(idx);
            AutogradMeta nm = node.meta;
            if (nm == null || nm.gradFn == null) continue;
            if (nm.grad == null) continue; // unreachable from loss

            Tensor[] grads = nm.gradFn.backward(nm.grad);
            Tensor[] inputs = nm.gradFn.getInputs();
            for (int i = 0; i < inputs.length; i++) {
                Tensor input = inputs[i];
                if (!input.requiresGrad) continue;

                AutogradMeta im = input.meta();
                if (im.grad == null) {
                    // This is a little bit dangerous since the grad function
                    // is responsible for cloning these tensors.
                    im.grad = grads[i];
                } else {
                    // accumulate in-place, bypassing checkInPlaceSafe since
                    // grad tensors are never part of the forward graph
                    Ops.binaryOp(ADD, im.grad, grads[i], im.grad.shape, im.grad.storage);
                }
            }
        }
    }

    public float item() {
        if (shape.numel() != 1)
            throw new IllegalStateException("item() only valid on single-element tensors, got shape " + shape);
        return read(offset);
    }

    public float get(int... indices) {
        return read(offset + shape.flatIndex(indices));
    }

    public void set(float val, int... indices) {
        write(offset + shape.flatIndex(indices), val);
    }

    // ----------------- //
    // --- VIEW OPS  --- //
    // ----------------- //

    public boolean isContiguous() {
        return shape.isContiguous();
    }

    public Tensor transpose() {
        return TransposeFn.apply(this);
    }

    /**
     * Reinterprets this tensor's dims in reverse order; not tracked by autograd.
     */
    @ApiStatus.Internal
    public Tensor reverseDims() {
        return new Tensor(storage, shape.reverseDims(), offset, false);
    }

    public Tensor view(int... newDims) {
        return ViewFn.apply(this, newDims);
    }

    public Tensor contiguous() {
        if (isContiguous())
            return this;

        int n = shape.numel();
        Storage dst = alloc(n);
        MemorySegment dstSeg = dst.segment();
        MemorySegment srcSeg = storage.segment();
        int[] indices = new int[shape.dimensions()];

        for (int flat = 0; flat < n; flat++) {
            float v = srcSeg.get(ValueLayout.JAVA_FLOAT, (long) (offset + shape.flatIndex(indices)) * Float.BYTES);
            dstSeg.set(ValueLayout.JAVA_FLOAT, (long) flat * Float.BYTES, v);

            for (int d = indices.length - 1; d >= 0; d--) {
                if (++indices[d] < shape.dim(d)) break;
                indices[d] = 0;
            }
        }

        return wrapResult(dst, new Shape(shape.dims()), new IdentityFn(this));
    }

    @Override
    public Tensor clone() {
        int capacity = (int) storage.capacity();
        Storage dst = alloc(capacity);
        MemorySegment.copy(storage.segment(), 0L, dst.segment(), 0L,
                (long) capacity * Float.BYTES);
        return wrapResult(dst, shape, new IdentityFn(this));
    }

    public Tensor detach() {
        return new Tensor(storage, shape, offset, false);
    }

    public Tensor indexSelect(int dim, int... indices) {
        if (!isContiguous())
            return contiguous().indexSelect(dim, indices);

        int ndim = shape.dimensions();
        Preconditions.checkArgument(dim >= 0 && dim < ndim,
                "dim %s out of range for %sD tensor", dim, ndim);

        int[] newDims = shape.dims();
        newDims[dim] = indices.length;

        // size of one slice along dim
        int sliceSize = 1;
        for (int d = dim + 1; d < ndim; d++)
            sliceSize *= shape.dim(d);

        // stride to jump between slices along dim
        int dimStride = shape.dim(dim) * sliceSize;

        // number of outer repetitions (product of dims before dim)
        int outerSize = 1;
        for (int d = 0; d < dim; d++)
            outerSize *= shape.dim(d);

        int outNumel = outerSize * indices.length * sliceSize;
        Storage out = alloc(outNumel);
        MemorySegment outSeg = out.segment();
        MemorySegment srcSeg = storage.segment();
        int outPos = 0;

        for (int outer = 0; outer < outerSize; outer++) {
            int outerOff = offset + outer * dimStride;
            for (int idx : indices) {
                MemorySegment.copy(srcSeg, (long) (outerOff + idx * sliceSize) * Float.BYTES,
                        outSeg, (long) outPos * Float.BYTES,
                        (long) sliceSize * Float.BYTES);
                outPos += sliceSize;
            }
        }

        return wrapResult(out, new Shape(newDims), new IndexSelectFn(this, dim, indices));
    }

    // -------------- //
    // --- MATMUL --- //
    // -------------- //

    public Tensor matmul(Tensor other) {
        return MatmulFn.apply(this, other);
    }

    // ------------------ //
    // --- REDUCTIONS --- //
    // ------------------ //

    public Tensor sum() {
        float val = Ops.sum(this);
        Storage out = alloc(1);
        out.segment().set(ValueLayout.JAVA_FLOAT, 0L, val);
        return wrapResult(out, new Shape(1, 1), new SumFn(this));
    }

    public Tensor sum(int dim) {
        return SumDimFn.apply(this, dim);
    }

    public Tensor max(int dim) {
        return MaxDimFn.apply(this, dim);
    }

    public Tensor argmax(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Storage resultData = Ops.argmaxDim(this, dim);
        int[] newDims = shape.dims();
        newDims[dim] = 1;
        return new Tensor(resultData, new Shape(newDims), 0, false);
    }

    public Tensor logSoftmax(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Tensor maxes = this.max(dim);
        Tensor shifted = this.sub(maxes);
        Tensor sumExp = shifted.exp().sum(dim);
        Tensor logSumExp = sumExp.log();
        return shifted.sub(logSumExp);
    }

    public Tensor softmax(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Tensor shifted = this.sub(this.max(dim).detach());
        Tensor expd = shifted.exp();
        return expd.div(expd.sum(dim));
    }

    public Tensor maskedSoftmax(Tensor mask, int dim) {
        // https://mcognetta.github.io/posts/masked_softmax/
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Tensor m = mask.requiresGrad() ? mask.detach() : mask;
        return this.add(m.rsub(1f).mul(-1e30f)).softmax(dim);
    }

    public Tensor mean() {
        float val = Ops.mean(this);
        Storage out = alloc(1);
        out.segment().set(ValueLayout.JAVA_FLOAT, 0L, val);
        return wrapResult(out, new Shape(1, 1), new MeanFn(this));
    }

    // -------------------------- //
    // --- UNARY ELEMENT-WISE --- //
    // -------------------------- //

    public Tensor tanh() {
        return TanhFn.apply(this);
    }

    public Tensor abs() {
        return AbsFn.apply(this);
    }

    public Tensor exp() {
        return ExpFn.apply(this);
    }

    public Tensor log() {
        return LogFn.apply(this);
    }

    public Tensor sqrt() {
        return SqrtFn.apply(this);
    }

    public Tensor square() {
        return SquareFn.apply(this);
    }

    public Tensor abs_() {
        checkInPlaceSafe();
        Ops.unaryOp(ABS, this, storage);
        return this;
    }

    public Tensor exp_() {
        checkInPlaceSafe();
        Ops.unaryOp(EXP, this, storage);
        return this;
    }

    public Tensor log_() {
        checkInPlaceSafe();
        Ops.unaryOp(LOG, this, storage);
        return this;
    }

    public Tensor sqrt_() {
        checkInPlaceSafe();
        Ops.unaryOp(SQRT, this, storage);
        return this;
    }

    public Tensor square_() {
        checkInPlaceSafe();
        Ops.binaryOp(MUL, this, this, shape, storage);
        return this;
    }

    // --------------------------- //
    // --- BINARY ELEMENT-WISE --- //
    // --------------------------- //

    public Tensor add(Tensor other) {
        return AddFn.apply(this, other);
    }

    public Tensor sub(Tensor other) {
        return SubFn.apply(this, other);
    }

    public Tensor mul(Tensor other) {
        return MulFn.apply(this, other);
    }

    public Tensor div(Tensor other) {
        return DivFn.apply(this, other);
    }

    public Tensor max(Tensor other) {
        return MaxFn.apply(this, other);
    }

    public Tensor min(Tensor other) {
        return MinFn.apply(this, other);
    }

    public Tensor add(float other) {
        return AddConstFn.applyAdd(this, other);
    }

    public Tensor sub(float other) {
        return AddConstFn.applySub(this, other);
    }

    public Tensor mul(float other) {
        return MulConstFn.apply(this, other);
    }

    public Tensor div(float other) {
        return DivConstFn.apply(this, other);
    }

    public Tensor max(float other) {
        return MaxConstFn.apply(this, other);
    }

    public Tensor min(float other) {
        return MinConstFn.apply(this, other);
    }

    public Tensor rsub(float other) {
        return SubRConstFn.apply(other, this);
    }

    public Tensor rdiv(float other) {
        return DivRConstFn.apply(other, this);
    }

    public Tensor add_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(ADD, this, other, resultShape, storage);
        return this;
    }

    public Tensor sub_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(SUB, this, other, resultShape, storage);
        return this;
    }

    public Tensor mul_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(MUL, this, other, resultShape, storage);
        return this;
    }

    public Tensor div_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(DIV, this, other, resultShape, storage);
        return this;
    }

    public Tensor max_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(MAX, this, other, resultShape, storage);
        return this;
    }

    public Tensor min_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(MIN, this, other, resultShape, storage);
        return this;
    }

    public Tensor add_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(ADD, this, other, storage);
        return this;
    }

    public Tensor sub_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(SUB, this, other, storage);
        return this;
    }

    public Tensor mul_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MUL, this, other, storage);
        return this;
    }

    public Tensor div_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(DIV, this, other, storage);
        return this;
    }

    public Tensor max_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MAX, this, other, storage);
        return this;
    }

    public Tensor min_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MIN, this, other, storage);
        return this;
    }

    public Tensor rsub_(float other) {
        checkInPlaceSafe();
        Ops.rbinaryOpConst(SUB, other, this, storage);
        return this;
    }

    public Tensor rdiv_(float other) {
        checkInPlaceSafe();
        Ops.rbinaryOpConst(DIV, other, this, storage);
        return this;
    }

    // ------------------------------- //
    // --- COMPARISON ELEMENT-WISE --- //
    // ------------------------------- //

    private Tensor compareTensor(jdk.incubator.vector.VectorOperators.Comparison op, Tensor other) {
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape != null,
                "cannot broadcast shapes %s and %s", this.shape, other.shape);
        Storage resultData = Ops.comparisonOp(op, this, other, resultShape);
        return new Tensor(resultData, resultShape, 0, false);
    }

    private Tensor compareTensorInPlace(jdk.incubator.vector.VectorOperators.Comparison op, Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape != null && this.shape.equals(resultShape),
                "cannot broadcast shapes %s and %s", this.shape, other.shape);
        Ops.comparisonOp(op, this, other, resultShape, storage);
        return this;
    }

    private Tensor compareConst(jdk.incubator.vector.VectorOperators.Comparison op, float other) {
        Storage resultData = Ops.comparisonOpConst(op, this, other);
        return new Tensor(resultData, shape, 0, false);
    }

    private Tensor compareConstInPlace(jdk.incubator.vector.VectorOperators.Comparison op, float other) {
        checkInPlaceSafe();
        Ops.comparisonOpConst(op, this, other, storage);
        return this;
    }

    public Tensor eq(Tensor other) {
        return compareTensor(EQ, other);
    }

    public Tensor lt(Tensor other) {
        return compareTensor(LT, other);
    }

    public Tensor le(Tensor other) {
        return compareTensor(LE, other);
    }

    public Tensor gt(Tensor other) {
        return compareTensor(GT, other);
    }

    public Tensor ge(Tensor other) {
        return compareTensor(GE, other);
    }

    public Tensor eq_(Tensor other) {
        return compareTensorInPlace(EQ, other);
    }

    public Tensor lt_(Tensor other) {
        return compareTensorInPlace(LT, other);
    }

    public Tensor le_(Tensor other) {
        return compareTensorInPlace(LE, other);
    }

    public Tensor gt_(Tensor other) {
        return compareTensorInPlace(GT, other);
    }

    public Tensor ge_(Tensor other) {
        return compareTensorInPlace(GE, other);
    }

    public Tensor eq(float other) {
        return compareConst(EQ, other);
    }

    public Tensor lt(float other) {
        return compareConst(LT, other);
    }

    public Tensor le(float other) {
        return compareConst(LE, other);
    }

    public Tensor gt(float other) {
        return compareConst(GT, other);
    }

    public Tensor ge(float other) {
        return compareConst(GE, other);
    }

    public Tensor eq_(float other) {
        return compareConstInPlace(EQ, other);
    }

    public Tensor lt_(float other) {
        return compareConstInPlace(LT, other);
    }

    public Tensor le_(float other) {
        return compareConstInPlace(LE, other);
    }

    public Tensor gt_(float other) {
        return compareConstInPlace(GT, other);
    }

    public Tensor ge_(float other) {
        return compareConstInPlace(GE, other);
    }

    // --------------------------- //
    // --- MISC / JAVA METHODS --- //
    // --------------------------- //

    @Override
    public String toString() {
        int ndim = shape.dimensions();
        int numel = shape.numel();

        // collect all values for consistent formatting
        float[] values = new float[numel];
        int[] indices = new int[ndim];
        for (int i = 0; i < numel; i++) {
            values[i] = read(offset + shape.flatIndex(indices));
            for (int d = ndim - 1; d >= 0; d--) {
                if (++indices[d] < shape.dim(d)) break;
                indices[d] = 0;
            }
        }

        // pytorch will simplify floats to integers, and use scientific
        // notation for extreme values.
        boolean allInt = true;
        boolean needSci = false;
        for (float v : values) {
            if (v != (int) v) allInt = false;
            if (v != 0f && (Math.abs(v) >= 1e8 || Math.abs(v) < 1e-4))
                needSci = true;
        }

        // format all values and find max width for alignment
        String[] formatted = new String[numel];
        for (int i = 0; i < numel; i++) {
            if (needSci)
                formatted[i] = String.format("%e", values[i]);
            else if (allInt)
                formatted[i] = String.format("%.0f.", values[i]);
            else
                formatted[i] = String.format("%.4f", values[i]);
        }

        int maxWidth = 0;
        for (String s : formatted)
            maxWidth = Math.max(maxWidth, s.length());

        // pad all values to max width (right-aligned)
        for (int i = 0; i < numel; i++)
            formatted[i] = " ".repeat(maxWidth - formatted[i].length()) + formatted[i];

        // build the nested structure recursively
        StringBuilder sb = new StringBuilder("tensor(");
        int[] pos = {0};
        appendDim(sb, formatted, 0, ndim, "       ", pos);
        sb.append(")");
        return sb.toString();
    }

    private void appendDim(StringBuilder sb, String[] formatted, int dim, int ndim, String indent, int[] pos) {
        sb.append("[");
        int size = shape.dim(dim);

        if (dim == ndim - 1) {
            // innermost dimension: print values
            for (int i = 0; i < size; i++) {
                if (i > 0) sb.append(", ");
                sb.append(formatted[pos[0]++]);
            }
        } else {
            for (int i = 0; i < size; i++) {
                if (i > 0) {
                    sb.append(",");
                    // blank lines between slices for 3D+
                    int blankLines = ndim - dim - 2;
                    for (int b = 0; b <= blankLines; b++) sb.append("\n");
                    sb.append(indent).append(" ".repeat(dim + 1));
                }
                appendDim(sb, formatted, dim + 1, ndim, indent, pos);
            }
        }
        sb.append("]");
    }

}
