package com.cjcrafter.tensor4j;

import com.cjcrafter.tensor4j.autograd.*;
import com.cjcrafter.tensor4j.memory.HeapStorage;
import com.cjcrafter.tensor4j.memory.Storage;
import com.cjcrafter.tensor4j.ops.Ops;
import com.google.common.base.Preconditions;
import org.jetbrains.annotations.ApiStatus;

import java.util.*;

import static jdk.incubator.vector.VectorOperators.*;

/**
 * A multi-dimensional float32 matrix backed by a heap-array {@link Storage}.
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
    }

    private static Storage alloc(int numel) {
        return HeapStorage.allocate(DType.F32, numel);
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

    /**
     * Concatenates {@code tensors} along {@code dim}; all other dimensions must match.
     */
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
        float[] outArr = out.array();

        int cursor = 0;
        for (int k = 0; k < tensors.length; k++) {
            Tensor t = tensors[k].isContiguous() ? tensors[k] : tensors[k].contiguous();
            int sizeK = sizesAlongDim[k];
            int inDimStride = sizeK * inner;
            float[] tArr = t.storage.array();
            int tOff = t.offset;
            for (int o = 0; o < outer; o++) {
                int outBase = o * outDimStride + cursor * inner;
                int inBase = tOff + o * inDimStride;
                System.arraycopy(tArr, inBase, outArr, outBase, sizeK * inner);
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

    /**
     * Returns the {@link Storage} backing this tensor's elements.
     */
    public Storage getStorage() {
        return storage;
    }

    /**
     * Returns the element type of this tensor.
     */
    public DType dtype() {
        return storage.dtype();
    }

    /**
     * Returns the device this tensor's data lives on.
     */
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
        System.arraycopy(src.storage.array(), src.offset, out, 0, n);
        return out;
    }

    private float read(int idx) {
        return storage.array()[idx];
    }

    private void write(int idx, float val) {
        storage.array()[idx] = val;
    }

    /**
     * Returns this tensor's shape.
     */
    public Shape getShape() {
        return shape;
    }

    /**
     * Returns this tensor's element offset into its backing {@link Storage}.
     */
    public int getOffset() {
        return offset;
    }

    /**
     * Returns {@code true} if operations on this tensor are tracked for {@link #backward()}.
     */
    public boolean requiresGrad() {
        return requiresGrad;
    }

    /**
     * Returns the accumulated gradient from the last {@link #backward()} call, or
     * {@code null} if none has been computed (or this tensor is unreachable from the loss).
     */
    public Tensor getGrad() {
        return meta == null ? null : meta.grad;
    }

    /**
     * Returns the {@link TensorFunction} that produced this tensor, or {@code null} if
     * this tensor is a leaf (e.g. built directly via {@link TensorBuilder}).
     */
    public TensorFunction getGradFn() {
        return meta == null ? null : meta.gradFn;
    }

    void setGradFn(TensorFunction fn) {
        meta().gradFn = fn;
    }

    /**
     * Clears this tensor's accumulated gradient.
     */
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

    /**
     * Runs backpropagation from this (scalar) tensor, accumulating gradients into every
     * ancestor tensor that {@link #requiresGrad()}.
     *
     * @throws IllegalStateException if this tensor is not a single-element tensor.
     */
    public void backward() {
        if (shape.numel() != 1)
            throw new IllegalStateException("backward() only supported on scalar tensors, got " + shape);

        // seed with dL/dL = 1
        Storage seedStorage = alloc(1);
        seedStorage.array()[0] = 1f;
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

    /**
     * Returns the value of this tensor as a plain {@code float}.
     *
     * @throws IllegalStateException if this tensor does not have exactly one element.
     */
    public float item() {
        if (shape.numel() != 1)
            throw new IllegalStateException("item() only valid on single-element tensors, got shape " + shape);
        return read(offset);
    }

    /**
     * Returns the element at the given n-dimensional {@code indices}.
     */
    public float get(int... indices) {
        return read(offset + shape.flatIndex(indices));
    }

    /**
     * Writes {@code val} to the element at the given n-dimensional {@code indices}.
     */
    public void set(float val, int... indices) {
        write(offset + shape.flatIndex(indices), val);
    }

    // ----------------- //
    // --- VIEW OPS  --- //
    // ----------------- //

    /**
     * Returns {@code true} if this tensor's data is laid out contiguously in row-major order.
     */
    public boolean isContiguous() {
        return shape.isContiguous();
    }

    /**
     * Swaps the last two dimensions of this tensor; tracked by autograd. Shares storage
     * with the original, so the result is typically not contiguous.
     */
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

    /**
     * Reshapes this tensor to {@code newDims} without copying data; one dimension may be
     * {@code -1} to infer its size. Requires this tensor to be contiguous.
     */
    public Tensor view(int... newDims) {
        return ViewFn.apply(this, newDims);
    }

    /**
     * Returns this tensor if it's already contiguous, otherwise a contiguous copy.
     */
    public Tensor contiguous() {
        if (isContiguous())
            return this;

        int n = shape.numel();
        Storage dst = alloc(n);
        float[] dstArr = dst.array();
        float[] srcArr = storage.array();
        int[] indices = new int[shape.dimensions()];

        for (int flat = 0; flat < n; flat++) {
            dstArr[flat] = srcArr[offset + shape.flatIndex(indices)];

            for (int d = indices.length - 1; d >= 0; d--) {
                if (++indices[d] < shape.dim(d)) break;
                indices[d] = 0;
            }
        }

        return wrapResult(dst, new Shape(shape.dims()), new IdentityFn(this));
    }

    /**
     * Returns a deep copy of this tensor's storage; tracked by autograd as an identity op.
     */
    @Override
    public Tensor clone() {
        int capacity = (int) storage.capacity();
        Storage dst = alloc(capacity);
        System.arraycopy(storage.array(), 0, dst.array(), 0, capacity);
        return wrapResult(dst, shape, new IdentityFn(this));
    }

    /**
     * Returns a tensor sharing this tensor's storage but detached from the autograd graph
     * (i.e. {@link #requiresGrad()} is {@code false} and it has no grad function).
     */
    public Tensor detach() {
        return new Tensor(storage, shape, offset, false);
    }

    /**
     * Selects the slices at {@code indices} along {@code dim} and stacks them into a new tensor.
     */
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
        float[] outArr = out.array();
        float[] srcArr = storage.array();
        int outPos = 0;

        for (int outer = 0; outer < outerSize; outer++) {
            int outerOff = offset + outer * dimStride;
            for (int idx : indices) {
                System.arraycopy(srcArr, outerOff + idx * sliceSize, outArr, outPos, sliceSize);
                outPos += sliceSize;
            }
        }

        return wrapResult(out, new Shape(newDims), new IndexSelectFn(this, dim, indices));
    }

    // -------------- //
    // --- MATMUL --- //
    // -------------- //

    /**
     * Matrix-multiplies this tensor by {@code other}, with N-D batched broadcasting
     * (PyTorch semantics): the last two dims of each input are the matrix dims, and
     * everything before is a batch shape broadcast against the other input's.
     */
    public Tensor matmul(Tensor other) {
        return MatmulFn.apply(this, other);
    }

    // ------------------ //
    // --- REDUCTIONS --- //
    // ------------------ //

    /**
     * Returns a {@code 1x1} tensor holding the sum of all elements.
     */
    public Tensor sum() {
        float val = Ops.sum(this);
        Storage out = alloc(1);
        out.array()[0] = val;
        return wrapResult(out, new Shape(1, 1), new SumFn(this));
    }

    /**
     * Sums this tensor along {@code dim}, collapsing it to size 1.
     */
    public Tensor sum(int dim) {
        return SumDimFn.apply(this, dim);
    }

    /**
     * Returns the maximum value along {@code dim}, collapsed to size 1.
     */
    public Tensor max(int dim) {
        return MaxDimFn.apply(this, dim);
    }

    /**
     * Returns the index of the maximum value along {@code dim}, collapsed to size 1.
     * Not tracked by autograd.
     */
    public Tensor argmax(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Storage resultData = Ops.argmaxDim(this, dim);
        int[] newDims = shape.dims();
        newDims[dim] = 1;
        return new Tensor(resultData, new Shape(newDims), 0, false);
    }

    /**
     * Returns {@code log(softmax(this, dim))}, computed in a numerically stable way.
     */
    public Tensor logSoftmax(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Tensor maxes = this.max(dim);
        Tensor shifted = this.sub(maxes);
        Tensor sumExp = shifted.exp().sum(dim);
        Tensor logSumExp = sumExp.log();
        return shifted.sub(logSumExp);
    }

    /**
     * Returns the softmax of this tensor along {@code dim}.
     */
    public Tensor softmax(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Tensor shifted = this.sub(this.max(dim).detach());
        Tensor expd = shifted.exp();
        return expd.div(expd.sum(dim));
    }

    /**
     * Returns the softmax of this tensor along {@code dim}, after masking out positions
     * where {@code mask} is 0 (given a very large negative value before exponentiating).
     */
    public Tensor maskedSoftmax(Tensor mask, int dim) {
        // https://mcognetta.github.io/posts/masked_softmax/
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Tensor m = mask.requiresGrad() ? mask.detach() : mask;
        return this.add(m.rsub(1f).mul(-1e30f)).softmax(dim);
    }

    /**
     * Returns a {@code 1x1} tensor holding the mean of all elements.
     */
    public Tensor mean() {
        float val = Ops.mean(this);
        Storage out = alloc(1);
        out.array()[0] = val;
        return wrapResult(out, new Shape(1, 1), new MeanFn(this));
    }

    // -------------------------- //
    // --- UNARY ELEMENT-WISE --- //
    // -------------------------- //

    /**
     * Returns the elementwise hyperbolic tangent of this tensor.
     */
    public Tensor tanh() {
        return TanhFn.apply(this);
    }

    /**
     * Returns the elementwise absolute value of this tensor.
     */
    public Tensor abs() {
        return AbsFn.apply(this);
    }

    /**
     * Returns the elementwise exponential ({@code e^x}) of this tensor.
     */
    public Tensor exp() {
        return ExpFn.apply(this);
    }

    /**
     * Returns the elementwise natural logarithm of this tensor.
     */
    public Tensor log() {
        return LogFn.apply(this);
    }

    /**
     * Returns the elementwise square root of this tensor.
     */
    public Tensor sqrt() {
        return SqrtFn.apply(this);
    }

    /**
     * Returns the elementwise square of this tensor.
     */
    public Tensor square() {
        return SquareFn.apply(this);
    }

    /**
     * In-place elementwise absolute value; not usable on views or graph tensors.
     */
    public Tensor abs_() {
        checkInPlaceSafe();
        Ops.unaryOp(ABS, this, storage);
        return this;
    }

    /**
     * In-place elementwise exponential; not usable on views or graph tensors.
     */
    public Tensor exp_() {
        checkInPlaceSafe();
        Ops.unaryOp(EXP, this, storage);
        return this;
    }

    /**
     * In-place elementwise natural logarithm; not usable on views or graph tensors.
     */
    public Tensor log_() {
        checkInPlaceSafe();
        Ops.unaryOp(LOG, this, storage);
        return this;
    }

    /**
     * In-place elementwise square root; not usable on views or graph tensors.
     */
    public Tensor sqrt_() {
        checkInPlaceSafe();
        Ops.unaryOp(SQRT, this, storage);
        return this;
    }

    /**
     * In-place elementwise square; not usable on views or graph tensors.
     */
    public Tensor square_() {
        checkInPlaceSafe();
        Ops.binaryOp(MUL, this, this, shape, storage);
        return this;
    }

    // --------------------------- //
    // --- BINARY ELEMENT-WISE --- //
    // --------------------------- //

    /**
     * Returns the elementwise sum of this tensor and {@code other}, broadcasting shapes as needed.
     */
    public Tensor add(Tensor other) {
        return AddFn.apply(this, other);
    }

    /**
     * Returns the elementwise difference {@code this - other}, broadcasting shapes as needed.
     */
    public Tensor sub(Tensor other) {
        return SubFn.apply(this, other);
    }

    /**
     * Returns the elementwise product of this tensor and {@code other}, broadcasting shapes as needed.
     */
    public Tensor mul(Tensor other) {
        return MulFn.apply(this, other);
    }

    /**
     * Returns the elementwise quotient {@code this / other}, broadcasting shapes as needed.
     */
    public Tensor div(Tensor other) {
        return DivFn.apply(this, other);
    }

    /**
     * Returns the elementwise maximum of this tensor and {@code other}, broadcasting shapes as needed.
     */
    public Tensor max(Tensor other) {
        return MaxFn.apply(this, other);
    }

    /**
     * Returns the elementwise minimum of this tensor and {@code other}, broadcasting shapes as needed.
     */
    public Tensor min(Tensor other) {
        return MinFn.apply(this, other);
    }

    /**
     * Returns this tensor with {@code other} added to every element.
     */
    public Tensor add(float other) {
        return AddConstFn.applyAdd(this, other);
    }

    /**
     * Returns this tensor with {@code other} subtracted from every element.
     */
    public Tensor sub(float other) {
        return AddConstFn.applySub(this, other);
    }

    /**
     * Returns this tensor with every element multiplied by {@code other}.
     */
    public Tensor mul(float other) {
        return MulConstFn.apply(this, other);
    }

    /**
     * Returns this tensor with every element divided by {@code other}.
     */
    public Tensor div(float other) {
        return DivConstFn.apply(this, other);
    }

    /**
     * Returns the elementwise maximum of this tensor and the constant {@code other}.
     */
    public Tensor max(float other) {
        return MaxConstFn.apply(this, other);
    }

    /**
     * Returns the elementwise minimum of this tensor and the constant {@code other}.
     */
    public Tensor min(float other) {
        return MinConstFn.apply(this, other);
    }

    /**
     * Returns {@code other - this} (reversed subtraction) for every element.
     */
    public Tensor rsub(float other) {
        return SubRConstFn.apply(other, this);
    }

    /**
     * Returns {@code other / this} (reversed division) for every element.
     */
    public Tensor rdiv(float other) {
        return DivRConstFn.apply(other, this);
    }

    /**
     * In-place elementwise add; not usable on views or graph tensors, and the broadcast
     * result must equal this tensor's shape.
     */
    public Tensor add_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(ADD, this, other, resultShape, storage);
        return this;
    }

    /**
     * In-place elementwise subtract; not usable on views or graph tensors, and the broadcast
     * result must equal this tensor's shape.
     */
    public Tensor sub_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(SUB, this, other, resultShape, storage);
        return this;
    }

    /**
     * In-place elementwise multiply; not usable on views or graph tensors, and the broadcast
     * result must equal this tensor's shape.
     */
    public Tensor mul_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(MUL, this, other, resultShape, storage);
        return this;
    }

    /**
     * In-place elementwise divide; not usable on views or graph tensors, and the broadcast
     * result must equal this tensor's shape.
     */
    public Tensor div_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(DIV, this, other, resultShape, storage);
        return this;
    }

    /**
     * In-place elementwise maximum; not usable on views or graph tensors, and the broadcast
     * result must equal this tensor's shape.
     */
    public Tensor max_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(MAX, this, other, resultShape, storage);
        return this;
    }

    /**
     * In-place elementwise minimum; not usable on views or graph tensors, and the broadcast
     * result must equal this tensor's shape.
     */
    public Tensor min_(Tensor other) {
        checkInPlaceSafe();
        Shape resultShape = this.shape.broadcastWith(other.shape);
        Preconditions.checkArgument(resultShape.equals(this.shape),
                "in-place: result shape must equal this.shape");
        Ops.binaryOp(MIN, this, other, resultShape, storage);
        return this;
    }

    /**
     * In-place add of a constant; not usable on views or graph tensors.
     */
    public Tensor add_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(ADD, this, other, storage);
        return this;
    }

    /**
     * In-place subtract of a constant; not usable on views or graph tensors.
     */
    public Tensor sub_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(SUB, this, other, storage);
        return this;
    }

    /**
     * In-place multiply by a constant; not usable on views or graph tensors.
     */
    public Tensor mul_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MUL, this, other, storage);
        return this;
    }

    /**
     * In-place divide by a constant; not usable on views or graph tensors.
     */
    public Tensor div_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(DIV, this, other, storage);
        return this;
    }

    /**
     * In-place elementwise maximum with a constant; not usable on views or graph tensors.
     */
    public Tensor max_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MAX, this, other, storage);
        return this;
    }

    /**
     * In-place elementwise minimum with a constant; not usable on views or graph tensors.
     */
    public Tensor min_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MIN, this, other, storage);
        return this;
    }

    /**
     * In-place reversed subtraction ({@code this = other - this}); not usable on views or graph tensors.
     */
    public Tensor rsub_(float other) {
        checkInPlaceSafe();
        Ops.rbinaryOpConst(SUB, other, this, storage);
        return this;
    }

    /**
     * In-place reversed division ({@code this = other / this}); not usable on views or graph tensors.
     */
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

    /**
     * Returns a mask tensor (1 where equal, 0 elsewhere) comparing this tensor to {@code other},
     * broadcasting shapes as needed. Not tracked by autograd.
     */
    public Tensor eq(Tensor other) {
        return compareTensor(EQ, other);
    }

    /**
     * Returns a mask tensor (1 where {@code this < other}, 0 elsewhere), broadcasting shapes
     * as needed. Not tracked by autograd.
     */
    public Tensor lt(Tensor other) {
        return compareTensor(LT, other);
    }

    /**
     * Returns a mask tensor (1 where {@code this <= other}, 0 elsewhere), broadcasting shapes
     * as needed. Not tracked by autograd.
     */
    public Tensor le(Tensor other) {
        return compareTensor(LE, other);
    }

    /**
     * Returns a mask tensor (1 where {@code this > other}, 0 elsewhere), broadcasting shapes
     * as needed. Not tracked by autograd.
     */
    public Tensor gt(Tensor other) {
        return compareTensor(GT, other);
    }

    /**
     * Returns a mask tensor (1 where {@code this >= other}, 0 elsewhere), broadcasting shapes
     * as needed. Not tracked by autograd.
     */
    public Tensor ge(Tensor other) {
        return compareTensor(GE, other);
    }

    /**
     * In-place equality comparison; not usable on views or graph tensors.
     */
    public Tensor eq_(Tensor other) {
        return compareTensorInPlace(EQ, other);
    }

    /**
     * In-place {@code <} comparison; not usable on views or graph tensors.
     */
    public Tensor lt_(Tensor other) {
        return compareTensorInPlace(LT, other);
    }

    /**
     * In-place {@code <=} comparison; not usable on views or graph tensors.
     */
    public Tensor le_(Tensor other) {
        return compareTensorInPlace(LE, other);
    }

    /**
     * In-place {@code >} comparison; not usable on views or graph tensors.
     */
    public Tensor gt_(Tensor other) {
        return compareTensorInPlace(GT, other);
    }

    /**
     * In-place {@code >=} comparison; not usable on views or graph tensors.
     */
    public Tensor ge_(Tensor other) {
        return compareTensorInPlace(GE, other);
    }

    /**
     * Returns a mask tensor (1 where equal to the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
     */
    public Tensor eq(float other) {
        return compareConst(EQ, other);
    }

    /**
     * Returns a mask tensor (1 where less than the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
     */
    public Tensor lt(float other) {
        return compareConst(LT, other);
    }

    /**
     * Returns a mask tensor (1 where less than or equal to the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
     */
    public Tensor le(float other) {
        return compareConst(LE, other);
    }

    /**
     * Returns a mask tensor (1 where greater than the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
     */
    public Tensor gt(float other) {
        return compareConst(GT, other);
    }

    /**
     * Returns a mask tensor (1 where greater than or equal to the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
     */
    public Tensor ge(float other) {
        return compareConst(GE, other);
    }

    /**
     * In-place equality comparison against a constant; not usable on views or graph tensors.
     */
    public Tensor eq_(float other) {
        return compareConstInPlace(EQ, other);
    }

    /**
     * In-place {@code <} comparison against a constant; not usable on views or graph tensors.
     */
    public Tensor lt_(float other) {
        return compareConstInPlace(LT, other);
    }

    /**
     * In-place {@code <=} comparison against a constant; not usable on views or graph tensors.
     */
    public Tensor le_(float other) {
        return compareConstInPlace(LE, other);
    }

    /**
     * In-place {@code >} comparison against a constant; not usable on views or graph tensors.
     */
    public Tensor gt_(float other) {
        return compareConstInPlace(GT, other);
    }

    /**
     * In-place {@code >=} comparison against a constant; not usable on views or graph tensors.
     */
    public Tensor ge_(float other) {
        return compareConstInPlace(GE, other);
    }

    // --------------------------- //
    // --- MISC / JAVA METHODS --- //
    // --------------------------- //

    /**
     * Formats this tensor like PyTorch's {@code print(tensor)}: nested brackets, aligned
     * columns, integers simplified when all values are whole, and scientific notation for
     * very large or very small magnitudes.
     */
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
