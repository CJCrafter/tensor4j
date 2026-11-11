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
      *
     * @param storage the storage that holds the result
     * @param shape the shape of the result
     * @param fn the backward function to record
     * @return the resulting tensor
     */
    public static Tensor wrapResult(Storage storage, Shape shape, TensorFunction fn) {
        return wrapResult(storage, shape, 0, fn);
    }

    /**
     * Variant of {@link #wrapResult(Storage, Shape, TensorFunction)} that
     * allows a non-zero offset (for view-like ops that share storage).
      *
     * @param storage the storage that holds the result
     * @param shape the shape of the result
     * @param offset the offset of the first element in the storage
     * @param fn the backward function to record
     * @return the resulting tensor
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
      *
     * @param dim the dimension to operate along
     * @param tensors the tensors to combine
     * @return the resulting tensor
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

    /**
     * Stacks {@code tensors} (all the same shape) along a new dimension inserted at {@code dim}.
     * Unlike {@link #cat(int, Tensor...)}, which joins tensors along an existing dimension,
     * this adds a new one (e.g. stacking {@code N} boards of shape {@code (C, H, W)} at
     * {@code dim=0} produces a batch of shape {@code (N, C, H, W)}).
      *
     * @param dim the dimension to operate along
     * @param tensors the tensors to combine
     * @return the resulting tensor
     */
    public static Tensor stack(int dim, Tensor... tensors) {
        Preconditions.checkArgument(tensors != null && tensors.length > 0,
                "stack requires at least one tensor");
        int ndim = tensors[0].shape.dimensions();
        Preconditions.checkArgument(dim >= 0 && dim <= ndim,
                "dim %s out of range for stacking %sD tensors", dim, ndim);

        Tensor[] expanded = new Tensor[tensors.length];
        for (int i = 0; i < tensors.length; i++) {
            Tensor t = tensors[i].isContiguous() ? tensors[i] : tensors[i].contiguous();
            int[] dims = t.getShape().dims();
            int[] newDims = new int[dims.length + 1];
            System.arraycopy(dims, 0, newDims, 0, dim);
            newDims[dim] = 1;
            System.arraycopy(dims, dim, newDims, dim + 1, dims.length - dim);
            expanded[i] = t.view(newDims);
        }
        return cat(dim, expanded);
    }

    private AutogradMeta meta() {
        if (meta == null)
            meta = new AutogradMeta();
        return meta;
    }

    /**
     * Returns the {@link Storage} backing this tensor's elements.
      *
     * @return the resulting storage
     */
    public Storage getStorage() {
        return storage;
    }

    /**
     * Returns the element type of this tensor.
     *
     * @return the element type
     */
    public DType dtype() {
        return storage.dtype();
    }

    /**
     * Returns the device this tensor's data lives on.
     *
     * @return the device
     */
    public Device device() {
        return storage.device();
    }

    /**
     * Returns a fresh {@code float[]} containing this tensor's logical
     * elements in row-major order. Always copies; mutations to the returned
     * array do not affect the tensor.
      *
     * @return a copy of the values, in row-major order
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
     *
     * @return the resulting shape
     */
    public Shape getShape() {
        return shape;
    }

    /**
     * Returns this tensor's element offset into its backing {@link Storage}.
      *
     * @return the computed value
     */
    public int getOffset() {
        return offset;
    }

    /**
     * Returns {@code true} if operations on this tensor are tracked for {@link #backward()}.
      *
     * @return the computed value
     */
    public boolean requiresGrad() {
        return requiresGrad;
    }

    /**
     * Returns the accumulated gradient from the last {@link #backward()} call, or
     * {@code null} if none has been computed (or this tensor is unreachable from the loss).
      *
     * @return the resulting tensor
     */
    public Tensor getGrad() {
        return meta == null ? null : meta.grad;
    }

    /**
     * Returns {@code true} if {@link #backward()} has produced a gradient for this tensor.
      *
     * @return the computed value
     */
    public boolean hasGrad() {
        return getGrad() != null;
    }

    /**
     * Returns the {@link TensorFunction} that produced this tensor, or {@code null} if
     * this tensor is a leaf (e.g. built directly via {@link TensorBuilder}).
      *
     * @return the function that produced this tensor, or null for a leaf
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
        Preconditions.checkState(!(this.requiresGrad && meta != null && meta.gradFn != null),
                "in-place operation on a tensor that is part of the computation graph");
        Preconditions.checkState(offset == 0 && storage.capacity() == shape.numel() && isContiguous(),
                "in-place operation not supported on views or non-contiguous tensors");
    }

    /**
     * Runs backpropagation from this (scalar) tensor, accumulating gradients into every
     * ancestor tensor that {@link #requiresGrad()}. The backward computation itself is not
     * tracked by autograd (no double-backward/{@code create_graph} support), so it's safe to
     * call this directly without wrapping it in {@link Tensor4j#noGrad()}.
     *
     * @throws IllegalStateException if this tensor is not a single-element tensor.
     */
    public void backward() {
        Preconditions.checkState(shape.numel() == 1,
                "backward() only supported on scalar tensors, got %s", shape);

        // seed with dL/dL = 1
        Storage seedStorage = alloc(1);
        seedStorage.array()[0] = 1f;
        meta().grad = new Tensor(seedStorage, new Shape(1, 1), 0, false);

        // build reverse-topological order of the forward graph so gradient is
        // fully accumulated from all paths before propagating further back.
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        buildTopo(this, visited, topo);

        // Backward computations themselves must not build a graph: this library has no
        // double-backward (create_graph) support, and several Fn.backward() implementations
        // use in-place ops on freshly-computed intermediates, which would otherwise throw
        // once those intermediates pick up a gradFn of their own.
        try (var ignored = Tensor4j.noGrad()) {
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
    }

    /**
     * Returns the value of this tensor as a plain {@code float}.
     *
     * @throws IllegalStateException if this tensor does not have exactly one element.
     * @return the computed value
     */
    public float item() {
        Preconditions.checkState(shape.numel() == 1,
                "item() only valid on single-element tensors, got shape %s", shape);
        return read(offset);
    }

    /**
     * Returns the element at the given n-dimensional {@code indices}.
      *
     * @param indices the indices to select
     * @return the computed value
     */
    public float get(int... indices) {
        return read(offset + shape.flatIndex(indices));
    }

    /**
     * Writes {@code val} to the element at the given n-dimensional {@code indices}.
      *
     * @param val the value to write
     * @param indices the indices to select
     * @return the resulting tensor
     */
    public Tensor set(float val, int... indices) {
        write(offset + shape.flatIndex(indices), val);
        return this;
    }

    // ----------------- //
    // --- VIEW OPS  --- //
    // ----------------- //

    /**
     * Returns {@code true} if this tensor's data is laid out contiguously in row-major order.
      *
     * @return the computed value
     */
    public boolean isContiguous() {
        return shape.isContiguous();
    }

    /**
     * Swaps the last two dimensions of this tensor; tracked by autograd. Shares storage
     * with the original, so the result is typically not contiguous.
     *
     * @return the resulting tensor
     */
    public Tensor transpose() {
        return TransposeFn.apply(this);
    }

    /**
     * Reorders the dimensions of this tensor so that output dimension {@code i} is input
     * dimension {@code dims[i]}. Shares storage with this tensor, so the result is typically
     * not contiguous. Tracked by autograd.
      *
     * @param dims the new order of the dimensions; output dimension i is input dimension dims[i]
     * @return the resulting tensor
     */
    public Tensor permute(int... dims) {
        return PermuteFn.apply(this, dims);
    }

    /**
     * Flattens every dimension after the first into one, giving shape {@code (N, rest)}.
      *
     * @return the resulting tensor
     */
    public Tensor flatten() {
        int[] dims = shape.dims();
        return contiguousForView().view(dims[0], shape.numel() / dims[0]);
    }

    /**
     * Removes a size-1 dimension at {@code dim}. The result must keep at least 2 dimensions.
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor squeeze(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Preconditions.checkArgument(shape.dimensions() > 2,
                "squeeze would leave fewer than 2 dimensions for shape %s", shape);
        Preconditions.checkArgument(shape.dim(dim) == 1,
                "squeeze dim %s has size %s, must be 1", dim, shape.dim(dim));
        int[] newDims = new int[shape.dimensions() - 1];
        for (int i = 0, j = 0; i < shape.dimensions(); i++)
            if (i != dim) newDims[j++] = shape.dim(i);
        return contiguousForView().view(newDims);
    }

    /**
     * Inserts a size-1 dimension at {@code dim} (which may equal the current number of dimensions).
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor unsqueeze(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim <= shape.dimensions(),
                "dim %s out of range for unsqueezing a %sD tensor", dim, shape.dimensions());
        int[] dims = shape.dims();
        int[] newDims = new int[dims.length + 1];
        System.arraycopy(dims, 0, newDims, 0, dim);
        newDims[dim] = 1;
        System.arraycopy(dims, dim, newDims, dim + 1, dims.length - dim);
        return contiguousForView().view(newDims);
    }

    private Tensor contiguousForView() {
        return isContiguous() ? this : contiguous();
    }

    /**
     * Reinterprets this tensor's dims in reverse order; not tracked by autograd.
     *
     * @return the resulting tensor
     */
    @ApiStatus.Internal
    public Tensor reverseDims() {
        return new Tensor(storage, shape.reverseDims(), offset, false);
    }

    /**
     * Reshapes this tensor to {@code newDims} without copying data; one dimension may be
     * {@code -1} to infer its size. Requires this tensor to be contiguous.
      *
     * @param newDims the new shape; one dimension may be -1 to infer its size
     * @return the resulting tensor
     */
    public Tensor view(int... newDims) {
        return ViewFn.apply(this, newDims);
    }

    /**
     * Returns this tensor if it's already contiguous, otherwise a contiguous copy.
     *
     * @return the resulting tensor
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
      *
     * @return the resulting tensor
     */
    public Tensor detach() {
        return new Tensor(storage, shape, offset, false);
    }

    /**
     * Selects the slices at {@code indices} along {@code dim} and stacks them into a new tensor.
      *
     * @param dim the dimension to operate along
     * @param indices the indices to select
     * @return the resulting tensor
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
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor matmul(Tensor other) {
        return MatmulFn.apply(this, other);
    }

    // ------------------ //
    // --- REDUCTIONS --- //
    // ------------------ //

    /**
     * Returns a {@code 1x1} tensor holding the sum of all elements.
      *
     * @return the resulting tensor
     */
    public Tensor sum() {
        float val = Ops.sum(this);
        Storage out = alloc(1);
        out.array()[0] = val;
        return wrapResult(out, new Shape(1, 1), new SumFn(this));
    }

    /**
     * Sums this tensor along {@code dim}, collapsing it to size 1.
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor sum(int dim) {
        return SumDimFn.apply(this, dim);
    }

    /**
     * Returns the maximum along {@code dim}, collapsed to size 1 (values only; PyTorch's
     * {@code torch.amax}).
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor amax(int dim) {
        return MaxDimFn.apply(this, dim);
    }

    /**
     * Returns the minimum along {@code dim}, collapsed to size 1 (values only; PyTorch's
     * {@code torch.amin}).
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor amin(int dim) {
        return this.mul(-1f).amax(dim).mul(-1f);
    }

    /**
     * Returns a {@code 1x1} tensor holding the maximum of all elements (PyTorch's {@code torch.max(x)}).
      *
     * @return the resulting tensor
     */
    public Tensor max() {
        return flatten().amax(1).amax(0);
    }

    /**
     * Returns a {@code 1x1} tensor holding the minimum of all elements (PyTorch's {@code torch.min(x)}).
      *
     * @return the resulting tensor
     */
    public Tensor min() {
        return flatten().amin(1).amin(0);
    }

    /**
     * Returns the index of the maximum value along {@code dim}, collapsed to size 1.
     * Not tracked by autograd.
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
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
     * Returns the index of the minimum value along {@code dim}, collapsed to size 1.
     * Not tracked by autograd.
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor argmin(int dim) {
        return this.detach().mul(-1f).argmax(dim);
    }

    /**
     * Selects one element per "row" along {@code dim}. {@code indices} must have this
     * tensor's shape with {@code dim} collapsed to size 1 (the same shape {@link #argmax(int)}
     * returns), holding the index to read along {@code dim} as a float. Returns a tensor of
     * that same collapsed shape. Differentiable with respect to this tensor.
      *
     * @param dim the dimension to operate along
     * @param indices the indices to select
     * @return the resulting tensor
     */
    public Tensor gather(int dim, Tensor indices) {
        return oneHot(indices, shape, dim).mul(this).sum(dim);
    }

    /**
     * Expands {@code classIndices} (shaped like {@code shape} with {@code dim} collapsed to
     * 1) into a one-hot mask shaped {@code shape}. Not tracked by autograd.
     */
    static Tensor oneHot(Tensor classIndices, Shape shape, int dim) {
        int numClasses = shape.dim(dim);

        int[] rangeDims = new int[shape.dimensions()];
        Arrays.fill(rangeDims, 1);
        rangeDims[dim] = numClasses;

        float[] range = new float[numClasses];
        for (int i = 0; i < numClasses; i++)
            range[i] = i;

        Tensor classRange = TensorBuilder.builder().shape(rangeDims).fromArray(range);
        return classRange.eq(classIndices);
    }

    /**
     * Returns {@code log(softmax(this, dim))}, computed in a numerically stable way.
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor logSoftmax(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Tensor maxes = this.amax(dim);
        Tensor shifted = this.sub(maxes);
        Tensor sumExp = shifted.exp().sum(dim);
        Tensor logSumExp = sumExp.log();
        return shifted.sub(logSumExp);
    }

    /**
     * Returns the softmax of this tensor along {@code dim}.
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor softmax(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Tensor shifted = this.sub(this.amax(dim).detach());
        Tensor expd = shifted.exp();
        return expd.div(expd.sum(dim));
    }

    /**
     * Returns the softmax of this tensor along {@code dim}, after masking out positions
     * where {@code mask} is 0 (given a very large negative value before exponentiating).
      *
     * @param mask 1 where a position is allowed, 0 where it is masked out
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor maskedSoftmax(Tensor mask, int dim) {
        // https://mcognetta.github.io/posts/masked_softmax/
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        Tensor m = mask.requiresGrad() ? mask.detach() : mask;
        return this.add(m.rsub(1f).mul(-1e30f)).softmax(dim);
    }

    /**
     * Returns the mean along {@code dim}, collapsed to size 1.
      *
     * @param dim the dimension to operate along
     * @return the resulting tensor
     */
    public Tensor mean(int dim) {
        Preconditions.checkArgument(dim >= 0 && dim < shape.dimensions(),
                "dim %s out of range for %sD tensor", dim, shape.dimensions());
        return sum(dim).mul(1f / shape.dim(dim));
    }

    /**
     * Returns a {@code 1x1} tensor holding the mean of all elements.
      *
     * @return the resulting tensor
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
     *
     * @return the resulting tensor
     */
    public Tensor tanh() {
        return TanhFn.apply(this);
    }

    /**
     * Returns the elementwise absolute value of this tensor.
     *
     * @return the resulting tensor
     */
    public Tensor abs() {
        return AbsFn.apply(this);
    }

    /**
     * Returns the elementwise exponential ({@code e^x}) of this tensor.
      *
     * @return the resulting tensor
     */
    public Tensor exp() {
        return ExpFn.apply(this);
    }

    /**
     * Returns the elementwise natural logarithm of this tensor.
     *
     * @return the resulting tensor
     */
    public Tensor log() {
        return LogFn.apply(this);
    }

    /**
     * Returns the elementwise square root of this tensor.
     *
     * @return the resulting tensor
     */
    public Tensor sqrt() {
        return SqrtFn.apply(this);
    }

    /**
     * Returns the elementwise square of this tensor.
     *
     * @return the resulting tensor
     */
    public Tensor square() {
        return SquareFn.apply(this);
    }

    /**
     * In-place elementwise absolute value; not usable on views or graph tensors.
     *
     * @return the resulting tensor
     */
    public Tensor abs_() {
        checkInPlaceSafe();
        Ops.unaryOp(ABS, this, storage);
        return this;
    }

    /**
     * In-place elementwise exponential; not usable on views or graph tensors.
     *
     * @return the resulting tensor
     */
    public Tensor exp_() {
        checkInPlaceSafe();
        Ops.unaryOp(EXP, this, storage);
        return this;
    }

    /**
     * In-place elementwise natural logarithm; not usable on views or graph tensors.
     *
     * @return the resulting tensor
     */
    public Tensor log_() {
        checkInPlaceSafe();
        Ops.unaryOp(LOG, this, storage);
        return this;
    }

    /**
     * In-place elementwise square root; not usable on views or graph tensors.
     *
     * @return the resulting tensor
     */
    public Tensor sqrt_() {
        checkInPlaceSafe();
        Ops.unaryOp(SQRT, this, storage);
        return this;
    }

    /**
     * In-place elementwise square; not usable on views or graph tensors.
     *
     * @return the resulting tensor
     */
    public Tensor square_() {
        checkInPlaceSafe();
        Ops.binaryOp(MUL, this, this, shape, storage);
        return this;
    }

    /**
     * Clamps every element to {@code [min, max]}.
      *
     * @param min the lower bound (inclusive)
     * @param max the upper bound (inclusive)
     * @return the resulting tensor
     */
    public Tensor clamp(float min, float max) {
        return this.max(min).min(max);
    }

    /**
     * In-place clamp of every element to {@code [min, max]}; not usable on views or graph tensors.
      *
     * @param min the lower bound (inclusive)
     * @param max the upper bound (inclusive)
     * @return the resulting tensor
     */
    public Tensor clamp_(float min, float max) {
        return max_(min).min_(max);
    }

    // --------------------------- //
    // --- BINARY ELEMENT-WISE --- //
    // --------------------------- //

    /**
     * Returns the elementwise sum of this tensor and {@code other}, broadcasting shapes as needed.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor add(Tensor other) {
        return AddFn.apply(this, other);
    }

    /**
     * Returns the elementwise difference {@code this - other}, broadcasting shapes as needed.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor sub(Tensor other) {
        return SubFn.apply(this, other);
    }

    /**
     * Returns the elementwise product of this tensor and {@code other}, broadcasting shapes as needed.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor mul(Tensor other) {
        return MulFn.apply(this, other);
    }

    /**
     * Returns the elementwise quotient {@code this / other}, broadcasting shapes as needed.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor div(Tensor other) {
        return DivFn.apply(this, other);
    }

    /**
     * Returns the elementwise maximum of this tensor and {@code other}, broadcasting shapes as needed.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor max(Tensor other) {
        return MaxFn.apply(this, other);
    }

    /**
     * Returns the elementwise minimum of this tensor and {@code other}, broadcasting shapes as needed.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor min(Tensor other) {
        return MinFn.apply(this, other);
    }

    /**
     * Returns this tensor with {@code other} added to every element.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor add(float other) {
        return AddConstFn.applyAdd(this, other);
    }

    /**
     * Returns this tensor with {@code other} subtracted from every element.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor sub(float other) {
        return AddConstFn.applySub(this, other);
    }

    /**
     * Returns this tensor with every element multiplied by {@code other}.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor mul(float other) {
        return MulConstFn.apply(this, other);
    }

    /**
     * Returns this tensor with every element divided by {@code other}.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor div(float other) {
        return DivConstFn.apply(this, other);
    }

    /**
     * Returns the elementwise maximum of this tensor and the constant {@code other}.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor max(float other) {
        return MaxConstFn.apply(this, other);
    }

    /**
     * Returns the elementwise minimum of this tensor and the constant {@code other}.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor min(float other) {
        return MinConstFn.apply(this, other);
    }

    /**
     * Returns {@code other - this} (reversed subtraction) for every element.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor rsub(float other) {
        return SubRConstFn.apply(other, this);
    }

    /**
     * Returns this tensor raised to the constant power {@code p}, elementwise. Negative bases
     * with non-integer exponents produce NaN.
      *
     * @param p the exponent
     * @return the resulting tensor
     */
    public Tensor pow(float p) {
        return PowConstFn.apply(this, p);
    }

    /**
     * Returns {@code other / this} (reversed division) for every element.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor rdiv(float other) {
        return DivRConstFn.apply(other, this);
    }

    /**
     * In-place elementwise add; not usable on views or graph tensors, and the broadcast
     * result must equal this tensor's shape.
     *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
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
     *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
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
     *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
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
     *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
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
     *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
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
     *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
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
     *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor add_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(ADD, this, other, storage);
        return this;
    }

    /**
     * In-place subtract of a constant; not usable on views or graph tensors.
     *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor sub_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(SUB, this, other, storage);
        return this;
    }

    /**
     * In-place multiply by a constant; not usable on views or graph tensors.
     *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor mul_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MUL, this, other, storage);
        return this;
    }

    /**
     * In-place divide by a constant; not usable on views or graph tensors.
     *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor div_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(DIV, this, other, storage);
        return this;
    }

    /**
     * In-place elementwise maximum with a constant; not usable on views or graph tensors.
     *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor max_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MAX, this, other, storage);
        return this;
    }

    /**
     * In-place elementwise minimum with a constant; not usable on views or graph tensors.
     *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor min_(float other) {
        checkInPlaceSafe();
        Ops.binaryOpConst(MIN, this, other, storage);
        return this;
    }

    /**
     * In-place reversed subtraction ({@code this = other - this}); not usable on views or graph tensors.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor rsub_(float other) {
        checkInPlaceSafe();
        Ops.rbinaryOpConst(SUB, other, this, storage);
        return this;
    }

    /**
     * In-place reversed division ({@code this = other / this}); not usable on views or graph tensors.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
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
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor eq(Tensor other) {
        return compareTensor(EQ, other);
    }

    /**
     * Returns a mask tensor (1 where {@code this < other}, 0 elsewhere), broadcasting shapes
     * as needed. Not tracked by autograd.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor lt(Tensor other) {
        return compareTensor(LT, other);
    }

    /**
     * Returns a mask tensor (1 where {@code this <= other}, 0 elsewhere), broadcasting shapes
     * as needed. Not tracked by autograd.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor le(Tensor other) {
        return compareTensor(LE, other);
    }

    /**
     * Returns a mask tensor (1 where {@code this > other}, 0 elsewhere), broadcasting shapes
     * as needed. Not tracked by autograd.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor gt(Tensor other) {
        return compareTensor(GT, other);
    }

    /**
     * Returns a mask tensor (1 where {@code this >= other}, 0 elsewhere), broadcasting shapes
     * as needed. Not tracked by autograd.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor ge(Tensor other) {
        return compareTensor(GE, other);
    }

    /**
     * In-place equality comparison; not usable on views or graph tensors.
     *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor eq_(Tensor other) {
        return compareTensorInPlace(EQ, other);
    }

    /**
     * In-place {@code <} comparison; not usable on views or graph tensors.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor lt_(Tensor other) {
        return compareTensorInPlace(LT, other);
    }

    /**
     * In-place {@code <=} comparison; not usable on views or graph tensors.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor le_(Tensor other) {
        return compareTensorInPlace(LE, other);
    }

    /**
     * In-place {@code >} comparison; not usable on views or graph tensors.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor gt_(Tensor other) {
        return compareTensorInPlace(GT, other);
    }

    /**
     * In-place {@code >=} comparison; not usable on views or graph tensors.
      *
     * @param other the other tensor; shapes are broadcast against each other
     * @return the resulting tensor
     */
    public Tensor ge_(Tensor other) {
        return compareTensorInPlace(GE, other);
    }

    /**
     * Returns a mask tensor (1 where equal to the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor eq(float other) {
        return compareConst(EQ, other);
    }

    /**
     * Returns a mask tensor (1 where less than the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor lt(float other) {
        return compareConst(LT, other);
    }

    /**
     * Returns a mask tensor (1 where less than or equal to the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor le(float other) {
        return compareConst(LE, other);
    }

    /**
     * Returns a mask tensor (1 where greater than the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor gt(float other) {
        return compareConst(GT, other);
    }

    /**
     * Returns a mask tensor (1 where greater than or equal to the constant {@code other}, 0 elsewhere).
     * Not tracked by autograd.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor ge(float other) {
        return compareConst(GE, other);
    }

    /**
     * In-place equality comparison against a constant; not usable on views or graph tensors.
     *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor eq_(float other) {
        return compareConstInPlace(EQ, other);
    }

    /**
     * In-place {@code <} comparison against a constant; not usable on views or graph tensors.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor lt_(float other) {
        return compareConstInPlace(LT, other);
    }

    /**
     * In-place {@code <=} comparison against a constant; not usable on views or graph tensors.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor le_(float other) {
        return compareConstInPlace(LE, other);
    }

    /**
     * In-place {@code >} comparison against a constant; not usable on views or graph tensors.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
     */
    public Tensor gt_(float other) {
        return compareConstInPlace(GT, other);
    }

    /**
     * In-place {@code >=} comparison against a constant; not usable on views or graph tensors.
      *
     * @param other the constant to combine with each element
     * @return the resulting tensor
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
