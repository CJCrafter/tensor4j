package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.google.common.base.Preconditions;
import com.cjcrafter.tensor4j.TensorBuilder;

/**
 * Normalizes the last dimension of the input to zero mean and unit variance, then applies
 * a learnable elementwise scale ({@code weight}, initialized to 1) and shift ({@code bias},
 * initialized to 0).
 */
public class LayerNorm extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final int featureSize;
    private final float eps;

    /**
     * Creates a layer norm over a last dimension of size {@code featureSize}, using a
     * default epsilon of {@code 1e-5} for numerical stability.
     * @param featureSize the expected size of the last dimension
     */
    public LayerNorm(int featureSize) {
        this(featureSize, 1e-5f);
    }

    /**
     * Creates a layer norm over a last dimension of size {@code featureSize}.
     *
     * @param featureSize The expected size of the input's last dimension.
     * @param eps         Added to the variance before taking the square root, to avoid division by zero.
     */
    public LayerNorm(int featureSize, float eps) {
        this.featureSize = featureSize;
        this.eps = eps;
        this.weight = TensorBuilder.builder().shape(1, featureSize).requiresGrad().ones();
        this.bias = TensorBuilder.builder().shape(1, featureSize).requiresGrad().zeros();
    }

    @Override
    public Tensor forward(Tensor input) {
        int lastDim = input.getShape().dimensions() - 1;
        Preconditions.checkArgument(input.getShape().dim(lastDim) == featureSize,
                "LayerNorm expected last dim %s, got shape %s", featureSize, input.getShape());

        float invN = 1f / featureSize;
        Tensor mean = input.sum(lastDim).mul(invN);
        Tensor centered = input.sub(mean);
        Tensor variance = centered.square().sum(lastDim).mul(invN);
        Tensor invStd = variance.add(eps).sqrt().rdiv(1f);
        Tensor normalized = centered.mul(invStd);
        return normalized.mul(weight).add(bias);
    }
}
