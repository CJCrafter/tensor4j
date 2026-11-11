package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.google.common.base.Preconditions;
import com.cjcrafter.tensor4j.TensorBuilder;

/**
 * Normalizes a {@code (N, C, H, W)} input to zero mean and unit variance per channel (over
 * the {@code N}, {@code H}, and {@code W} axes), then applies a learnable per-channel scale
 * ({@code weight}, initialized to 1) and shift ({@code bias}, initialized to 0).
 *
 * <p>During training, batch statistics are used and a running mean/variance is updated
 * (exponential moving average, PyTorch's default {@code momentum = 0.1} convention:
 * {@code running = (1 - momentum) * running + momentum * batch}). During evaluation
 * ({@link #eval()}), the running statistics are used instead so a single input can be
 * normalized consistently. The running statistics are plain (non-trained) tensors, so they
 * round-trip through {@link Module#save} / {@link Module#load} along with {@code weight}/
 * {@code bias}, matching how real frameworks persist BatchNorm state.
 */
public class BatchNorm2d extends Module {

    private final Tensor weight;      // (1, C, 1, 1), learnable
    private final Tensor bias;        // (1, C, 1, 1), learnable
    private final Tensor runningMean; // (1, C, 1, 1), not learnable
    private final Tensor runningVar;  // (1, C, 1, 1), not learnable
    private final float eps;
    private final float momentum;
    private boolean isTrain = true;

    /**
     * Creates a batch norm over {@code channels} channels, using the default
     * {@code eps = 1e-5} and {@code momentum = 0.1}.
     * @param channels the number of channels
     */
    public BatchNorm2d(int channels) {
        this(channels, 1e-5f, 0.1f);
    }

    /**
     * Creates a batch norm over {@code channels} channels.
     *
     * @param channels The number of input channels.
     * @param eps       Added to the variance before taking the square root, to avoid division by zero.
     * @param momentum  The weight given to each new batch's statistics when updating the running average.
     */
    public BatchNorm2d(int channels, float eps, float momentum) {
        this.eps = eps;
        this.momentum = momentum;
        this.weight = TensorBuilder.builder().shape(1, channels, 1, 1).requiresGrad().ones();
        this.bias = TensorBuilder.builder().shape(1, channels, 1, 1).requiresGrad().zeros();
        this.runningMean = TensorBuilder.builder().shape(1, channels, 1, 1).zeros();
        this.runningVar = TensorBuilder.builder().shape(1, channels, 1, 1).ones();
    }

    @Override
    public Tensor forward(Tensor input) {
        int channels = weight.getShape().dim(1);
        Preconditions.checkArgument(input.getShape().dim(1) == channels,
                "BatchNorm2d expected %s channels, got shape %s", channels, input.getShape());

        Tensor mean;
        Tensor variance;
        if (isTrain) {
            int n = input.getShape().dim(0);
            int h = input.getShape().dim(2);
            int w = input.getShape().dim(3);
            float invCount = 1f / (n * h * w);

            mean = input.sum(0).sum(2).sum(3).mul(invCount);
            variance = input.sub(mean).square().sum(0).sum(2).sum(3).mul(invCount);
            updateRunningStats(mean, variance);
        } else {
            mean = runningMean;
            variance = runningVar;
        }

        Tensor centered = input.sub(mean);
        Tensor invStd = variance.add(eps).sqrt().rdiv(1f);
        Tensor normalized = centered.mul(invStd);
        return normalized.mul(weight).add(bias);
    }

    private void updateRunningStats(Tensor batchMean, Tensor batchVar) {
        float[] mean = batchMean.toFloatArray();
        float[] var = batchVar.toFloatArray();
        float[] rMean = runningMean.getStorage().array();
        float[] rVar = runningVar.getStorage().array();
        int meanOff = runningMean.getOffset();
        int varOff = runningVar.getOffset();

        for (int c = 0; c < mean.length; c++) {
            rMean[meanOff + c] = momentum * mean[c] + (1f - momentum) * rMean[meanOff + c];
            rVar[varOff + c] = momentum * var[c] + (1f - momentum) * rVar[varOff + c];
        }
    }

    @Override
    public void train(boolean mode) {
        this.isTrain = mode;
    }
}
