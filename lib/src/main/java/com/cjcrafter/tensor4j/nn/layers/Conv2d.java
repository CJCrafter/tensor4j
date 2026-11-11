package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.autograd.Conv2dFn;

import java.util.random.RandomGenerator;

/**
 * A 2D convolutional layer over {@code (N, Cin, H, W)} input, producing
 * {@code (N, Cout, outH, outW)}. Weights are shaped {@code (Cout, Cin, KH, KW)} and
 * uniformly initialized in {@code [-k, k)} where {@code k = 1 / sqrt(Cin * KH * KW)},
 * matching {@link Linear}'s convention.
 *
 * <p>Use {@code padding = (kernelSize - 1) / 2} with an odd {@code kernelSize} to keep the
 * spatial size unchanged ("same" padding), which is what most board-game CNN towers want.
 */
public class Conv2d extends Module {

    private final Tensor weight; // (out, in, kh, kw)
    private final Tensor bias;   // (1, out), or null if no bias
    private final int stride;
    private final int padding;

    /**
     * Creates a conv layer with a bias, stride 1, and no padding, using the default random generator.
     *
     * @param inChannels the number of input channels
     * @param outChannels the number of output channels
     * @param kernelSize the side length of the square kernel
     */
    public Conv2d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0);
    }

    /**
     * Creates a conv layer with a bias, using the default random generator.
     *
     * @param inChannels the number of input channels
     * @param outChannels the number of output channels
     * @param kernelSize the side length of the square kernel
     * @param stride the step between window positions
     * @param padding the zero-padding added on each side
     */
    public Conv2d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(Tensor4j.getRandom(), inChannels, outChannels, kernelSize, stride, padding,
                (float) (1.0 / Math.sqrt((double) inChannels * kernelSize * kernelSize)), true);
    }

    /**
     * Creates a conv layer with weights/bias uniformly initialized in {@code [-k, k)}.
     *
     * @param rng         The random generator to sample initial weights/bias from.
     * @param inChannels  The number of input channels.
     * @param outChannels The number of output channels (filters).
     * @param kernelSize  The size of the (square) convolution kernel.
     * @param stride      The stride of the convolution.
     * @param padding     Symmetric zero-padding applied to both spatial dimensions.
     * @param k           The half-width of the uniform initialization range.
     * @param includeBias If {@code false}, this layer has no bias term (typical when
     *                    followed by {@link BatchNorm2d}, whose own shift subsumes it).
     */
    public Conv2d(RandomGenerator rng, int inChannels, int outChannels, int kernelSize,
                 int stride, int padding, float k, boolean includeBias) {
        this.weight = TensorBuilder.builder()
                .shape(outChannels, inChannels, kernelSize, kernelSize)
                .requiresGrad()
                .rand(rng, -k, k);
        this.bias = !includeBias ? null : TensorBuilder.builder()
                .shape(1, outChannels)
                .requiresGrad()
                .rand(rng, -k, k);
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor b = (bias != null) ? bias
                : TensorBuilder.builder().shape(1, weight.getShape().dim(0)).zeros();
        return Conv2dFn.apply(input, weight, b, stride, padding);
    }
}
