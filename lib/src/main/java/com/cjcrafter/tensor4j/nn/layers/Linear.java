package com.cjcrafter.tensor4j.nn.layers;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;

import java.util.random.RandomGenerator;

/**
 * A fully connected dense layer with weights and bias. All weights and biases
 * will be uniformly distributed {@code [-k, k)}, where
 * {@code k = 1.0 / Math.sqrt(inFeatures)} (by pytorch convention).
 */
public class Linear extends Module {

    private final Tensor weight; // (out, in)
    private final Tensor bias;   // (1, out)

    public Linear(int inFeatures, int outFeatures) {
        this(Tensor4j.getRandom(), inFeatures, outFeatures, (float) (1.0 / Math.sqrt(inFeatures)), true);
    }

    public Linear(RandomGenerator rng, int inFeatures, int outFeatures, float k, boolean includeBias) {
        this.weight = TensorBuilder.builder()
                .shape(inFeatures, outFeatures) // these are "swapped" for implicit transpose...
                .requiresGrad()
                .rand(rng, -k, k);  // pytorch convention
        this.bias = (!includeBias) ? null : TensorBuilder.builder()
                .shape(1, outFeatures)
                .requiresGrad()
                .rand(rng, -k, k);  // pytorch convention
    }

    @Override
    public Tensor forward(Tensor input) {
        // input: (batch, in), weight: (out, in)
        // y = input @ weight^T + bias
        Tensor wx = input.matmul(weight);
        if (bias != null)
            return wx.add(bias);

        return wx;
    }
}
