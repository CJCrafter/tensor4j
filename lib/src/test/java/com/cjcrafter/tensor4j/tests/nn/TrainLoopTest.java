package com.cjcrafter.tensor4j.tests.nn;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.data.*;
import com.cjcrafter.tensor4j.io.NpzFormat;
import com.cjcrafter.tensor4j.nn.layers.Linear;
import com.cjcrafter.tensor4j.nn.layers.Module;
import com.cjcrafter.tensor4j.nn.layers.ReLU;
import com.cjcrafter.tensor4j.nn.layers.Sequential;
import com.cjcrafter.tensor4j.nn.losses.MSELoss;
import com.cjcrafter.tensor4j.nn.optimizers.AdamW;
import com.cjcrafter.tensor4j.nn.optimizers.Optimizer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class TrainLoopTest {

    @Test
    void testMnistNetwork() throws IOException, URISyntaxException {
        Tensor4j.manualSeed(115);

        System.out.println("Loading mnist dataset...");
        URL url = getClass().getClassLoader().getResource("mnist.npz");
        Path mnistPath = Path.of(url.toURI());
        Map<String, Tensor> tensors = NpzFormat.load(mnistPath);

        System.out.println("Reshaping tensors...");
        Tensor x_train = tensors.get("x_train").view(-1, 28 * 28).mul(1f / 255f);
        Tensor x_test = tensors.get("x_test").view(-1, 28 * 28).mul(1f / 255f);
        Tensor y_train = oneHot(tensors.get("y_train"), 10);
        Tensor y_test = oneHot(tensors.get("y_test"), 10);

        Module model = new Sequential(
                new Linear(28 * 28, 800),
                new ReLU(),
                new Linear(800, 10)
        );

        MSELoss loss = new MSELoss();
        Collection<Tensor> parameters = model.parameters();
        Optimizer optimizer = new AdamW(parameters, AdamW.DEFAULT_LR, AdamW.DEFAULT_BETA1, AdamW.DEFAULT_BETA2, 0.01f, AdamW.DEFAULT_EPS);

        Dataset dataset = new TensorDataset(x_train, y_train);
        Sampler sampler = new RandomSampler(dataset.size());
        DataLoader batcher = new DataLoader(dataset, 256, sampler);

        try (var ignored = Tensor4j.noGrad()) {
            Tensor pred = model.forward(x_test);
            float accuracy = pred.argmax(1).eq(y_test.argmax(1)).mean().item();
            System.out.println("Before training, we have accuracy: " + accuracy);
        }

        int batches = 0;

        for (int epoch = 0; epoch < 3; epoch++) {
            System.out.println("Starting epoch " + (epoch + 1));
            for (Tensor[] batch : batcher) {
                batches++;

                optimizer.zeroGrad();
                Tensor pred = model.forward(batch[0]);
                Tensor l = loss.forward(pred, batch[1]);

                try (var ignored = Tensor4j.noGrad()) {
                    l.backward();
                    optimizer.step();
                }

                // eval
                try (var ignored = Tensor4j.noGrad()) {
                    Tensor predTest = model.forward(x_test);
                    float accuracy = predTest.argmax(1).eq(y_test.argmax(1)).mean().item();
                    System.out.println("Batch " + batches + " accuracy: " + accuracy);
                    if (accuracy > 0.90f) {
                        System.out.println("Achieved 90% accuracy!");
                        return;
                    }
                }
            }
            System.out.println("Finished training epoch " + (epoch + 1) + "... now evaluating");
        }

        fail("Failed to learn mnist in " + batches + " batches");
    }

    private static Tensor oneHot(Tensor labels, int numClasses) {
        int n = labels.getShape().dim(0);
        float[] out = new float[n * numClasses];
        for (int i = 0; i < n; i++) {
            int label = (int) labels.get(i, 0);
            out[i * numClasses + label] = 1f;
        }
        return TensorBuilder.builder().shape(n, numClasses).fromArray(out);
    }
}
