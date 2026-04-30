package com.cjcrafter.tensor4j.bench;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import com.cjcrafter.tensor4j.data.*;
import com.cjcrafter.tensor4j.io.NpzFormat;
import com.cjcrafter.tensor4j.nn.layers.Linear;
import com.cjcrafter.tensor4j.nn.layers.Module;
import com.cjcrafter.tensor4j.nn.layers.ReLU;
import com.cjcrafter.tensor4j.nn.layers.Sequential;
import com.cjcrafter.tensor4j.nn.losses.Loss;
import com.cjcrafter.tensor4j.nn.losses.MSELoss;
import com.cjcrafter.tensor4j.nn.optimizers.Adam;
import com.cjcrafter.tensor4j.nn.optimizers.AdamW;
import com.cjcrafter.tensor4j.nn.optimizers.Optimizer;
import com.cjcrafter.tensor4j.nn.optimizers.SGD;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * {@code sudo sysctl kernel.perf_event_paranoid=-1}
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 2, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx4g"})
public class AdamBenchmark {

    @Param({"1"})
    private int iterations;

    private Tensor x_train;
    private Tensor x_test;
    private Tensor y_train;
    private Tensor y_test;

    private Module model;
    private Loss loss;
    private Dataset dataset;
    private Sampler sampler;
    private DataLoader batcher;

    @Setup(Level.Trial)
    public void setup() throws IOException, URISyntaxException {
        Tensor4j.manualSeed(113);
        URL url = getClass().getClassLoader().getResource("mnist.npz");
        Path mnistPath = Path.of(url.toURI());
        Map<String, Tensor> tensors = NpzFormat.load(mnistPath);

        x_train = tensors.get("x_train").view(-1, 28 * 28).mul(1f / 255f);
        x_test = tensors.get("x_test").view(-1, 28 * 28).mul(1f / 255f);
        y_train = oneHot(tensors.get("y_train"), 10);
        y_test = oneHot(tensors.get("y_test"), 10);

        model = new Sequential(
                new Linear(28 * 28, 800),
                new ReLU(),
                new Linear(800, 10)
        );

        loss = new MSELoss();
        dataset = new TensorDataset(x_train, y_train);
        sampler = new RandomSampler(dataset.size());
        batcher = new DataLoader(dataset, 256, sampler);
    }

    @Benchmark
    public Tensor baseline() {
        Tensor4j.manualSeed(47);

        Collection<Tensor> parameters = model.parameters();
        Optimizer optimizer = new SGD(parameters, 1e-3f);

        Tensor bestLoss = null;
        for (int i = 0; i < iterations; i++) {
            for (Tensor[] batch : batcher) {

                optimizer.zeroGrad();
                Tensor pred = model.forward(batch[0]);
                Tensor l = loss.forward(pred, batch[1]);

                if (bestLoss == null || l.item() <= bestLoss.item())
                    bestLoss = l;

                try (var ignored = Tensor4j.noGrad()) {
                    l.backward();
                    // comment this out so we can try to measure specifically
                    // the optimizer itself, not just the forward/backward pass
                    //optimizer.step();
                }
            }
        }

        return bestLoss;
    }

    @Benchmark
    public Tensor adam() {
        Tensor4j.manualSeed(47);

        Collection<Tensor> parameters = model.parameters();
        Optimizer optimizer = new Adam(parameters, Adam.DEFAULT_LR, Adam.DEFAULT_BETA1, Adam.DEFAULT_BETA2, Adam.DEFAULT_EPS);

        Tensor bestLoss = null;
        for (int i = 0; i < iterations; i++) {
            for (Tensor[] batch : batcher) {

                optimizer.zeroGrad();
                Tensor pred = model.forward(batch[0]);
                Tensor l = loss.forward(pred, batch[1]);

                if (bestLoss == null || l.item() <= bestLoss.item())
                    bestLoss = l;

                try (var ignored = Tensor4j.noGrad()) {
                    l.backward();
                    optimizer.step();
                }
            }
        }

        return bestLoss;
    }

    @Benchmark
    public Tensor adamw() {
        Tensor4j.manualSeed(47);

        Collection<Tensor> parameters = model.parameters();
        Optimizer optimizer = new AdamW(parameters, AdamW.DEFAULT_LR, AdamW.DEFAULT_BETA1, AdamW.DEFAULT_BETA2, 0f, AdamW.DEFAULT_EPS);

        Tensor bestLoss = null;
        for (int i = 0; i < iterations; i++) {
            for (Tensor[] batch : batcher) {

                optimizer.zeroGrad();
                Tensor pred = model.forward(batch[0]);
                Tensor l = loss.forward(pred, batch[1]);

                if (bestLoss == null || l.item() <= bestLoss.item())
                    bestLoss = l;

                try (var ignored = Tensor4j.noGrad()) {
                    l.backward();
                    optimizer.step();
                }
            }
        }

        return bestLoss;
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
