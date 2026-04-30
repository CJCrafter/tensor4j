package com.cjcrafter.tensor4j.bench;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.openjdk.jmh.annotations.*;

import java.util.Random;
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
public class AddBenchmark {

    @Param({"64", "256", "1024"})
    private int size;

    private Tensor a;
    private Tensor b;
    private float c;

    @Setup(Level.Trial)
    public void setup() {
        Random rng = new Random(42);
        a = TensorBuilder.builder().shape(size, size).randn(rng);
        b = TensorBuilder.builder().shape(size, size).randn(rng);
        c = (float) rng.nextGaussian();
    }

    @Benchmark
    public Tensor add() {
        return a.add(b);
    }

    @Benchmark
    public Tensor addConst() {
        return a.add(c);
    }

    @Benchmark
    public Tensor mul() {
        return a.mul(b);
    }

    @Benchmark
    public Tensor mulConst() {
        return a.mul(c);
    }
}
