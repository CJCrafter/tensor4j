package com.cjcrafter.tensor4j.bench;

import com.cjcrafter.tensor4j.ops.Kernels;
import org.openjdk.jmh.annotations.*;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 2, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx4g"})
public class MeanBenchmark {

    @Param({"64", "4096", "65536"})
    private int size;

    private float[] data;

    @Setup(Level.Trial)
    public void setup() {
        Random rng = new Random(42);
        data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = (float) rng.nextGaussian();
        }
    }

    @Benchmark
    public float sumThenDivide() {
        return Kernels.sum(data, 0, size) / size;
    }

    @Benchmark
    public float fmaMean() {
        // this is not really a fair comparison because this mean operation is
        // doing extra work to maintain stability
        return Kernels.mean(data, 0, size);
    }
}
