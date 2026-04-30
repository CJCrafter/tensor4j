package com.cjcrafter.tensor4j.bench;

import com.cjcrafter.tensor4j.Tensor;
import com.cjcrafter.tensor4j.Tensor4j;
import com.cjcrafter.tensor4j.TensorBuilder;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * {@code sudo sysctl kernel.perf_event_paranoid=-1}
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(value = 2, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx4g"})
public class MatmulBenchmark {

    @Param({"64", "256", "2048", "4096"})
    private int size;

    private Tensor a;
    private Tensor b;

    @Setup(Level.Trial)
    public void setup() {
        Tensor4j.manualSeed(42);
        a = TensorBuilder.builder().shape(size, size).randn();
        b = TensorBuilder.builder().shape(size, size).randn();
    }

    @Benchmark
    public Tensor matmul() {
        return a.matmul(b);
    }
}
