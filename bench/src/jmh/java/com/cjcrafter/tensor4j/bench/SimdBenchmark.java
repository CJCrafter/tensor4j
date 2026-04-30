package com.cjcrafter.tensor4j.bench;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 2, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx4g"})
public class SimdBenchmark {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    @Param({"2"})
    private int x;

    @Param({"3"})
    private int y;

    @Param({"100000"})
    private int iterations;

    private FloatVector a;
    private FloatVector b;

    @Setup(Level.Trial)
    public void setup() {
        a = FloatVector.broadcast(SPECIES, x);
        b = FloatVector.broadcast(SPECIES, y);
    }

    @Benchmark
    public FloatVector simd() {
        FloatVector sum = FloatVector.broadcast(SPECIES, 0);
        for (int i = 0; i < iterations; i++) {
            sum = sum.add(a.add(b).mul(a.sub(b)));
        }
        return sum;
    }
}
