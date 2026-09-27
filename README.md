# tensor4j (Java 21 course edition)

A small PyTorch-style tensor/autograd library for Java. Tensor 
storage is a plain heap `float[]`; the JVM garbage collector owns its lifecycle.

## Requirements

- Java 21+
- One JVM flag, both to compile and to run: `--add-modules jdk.incubator.vector`
  (this library uses the Java Vector API for fast elementwise/matmul kernels;
  it's still an incubator module on every current JDK line, so the flag is
  unavoidable regardless of Java version)

If you're running via `java -jar` or invoking a class directly, pass the flag
on the command line:

```
java --add-modules jdk.incubator.vector -cp tensor4j-all.jar YourMain
```

or set it once via an environment variable so you don't have to repeat it:

```
export JDK_JAVA_OPTIONS="--add-modules jdk.incubator.vector"
```

## Building

```
./gradlew :lib:test                              # run tests
./gradlew :lib:jacocoTestReport                   # coverage report (lib/build/reports/jacoco)
./gradlew :lib:shadowJar                          # fat jar at lib/build/libs/lib-all.jar
```

The fat jar bundles a relocated, minimized copy of Guava (used only for
`Preconditions`) so it won't collide with any Guava version you add yourself.
