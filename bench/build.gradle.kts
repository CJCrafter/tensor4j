plugins {
    java
    id("me.champeau.jmh") version "0.7.3"
}

repositories {
    mavenCentral()
}

dependencies {
    jmh(project(":lib"))
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(26)
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

jmh {
    jvmArgs = listOf(
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED"
    )
}
