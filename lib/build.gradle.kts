plugins {
    `java-library`
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    implementation(libs.guava)
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(26)
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.named<Test>("test") {
    useJUnitPlatform()
    maxHeapSize = "2g"
    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED"
    )
}
