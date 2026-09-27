plugins {
    `java-library`
    jacoco
    id("com.gradleup.shadow") version "9.6.1"
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    implementation(libs.guava)
    implementation(libs.jetbrains.annotations)
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

jacoco {
    toolVersion = libs.versions.jacoco.get()
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.named<Test>("test") {
    useJUnitPlatform()
    maxHeapSize = "2g"
    jvmArgs(
        "--add-modules", "jdk.incubator.vector"
    )
    finalizedBy(tasks.named("jacocoTestReport"))
}

tasks.named<JacocoReport>("jacocoTestReport") {
    dependsOn(tasks.named("test"))
    reports {
        xml.required.set(true)
        html.required.set(true)
    }
}

tasks.named<JacocoCoverageVerification>("jacocoTestCoverageVerification") {
    dependsOn(tasks.named("jacocoTestReport"))

    violationRules {
        rule {
            limit {
                counter = "LINE"
                minimum = "0.70".toBigDecimal()
            }
        }
    }
}

tasks.named("check") {
    dependsOn(tasks.named("jacocoTestCoverageVerification"))
}

tasks.shadowJar {
    archiveClassifier.set("all")

    relocate("com.google.common", "com.cjcrafter.tensor4j.shaded.guava")
    relocate("com.google.thirdparty", "com.cjcrafter.tensor4j.shaded.thirdparty")

    dependencies {
        // compile/source-retention only; not needed at runtime
        exclude(dependency("org.jetbrains:annotations"))
    }

    // Preconditions is the only Guava API this library uses; strip the unused rest.
    minimize()
}
