package com.github.klane.wekanet.function.normalization;

import java.util.Arrays;

public class Gaussian {

    private final double avg;
    private final double std;

    public Gaussian(final double[] values) {
        this.avg = Arrays.stream(values).summaryStatistics().getAverage();
        this.std = Math.sqrt(Arrays.stream(values).map(d -> (d - this.avg) * (d - this.avg)).average().getAsDouble());
    }

    public double[] normalize(final double[] values) {
        return Arrays.stream(values).map(d -> (d - this.avg) / this.std).toArray();
    }
}
