package com.github.klane.wann.function.activation;

public final class Step implements ActivationFunction {

    private final double min;
    private final double max;
    private final double threshold;

    public Step(final double min, final double max, final double threshold) {
        this.min = min;
        this.max = max;
        this.threshold = threshold;
    }

    @Override
    public double applyAsDouble(final double input) {
        return (input >= this.threshold) ? this.max : this.min;
    }
}
