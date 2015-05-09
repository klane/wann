package com.github.klane.wann.function.activation;

public final class Logistic implements ActivationFunction {

    private final double max;
    private final double slope;
    private final double midpoint;

    public Logistic(final double max, final double slope, final double midpoint) {
        this.max = max;
        this.slope = slope;
        this.midpoint = midpoint;
    }

    @Override
    public double applyAsDouble(final double input) {
        return this.max / (1.0 + Math.exp(-this.slope * (input - this.midpoint)));
    }

    @Override
    public double derivative(final double input) {
        double output = this.applyAsDouble(input);
        return this.slope * output * (1.0 - output / this.max);
    }
}
