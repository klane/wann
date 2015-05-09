package com.github.klane.wann.function.activation;

public final class Linear implements ActivationFunction {

    private final double slope;

    public Linear(final double slope) {
        this.slope = slope;
    }

    @Override
    public double applyAsDouble(final double input) {
        return this.slope * input;
    }

    @Override
    public double derivative(final double input) {
        return this.slope;
    }
}
