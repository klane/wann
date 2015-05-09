package com.github.klane.wann.function.activation;

public enum ActivationFunctions implements ActivationFunction {

    LINEAR(new Linear(1)),
    RAMP(new Ramp(0, 1, 0, 1)),
    SIGMOID(new Logistic(1, 1, 0)),
    STEP(new Step(0, 1, 0)),
    TANH(new ActivationFunction() {
        @Override
        public double applyAsDouble(final double input) {
            return Math.tanh(input);
        }

        @Override
        public double derivative(final double input) {
            double output = this.applyAsDouble(input);
            return 1.0 - output * output;
        }
    });

    private final ActivationFunction function;

    ActivationFunctions(final ActivationFunction function) {
        this.function = function;
    }

    public double applyAsDouble(final double input) {
        return this.function.applyAsDouble(input);
    }

    public double derivative(final double input) {
        return this.function.derivative(input);
    }
}
