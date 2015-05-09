package com.github.klane.wann.function.error;

import java.util.stream.DoubleStream;

public enum ErrorFunctions implements ErrorFunction {

    MSE(error -> DoubleStream.of(error).map(d -> d * d).average().getAsDouble()),
    RMSE(error -> Math.sqrt(DoubleStream.of(error).map(d -> d * d).average().getAsDouble()));

    private final ErrorFunction function;

    ErrorFunctions(final ErrorFunction function) {
        this.function = function;
    }

    public double applyAsDouble(final double[] value) {
        return this.function.applyAsDouble(value);
    }
}
