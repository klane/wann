package com.github.klane.wann.function.input;

import com.github.klane.wann.core.Connection;

import java.util.List;

public enum InputFunctions implements InputFunction {

    WEIGHTED_MAX(conn -> conn.stream().mapToDouble(Connection::getWeightedValue).max().getAsDouble()),
    WEIGHTED_MEAN(conn -> conn.stream().mapToDouble(Connection::getWeightedValue).average().getAsDouble()),
    WEIGHTED_SUM(conn -> conn.stream().mapToDouble(Connection::getWeightedValue).sum());

    private final InputFunction function;

    InputFunctions(final InputFunction function) {
        this.function = function;
    }

    public double applyAsDouble(final List<Connection> value) {
        return this.function.applyAsDouble(value);
    }
}
