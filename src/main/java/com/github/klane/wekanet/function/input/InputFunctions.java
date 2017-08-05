package com.github.klane.wann.function.input;

import com.github.klane.wann.core.Connection;
import lombok.AllArgsConstructor;

import java.util.List;

@AllArgsConstructor
public enum InputFunctions implements InputFunction {

    WEIGHTED_MAX(conn -> conn.stream().mapToDouble(Connection::getWeightedValue).max().getAsDouble()),
    WEIGHTED_MEAN(conn -> conn.stream().mapToDouble(Connection::getWeightedValue).average().getAsDouble()),
    WEIGHTED_SUM(conn -> conn.stream().mapToDouble(Connection::getWeightedValue).sum());

    private final InputFunction function;

    public double applyAsDouble(final List<Connection> value) {
        return this.function.applyAsDouble(value);
    }
}
