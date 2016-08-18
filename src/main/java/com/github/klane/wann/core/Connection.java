package com.github.klane.wann.core;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;

@AllArgsConstructor
public final class Connection {

    // TODO Check that NonNull is implemented
    @Getter @NonNull private final Neuron fromNeuron;
    @Getter @NonNull private final Neuron toNeuron;
    @Getter private double weight;

    Connection(final Neuron fromNeuron, final Neuron toNeuron) {
        this(fromNeuron, toNeuron, 2*Math.random() - 1);
    }

    public double getValue() {
        return this.fromNeuron.getValue();
    }

    public double getWeightedValue() {
        return this.fromNeuron.getValue() * this.weight;
    }

    @Override
    public String toString() {
        return this.fromNeuron.getName() + " -> " + this.toNeuron.getName() + ": " + this.weight;
    }

    public void updateWeight(final double delta) {
        this.weight += delta;
    }
}
