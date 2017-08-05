package com.github.klane.wann.learn;

import com.github.klane.wann.core.Connection;
import com.github.klane.wann.core.Neuron;

import java.util.HashMap;
import java.util.Map;

public final class MomentumBackpropagation extends Backpropagation {

    public static final double DEFAULT_MOMENTUM = 0.2;

    private final double momentum;
    private final Map<Connection, Double> previousWeights;

    public MomentumBackpropagation(final int numEpochs) {
        this(numEpochs, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM);
    }

    public MomentumBackpropagation(final int numEpochs, final double learningRate) {
        this(numEpochs, learningRate, DEFAULT_MOMENTUM);
    }

    public MomentumBackpropagation(final int numEpochs, final double learningRate, final double momentum) {
        super(numEpochs, learningRate);
        this.momentum = momentum;
        this.previousWeights = new HashMap<>();
    }

    @Override
    protected void beforeTraining() {
        super.network.forEach(layer -> layer.forEach(neuron -> neuron.forEach(
                connection -> this.previousWeights.put(connection, 0.0))));
    }

    @Override
    void updateNeuronWeights(final Neuron neuron) {
        super.updateNeuronWeights(neuron);

        neuron.getInputConnections().forEach(c -> {
            c.updateWeight(this.momentum * (c.getWeight() - this.previousWeights.get(c)));
            this.previousWeights.put(c, c.getWeight());
        });
    }
}
