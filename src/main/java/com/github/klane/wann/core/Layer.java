package com.github.klane.wann.core;

import com.google.common.base.Preconditions;

import java.util.*;
import java.util.stream.Collectors;

public final class Layer implements Iterable<Neuron> {

    private final String name;
    private final Layer previous;
    private Layer next;
    final Network network;
    final List<Neuron> neurons;

    private Layer(final Builder builder) {
        this.name = builder.name;
        this.network = builder.network;

        if (this.network.size() > 0) {
            this.previous = this.network.getLayer(this.network.size() - 1);
            this.previous.next = this;
        } else {
            this.previous = null;
        }

        this.neurons = builder.neurons.stream()
                .map(b -> b
                        .layer(this)
                        .name("L" + this.name.charAt(this.name.length() - 1) + "N" + (builder.neurons.indexOf(b) + 1))
                        .inputFunction(builder.inputFunction)
                        .activationFunction(builder.activationFunction)
                        .bias(builder.bias)
                        .bias(builder.biasFlag)
                        .build())
                .collect(Collectors.toList());
    }

    public static Builder builder() {
        return new Builder();
    }

    public Neuron get(int i) {
        Preconditions.checkElementIndex(i, this.neurons.size(), "Invalid neuron index");
        return this.neurons.get(i);
    }

    public String getName() {
        return this.name;
    }

    public Layer getNext() {
        return this.next;
    }

    public Layer getPrevious() {
        return this.previous;
    }

    @Override
    public Iterator<Neuron> iterator() {
        return this.neurons.iterator();
    }

    public int size() {
        return this.neurons.size();
    }

    @Override
    public String toString() {
        return this.name;
    }

    void calculate() {
        this.neurons.forEach(Neuron::calculate);
    }

    public static final class Builder extends WANNBuilder<Layer, Builder> {

        private Network network;
        final List<Neuron.Builder> neurons;

        private Builder() {
            this.neurons = new ArrayList<>();
        }

        @Override
        public Layer build() {
            Preconditions.checkArgument(this.neurons.size() > 0, "Empty layer");
            return new Layer(this);
        }

        public Builder neuron(final Collection<Neuron.Builder> neurons) {
            this.neurons.addAll(neurons);
            return this;
        }

        public Builder neuron(final double... weights) {
            Neuron.Builder builder = Neuron.builder();

            for (int i=0; i<weights.length; i++) {
                builder.connection(i, weights[i]);
            }

            return this.neuron(builder);
        }

        public Builder neuron(final Neuron.Builder... neurons) {
            return this.neuron(Arrays.asList(neurons));
        }

        @Override
        Builder get() {
            return this;
        }

        Builder network(final Network network) {
            this.network = network;
            return this;
        }
    }
}
