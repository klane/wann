package com.github.klane.wann.core;

import com.github.klane.wann.function.activation.ActivationFunctions;
import com.github.klane.wann.function.input.InputFunctions;
import com.github.klane.wann.learn.Backpropagation;
import com.github.klane.wann.learn.LearningRule;
import com.google.common.base.Preconditions;
import klane.weka.WekaUtils;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public final class Network extends Classifier implements Iterable<Layer> {

    private final String name;
    private final Layer inputLayer;
    private final Layer outputLayer;
    private final List<Layer> layers;
    private final LearningRule learningRule;
    private double[] epochError;
    final Neuron bias;

    private Network(final Builder builder) {
        this.name = builder.name;
        this.layers = new ArrayList<>();
        this.learningRule = builder.learningRule;

        this.bias = Neuron.builder().name("Bias").activationFunction(ActivationFunctions.LINEAR).build();
        this.bias.setInput(1);
        this.bias.calculate();

        this.inputLayer = Layer.builder()
                .name("Input Layer")
                .network(this)
                .neuron(builder.inputValues.stream()
                        .map(s -> Neuron.builder().name(s).activationFunction(ActivationFunctions.LINEAR))
                        .collect(Collectors.toList()))
                .bias(false)
                .build();

        this.layers.add(this.inputLayer);

        (builder.outputValues.isEmpty() ? builder.layers.subList(0, builder.layers.size()-1) : builder.layers)
                .forEach(b -> this.layers.add(b
                        .name("Hidden Layer " + (builder.layers.indexOf(b) + 1))
                        .network(this)
                        .inputFunction(builder.inputFunction)
                        .activationFunction(builder.activationFunction)
                        .bias(builder.bias)
                        .bias(builder.biasFlag)
                        .build()));

        final Layer.Builder outputBuilder = (builder.outputValues.isEmpty() ?
                builder.layers.get(builder.layers.size()-1) : Layer.builder())
                .name("Output Layer")
                .network(this)
                .inputFunction(builder.inputFunction)
                .activationFunction(builder.activationFunction)
                .bias(builder.bias)
                .bias(builder.biasFlag);

        if (builder.outputValues.isEmpty()) {
            List<Neuron.Builder> neurons = builder.layers.get(builder.layers.size()-1).neurons;
            IntStream.rangeClosed(0, neurons.size()-1).forEach(i -> neurons.get(i).name("Output " + (i + 1)));
        } else {
            outputBuilder.neuron(builder.outputValues.stream()
                    .map(s -> Neuron.builder().name(s)).collect(Collectors.toList()));
        }

        this.outputLayer = outputBuilder.build();
        this.layers.add(this.outputLayer);
    }

    @Override
    public void buildClassifier(final Instances dataSet) {
        Preconditions.checkNotNull(dataSet);
        //TODO check if data set class attribute matches network output attribute
        //TODO reset network to initial state?

        this.epochError = this.learningRule.apply(this, dataSet);
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public double[] distributionForInstance(final Instance instance) {
        Preconditions.checkNotNull(instance);
        int i=0;

        for (Neuron neuron : this.inputLayer) {
            Attribute a = instance.attribute(i);

            //TODO check logic
            if ((a.isNumeric() || a.numValues() < 3) ? neuron.getName().contains(a.name()) :
                    neuron.getName().contains(WekaUtils.values(a).get((int) instance.value(i)))) {
                if (a.isNumeric() || a.numValues() < 3) {
                    neuron.setInput(instance.value(i));
                } else {
                    neuron.setInput(1);
                }

                i++;
            } else {
                neuron.setInput(0);
            }
        }

        this.calculate();

        return this.outputLayer.neurons.stream().mapToDouble(Neuron::getOutput).toArray();
    }

    public double[] getEpochError() {
        return (this.epochError != null) ? Arrays.copyOf(this.epochError, this.epochError.length) : new double[0];
    }

    public Layer getInputLayer() {
        return this.inputLayer;
    }

    public Layer getLayer(int i) {
        Preconditions.checkElementIndex(i, this.layers.size(), "Invalid layer index");
        return this.layers.get(i);
    }

    public String getName() {
        return this.name;
    }

    public Layer getOutputLayer() {
        return this.outputLayer;
    }

    @Override
    public Iterator<Layer> iterator() {
        return this.layers.iterator();
    }

    public int size() {
        return this.layers.size();
    }

    @Override
    public String toString() {
        return this.name;
    }

    private void calculate() {
        this.layers.forEach(Layer::calculate);
    }

    public static final class Builder extends WANNBuilder<Network, Builder> {

        private final List<String> inputValues;
        private final List<String> outputValues;
        private final List<Layer.Builder> layers;
        private LearningRule learningRule;

        private Builder() {
            this.layers = new ArrayList<>();
            this.inputValues = new ArrayList<>();
            this.outputValues = new ArrayList<>();
        }

        @Override
        public Network build() {
            Preconditions.checkArgument(this.layers.size() > 0, "Empty network");

            if (super.inputFunction == null) {
                super.inputFunction(InputFunctions.WEIGHTED_SUM);
            }

            if (super.activationFunction == null) {
                super.activationFunction(ActivationFunctions.SIGMOID);
            }

            if (this.learningRule == null) {
                this.learningRule(new Backpropagation());
            }

            return new Network(this);
        }

        public Builder dataSet(final Instances dataSet) {
            Preconditions.checkNotNull(dataSet);

            return this.name(dataSet.relationName())
                    .inputs(WekaUtils.attributesIgnoreClass(dataSet)).output(dataSet.classAttribute());
        }

        public Builder inputs(final Attribute... inputs) {
            Preconditions.checkNotNull(inputs);
            return this.inputs(Arrays.asList(inputs));
        }

        public Builder inputs(final Collection<Attribute> inputs) {
            Preconditions.checkNotNull(inputs);

            for (Attribute a : inputs) {
                if (a.isNumeric() || a.numValues() < 3) {
                    this.inputValues.add(a.name());
                } else {
                    this.inputValues.addAll(WekaUtils.values(a).stream()
                            .map(s -> a.name() + "=" + s).collect(Collectors.toList()));
                }
            }

            return this;
        }

        public Builder layer(final int... numNeurons) {
            Preconditions.checkNotNull(numNeurons);

            IntStream.range(0, numNeurons.length).forEach(i -> this.layer(Layer.builder().neuron(
                    IntStream.range(0, numNeurons[i]).mapToObj(j -> Neuron.builder()).collect(Collectors.toList()))));

            return this;
        }

        public Builder layer(final Layer.Builder layer) {
            Preconditions.checkNotNull(layer);
            this.layers.add(layer);
            return this;
        }

        public Builder learningRule(final LearningRule learningRule) {
            Preconditions.checkNotNull(learningRule);
            this.learningRule = learningRule;
            return this;
        }

        public Builder output(final Attribute output) {
            Preconditions.checkNotNull(output);

            if (output.isNumeric()) {
                this.outputValues.add(output.name());
            } else {
                this.outputValues.addAll(WekaUtils.values(output));
            }

            return this;
        }

        @Override
        Builder get() {
            return this;
        }
    }
}
