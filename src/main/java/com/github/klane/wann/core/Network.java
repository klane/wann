package com.github.klane.wann.core;

import com.github.klane.wann.function.activation.ActivationFunctions;
import com.github.klane.wann.function.input.InputFunctions;
import com.github.klane.wann.learn.LearningRule;
import com.github.klane.wann.util.WekaUtils;
import com.google.common.base.Preconditions;
import javafx.util.Builder;
import lombok.Getter;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public final class Network extends AbstractClassifier implements Iterable<Layer> {

    @Getter private final String name;
    @Getter private final Layer inputLayer;
    @Getter private final Layer outputLayer;
    private final List<Layer> layers;
    private final LearningRule learningRule;
    private double[] epochError;
    final Neuron bias;

    private Network(final NetworkBuilder builder) {
        this.name = builder.name;
        this.layers = new ArrayList<>();
        this.learningRule = builder.learningRule;

        this.bias = Neuron.builder().name("Bias").build();
        this.bias.setValue(1);

        this.inputLayer = Layer.builder()
                .name("Input Layer")
                .network(this)
                .activationFunction(ActivationFunctions.IDENTITY)
                .neuron(builder.inputValues.stream().map(s -> Neuron.builder().name(s)).collect(Collectors.toList()))
                .build();

        this.layers.add(this.inputLayer);

        (builder.outputValues.isEmpty() ? builder.layers.subList(0, builder.layers.size()-1) : builder.layers)
                .forEach(b -> this.layers.add(b
                        .name("Hidden Layer " + (builder.layers.indexOf(b) + 1))
                        .network(this)
                        .inputFunction(InputFunctions.WEIGHTED_SUM)
                        .activationFunction(ActivationFunctions.SIGMOID)
                        .build()));

        final Layer.LayerBuilder outputBuilder = (builder.outputValues.isEmpty() ?
                builder.layers.get(builder.layers.size()-1) : Layer.builder())
                .name("Output Layer")
                .network(this)
                .inputFunction(InputFunctions.WEIGHTED_SUM)
                .activationFunction(ActivationFunctions.SOFTMAX);

        if (builder.outputValues.isEmpty()) {
            List<Neuron.NeuronBuilder> neurons = builder.layers.get(builder.layers.size()-1).neurons;
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
        Preconditions.checkNotNull(this.learningRule, "Must specify a learning rule");
        //TODO check if data set class attribute matches network output attribute
        //TODO reset network to initial state?

        this.epochError = this.learningRule.apply(this, dataSet);
    }

    public static NetworkBuilder builder() {
        return new NetworkBuilder();
    }

    @Override
    public double[] distributionForInstance(final Instance instance) {
        Preconditions.checkNotNull(instance);
        int i=0;
        int index=0;
        double[] input = new double[this.inputLayer.size()];

        for (Neuron neuron : this.inputLayer) {
            Attribute a = instance.attribute(i);

            //TODO check logic
            if ((a.isNumeric() || a.numValues() < 3) ? neuron.getName().contains(a.name()) :
                    neuron.getName().contains(WekaUtils.values(a).get((int) instance.value(i)))) {
                if (a.isNumeric() || a.numValues() < 3) {
                    input[index++] = instance.value(i);
                } else {
                    input[index++] = 1;
                }

                i++;
            } else {
                input[index++] = 0;
            }
        }

        this.inputLayer.setInput(input);
        this.calculate();

        return this.outputLayer.getOutput();
    }

    public double[] getEpochError() {
        return (this.epochError != null) ? Arrays.copyOf(this.epochError, this.epochError.length) : new double[0];
    }

    public Layer getLayer(int i) {
        Preconditions.checkElementIndex(i, this.layers.size(), "Invalid layer index");
        return this.layers.get(i);
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

    public static final class NetworkBuilder implements Builder<Network> {

        private String name;
        private final List<String> inputValues;
        private final List<String> outputValues;
        private final List<Layer.LayerBuilder> layers;
        private LearningRule learningRule;

        private NetworkBuilder() {
            this.layers = new ArrayList<>();
            this.inputValues = new ArrayList<>();
            this.outputValues = new ArrayList<>();
        }

        @Override
        public Network build() {
            Preconditions.checkArgument(this.layers.size() > 0, "Empty network");
            return new Network(this);
        }

        public NetworkBuilder dataSet(final Instances dataSet) {
            Preconditions.checkNotNull(dataSet);

            return this.name(dataSet.relationName())
                    .inputs(WekaUtils.attributesIgnoreClass(dataSet)).output(dataSet.classAttribute());
        }

        public NetworkBuilder inputs(final Attribute... inputs) {
            Preconditions.checkNotNull(inputs);
            return this.inputs(Arrays.asList(inputs));
        }

        public NetworkBuilder inputs(final Collection<Attribute> inputs) {
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

        public NetworkBuilder layer(final int... numNeurons) {
            Preconditions.checkNotNull(numNeurons);

            IntStream.range(0, numNeurons.length).forEach(i -> this.layer(Layer.builder().neuron(
                    IntStream.range(0, numNeurons[i]).mapToObj(j -> Neuron.builder()).collect(Collectors.toList()))));

            return this;
        }

        public NetworkBuilder layer(final Layer.LayerBuilder layer) {
            Preconditions.checkNotNull(layer);
            this.layers.add(layer);
            return this;
        }

        public NetworkBuilder learningRule(final LearningRule learningRule) {
            Preconditions.checkNotNull(learningRule);
            this.learningRule = learningRule;
            return this;
        }

        public NetworkBuilder name(final String name) {
            this.name = name;
            return this;
        }

        public NetworkBuilder output(final Attribute output) {
            Preconditions.checkNotNull(output);

            if (output.isNumeric()) {
                this.outputValues.add(output.name());
            } else {
                this.outputValues.addAll(WekaUtils.values(output));
            }

            return this;
        }
    }
}
