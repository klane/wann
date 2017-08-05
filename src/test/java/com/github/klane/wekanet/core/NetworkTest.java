package com.github.klane.wekanet.core;

import static org.junit.Assert.assertEquals;

import com.github.klane.wekanet.function.activation.ActivationFunctions;
import com.github.klane.wekanet.function.activation.Step;
import com.github.klane.wekanet.util.DataSetBuilder;
import com.github.klane.wekanet.util.WekaUtils;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public final class NetworkTest {

    private static final String PATH = "./src/test/resources/";

    @Test
    public void nominalXNOR() throws Exception {
        Instances instances = new DataSetBuilder().arffLoad(new File(PATH + "XNOR_nominal.arff")).build();

        Network network = Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 2.0).connection(1, 2.0).bias(-3.0),
                        Neuron.builder().connection(0, -2.0).connection(1, -2.0).bias(1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, -2.0).connection(1, -2.0).bias(1.0),
                        Neuron.builder().connection(0, 2.0).connection(1, 2.0).bias(-1.0)))
                .build();

        networksClassifyInstances(instances, network);
    }

    @Test
    public void nominalXOR() throws Exception {
        Instances instances = new DataSetBuilder().arffLoad(new File(PATH + "XOR_nominal.arff")).build();

        Network network = Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-2.0),
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 2.0).connection(1, -1.0).bias(0.5),
                        Neuron.builder().connection(0, -1.0).connection(1, 1.0).bias(-1.0)))
                .build();

        networksClassifyInstances(instances, network);
    }

    @Test
    public void numericXNOR() throws Exception {
        Instances instances = new DataSetBuilder().arffLoad(new File(PATH + "XNOR_numeric.arff")).build();

        Network network = Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 2.0).connection(1, 2.0).bias(-3.0),
                        Neuron.builder().connection(0, -2.0).connection(1, -2.0).bias(1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuronWithBias(-1.0, 2.0, 2.0))
                .build();

        networksClassifyInstances(instances, network);
    }

    @Test
    public void numericXOR() throws Exception {
        Instances instances = new DataSetBuilder().arffLoad(new File(PATH + "XOR_numeric.arff")).build();
        List<Network> networks = new ArrayList<>();
        Step step = new Step(0.0, 1.0, 1.0);

        networks.add(Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 1.0).bias(-1.0),
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-2.0),
                        Neuron.builder().connection(1, 1.0).bias(-1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuronWithBias(-1.0, 1.0, -2.0, 1.0))
                .build());

        networks.add(Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(step).neuron(
                        Neuron.builder().connection(0, 1.0),
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-1.0),
                        Neuron.builder().connection(1, 1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, -2.0).connection(2, 1.0).bias(-1.0)))
                .build());

        networks.add(Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(step).neuron(
                        Neuron.builder().connection(0, 1.0),
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-1.0),
                        Neuron.builder().connection(1, 1.0)))
                .layer(Layer.builder().activationFunction(step).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, -2.0).connection(2, 1.0)))
                .build());

        networks.add(Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-2.0),
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuronWithBias(-1.0, -1.0, 1.0))
                .build());

        networks.add(Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(step).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-1.0),
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, -1.0).connection(1, 1.0).bias(-1.0)))
                .build());

        networks.add(Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(step).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-1.0),
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0)))
                .layer(Layer.builder().activationFunction(step).neuron(
                        Neuron.builder().connection(0, -1.0).connection(1, 1.0)))
                .build());

        networks.add(Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, -1.0).bias(-1.0),
                        Neuron.builder().connection(0, -1.0).connection(1, 1.0).bias(-1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuronWithBias(-1.0, 1.0, 1.0))
                .build());

        networks.add(Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(step).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, -1.0),
                        Neuron.builder().connection(0, -1.0).connection(1, 1.0)))
                .layer(Layer.builder().activationFunction(ActivationFunctions.STEP).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0).bias(-1.0)))
                .build());

        networks.add(Network.builder()
                .name(instances.relationName())
                .inputs(WekaUtils.attributesIgnoreClass(instances))
                .layer(Layer.builder().activationFunction(step).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, -1.0),
                        Neuron.builder().connection(0, -1.0).connection(1, 1.0)))
                .layer(Layer.builder().activationFunction(step).neuron(
                        Neuron.builder().connection(0, 1.0).connection(1, 1.0)))
                .build());

        networksClassifyInstances(instances, networks.toArray(new Network[1]));
    }

    private void networksClassifyInstances(final Instances instances, final Network... networks) throws Exception {
        for (Network n : networks) {
            for (Instance i : WekaUtils.toList(instances)) {
                assertEquals(n.classifyInstance(i), i.classValue(), 0);
            }
        }
    }
}
