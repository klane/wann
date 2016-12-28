package com.github.klane.wann.util;

import com.google.common.base.Preconditions;
import javafx.util.Builder;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

public final class DataSetBuilder implements Builder<Instances> {

    private final ArrayList<Attribute> attributes;
    private final Collection<InstanceBuilder> instances;
    private Attribute classAttribute;
    private int classIndex;
    private String name;
    private File arffLoad;
    private File arffSave;

    public DataSetBuilder() {
        this.attributes = new ArrayList<>();
        this.instances = new ArrayList<>();
        this.classAttribute = null;
        this.classIndex = -1;
    }

    public DataSetBuilder name(final String name) {
        this.name = name;
        return this;
    }

    public DataSetBuilder addAttribute(final Attribute... attributes) {
        Preconditions.checkNotNull(attributes);
        return this.addAttribute(Arrays.asList(attributes));
    }

    public DataSetBuilder addAttribute(final Collection<Attribute> attributes) {
        Preconditions.checkNotNull(attributes);
        this.attributes.addAll(attributes);
        return this;
    }

    public DataSetBuilder setClass(final Attribute a) {
        if (!this.attributes.contains(a)) {
            //TODO throw exception
        }
        this.classAttribute = a;
        return this;
    }

    public DataSetBuilder setClassIndex(final int index) {
        //TODO check if attribute already set
        this.classIndex = index;
        return this;
    }

    public DataSetBuilder addInstance(final InstanceBuilder... instances) {
        Preconditions.checkNotNull(instances);
        return this.addInstance(Arrays.asList(instances));
    }

    public DataSetBuilder addInstance(final Collection<InstanceBuilder> instances) {
        Preconditions.checkNotNull(instances);
        this.instances.addAll(instances);
        return this;
    }

    public DataSetBuilder arffLoad(final File arff) {
        Preconditions.checkNotNull(arff);
        Preconditions.checkArgument(arff.exists(), "Specified file does not exist");
        this.arffLoad = arff;
        return this;
    }

    public DataSetBuilder arffSave(final File arff) {
        Preconditions.checkNotNull(arff);
        this.arffSave = arff;
        return this;
    }

    @Override
    public Instances build() {
        Instances temp = null;

        if (this.arffLoad != null) {
            try {
                ArffLoader loader = new ArffLoader();
                loader.setFile(this.arffLoad);
                temp = loader.getDataSet();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }

        Instances dataSet = (temp == null) ? new Instances(this.name, this.attributes, this.instances.size()) : temp;

        if (this.classIndex == -1 && this.classAttribute == null) {
            dataSet.setClassIndex(dataSet.numAttributes() - 1);
        } else if (this.classAttribute != null) {
            dataSet.setClass(this.classAttribute);
        } else {
            dataSet.setClassIndex(this.classIndex);
        }

        this.instances.stream().map(i -> i.dataSet(dataSet).build()).forEach(dataSet::add);

        if (this.arffSave != null) {
            ArffSaver saver = new ArffSaver();

            try {
                saver.setInstances(dataSet);
                saver.setFile(this.arffSave);
                saver.writeBatch();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }

        return dataSet;
    }
}
