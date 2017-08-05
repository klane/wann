package com.github.klane.wekanet.util;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public final class WekaUtils {

    private WekaUtils() {
        throw new UnsupportedOperationException();
    }

    public static List<Attribute> attributes(final Instances dataSet) {
        List<Attribute> attributes = new ArrayList<>(dataSet.numAttributes());

        for (int i=0; i<dataSet.numAttributes(); i++) {
            attributes.add(dataSet.attribute(i));
        }

        return attributes;
    }

    public static List<Attribute> attributesIgnoreClass(final Instances dataSet) {
        List<Attribute> attributes = new ArrayList<>(dataSet.numAttributes()-1);
        Attribute classAttribute = dataSet.classAttribute();

        for (int i=0; i<dataSet.numAttributes(); i++) {
            if (!dataSet.attribute(i).equals(classAttribute)) {
                attributes.add(dataSet.attribute(i));
            }
        }

        return attributes;
    }

    public static List<Instance> toList(final Instances dataSet) {
        List<Instance> instances = new ArrayList<>(dataSet.numInstances());

        for (int i=0; i<dataSet.numInstances(); i++) {
            instances.add(dataSet.instance(i));
        }

        return instances;
    }

    public static List<String> values(final Attribute attribute) {
        List<String> values = new ArrayList<>(attribute.numValues());

        for (int i=0; i<attribute.numValues(); i++) {
            values.add(attribute.value(i));
        }

        return values;
    }
}
