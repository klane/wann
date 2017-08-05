package com.github.klane.wekanet.function.input;

import com.github.klane.wekanet.core.Connection;

import java.util.List;
import java.util.function.ToDoubleFunction;

public interface InputFunction extends ToDoubleFunction<List<Connection>> {}
