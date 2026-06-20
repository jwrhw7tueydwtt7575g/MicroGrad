add_library(micrograd_core STATIC
    core/storage.cpp
    core/tensor.cpp
    core/op.cpp
    core/op_registry.cpp
    core/ir.cpp
    core/function.cpp
    core/autograd.cpp
    core/module.cpp
    core/linear.cpp
    core/sequential.cpp
    core/relu.cpp
    core/softmax.cpp
    core/loss.cpp
    core/sgd.cpp
    core/momentum.cpp
    core/adam.cpp
    core/optimizer.cpp
    core/stream.cpp
    core/op_kernels.cpp
    core/active_fn.cpp
    core/serializer.cpp
    kernels/cpu/binary_kernels.cpp
    kernels/cpu/unary_kernels.cpp
    kernels/cpu/reduce_kernels.cpp
    kernels/cpu/matmul_kernels.cpp
)
target_include_directories(micrograd_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(micrograd_core PUBLIC Threads::Threads)

if(MICROGRAD_BUILD_PYTHON)
    pybind11_add_module(micrograd_python
        python/py_tensor.cpp
        python/py_function.cpp
        python/py_module.cpp
        python/py_optim.cpp
        python/py_loss.cpp
        python/py_op_registry.cpp
        python/py_serializer.cpp
        python/py_transforms.cpp
        python/module.cpp
    )
    target_link_libraries(micrograd_python PUBLIC micrograd_core)
    set_target_properties(micrograd_python PROPERTIES
        OUTPUT_NAME "_C"
    )
    # Install the compiled module into the python package directory so it
    # becomes importable as `micrograd._C` once installed.
    install(TARGETS micrograd_python
        LIBRARY DESTINATION micrograd
    )
endif()
