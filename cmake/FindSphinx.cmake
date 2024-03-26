# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# The following definitions are added on success
# 
#  Sphinx_FOUND, a variable which stores whether Sphinx was found or not
#  Sphinx_BUILD_COMMAND, the Sphinx command which build the documentation

include(FindPackageHandleStandardArgs)

find_package(PythonInterp)

function (has_python_module out_var module)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import ${module}"
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
        RESULT_VARIABLE status)
    
    if (status EQUAL 0)
        set(${out_var} TRUE PARENT_SCOPE)
        return()
    endif ()
    message(STATUS "Sphinx: Could NOT find Python module \"${module}\"")
    if (NOT error MATCHES "ModuleNotFoundError" AND NOT MATCHES "ImportError")
        message(${error})
    endif ()
    set(${out_var} FALSE PARENT_SCOPE)
endfunction ()

if (PYTHONINTERP_FOUND)
    set(BUILD_MODULE "sphinx.cmd.build")

    has_python_module(HAS_SPHINX_BUILD ${BUILD_MODULE})

    set(HAS_REQUIRED_EXTENSIONS TRUE)
    foreach (sphinx_ext IN LISTS ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
        has_python_module(HAS_EXTENSION ${sphinx_ext})
        if (NOT HAS_EXTENSION)
            set(HAS_REQUIRED_EXTENSIONS FALSE)
        endif ()
    endforeach ()

    if (HAS_SPHINX_BUILD AND HAS_REQUIRED_EXTENSIONS)
        set(Sphinx_FOUND TRUE)
        set(Sphinx_BUILD_COMMAND ${PYTHON_EXECUTABLE} -m ${BUILD_MODULE})
    endif ()

endif ()

find_package_handle_standard_args(Sphinx DEFAULT_MSG Sphinx_BUILD_COMMAND)