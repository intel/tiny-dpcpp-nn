/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   sycl_graph.h
 *  @brief  Implementation of a SYCL graph capture with subsequent execution
 */

#pragma once

#include <common_host.h>

#include <sycl/sycl.hpp> //#include <cuda.h>

#include <deque>
#include <functional>

namespace tcnn {

class SyclGraph;

inline std::deque<SyclGraph*>& current_captures() {
	static thread_local std::deque<SyclGraph*> s_current_captures;
	return s_current_captures;
}

inline SyclGraph* current_capture() {
	return current_captures().empty() ? nullptr : current_captures().front();
}

namespace sycl_ext = sycl::ext::oneapi::experimental;
class SyclGraph {
    
public:
	~SyclGraph() {
		if(m_graph) delete m_graph;
	}
	SyclGraph() {
		m_graph = nullptr;
		//m_graph_instance = nullptr;
	}

	ScopeGuard capture_guard(sycl::queue * stream) {
		if (stream == nullptr ) {
			return ScopeGuard{};
		}

		// If the caller is already capturing, no need for a nested capture.
		sycl_ext::queue_state capture_status = stream->ext_oneapi_get_state();
		if (capture_status == sycl_ext::queue_state::recording) {
			return ScopeGuard{};
		}

		if(m_graph){
			delete m_graph;
			m_graph = nullptr;
		}
		if(!m_graph)
			m_graph = new sycl_ext::command_graph(stream->get_context(), stream->get_device());
		//CUDA_CHECK_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
		m_graph->begin_recording(* stream);
		current_captures().push_back(this);

		// Stop capturing again once the returned object goes out of scope
		return ScopeGuard{[this, stream]() {
			//CUDA_CHECK_THROW(cudaStreamEndCapture(stream, &m_graph));
			bool endRecordingSucceed = m_graph->end_recording(*stream);
			if (current_captures().back() != this) {
				throw std::runtime_error{"SyclGraph: must end captures in reverse order of creation."};
			}
			current_captures().pop_back();

			if (m_synchronize_when_capture_done) {
				//CUDA_CHECK_THROW(cudaDeviceSynchronize());
				stream->wait();  //FIXME, this is likely different from cudaDeviceSynchronize
				m_synchronize_when_capture_done = false;
			}
#if 0
			// If we previously created a graph instance, try to update it with the newly captured graph.
			// This is cheaper than creating a new instance from scratch (and may involve just updating
			// pointers rather than changing the topology of the graph.)
			if (m_graph_instance) {
			//FIXME. disable this as SYCL graph cannot update finalized graph
#if CUDA_VERSION >= 12000
				cudaGraphExecUpdateResultInfo update_result;
				CUDA_CHECK_THROW(cudaGraphExecUpdate(m_graph_instance, m_graph, &update_result));

				// If the update failed, reset graph instance. We will create a new one next.
				if (update_result.result != cudaGraphExecUpdateSuccess) {
					CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
					m_graph_instance = nullptr;
				}
#else
				cudaGraphExecUpdateResult update_result;
				cudaGraphNode_t error_node;
				CUDA_CHECK_THROW(cudaGraphExecUpdate(m_graph_instance, m_graph, &error_node, &update_result));

				// If the update failed, reset graph instance. We will create a new one next.
				if (update_result != cudaGraphExecUpdateSuccess) {
					CUDA_CHECK_THROW(cudaGraphExecDestroy(m_graph_instance));
					m_graph_instance = nullptr;
				}
#endif
			}
#endif
			auto m_graph_instance = m_graph->finalize();
			
			sycl::queue qexec{stream->get_context(), stream->get_device(),
				{sycl::ext::intel::property::queue::no_immediate_command_list()}};
			//CUDA_CHECK_THROW(cudaGraphLaunch(m_graph_instance, stream));
			qexec.ext_oneapi_graph(m_graph_instance).wait();
		}};
	}

	void schedule_synchronize() {
		m_synchronize_when_capture_done = true;
	}

private:
	sycl_ext::command_graph<sycl_ext::graph_state::modifiable>  *m_graph;
	//sycl_ext::command_graph<sycl_ext::graph_state::executable> *m_graph_instance;

	bool m_synchronize_when_capture_done = false;
};

}
