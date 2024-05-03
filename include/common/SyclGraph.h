/** @file   sycl_graph.h
 *  @author Yantao Zhang (yantao.zhang@intel.com)
 *  @brief  Head file for SYCL graph recording/execution functionalities
 */

#pragma once

#include <sycl/sycl.hpp>
#include <deque>
#include <functional>
#include <memory>
#include <ScopeGuard.h>

namespace tinydpcppnn{

namespace sycl_ext = sycl::ext::oneapi::experimental;
typedef sycl_ext::command_graph<sycl_ext::graph_state::modifiable> graph_t;
typedef sycl_ext::command_graph<sycl_ext::graph_state::executable> exegraph_t;

class SyclGraph {
public:
	~SyclGraph();
	SyclGraph();
	void reset();
	ScopeGuard capture_guard(sycl::queue * stream);
	void schedule_synchronize();

private:
	std::shared_ptr<graph_t>  graph;
	std::shared_ptr<exegraph_t> exeGraph; //finalized executable graph
	std::shared_ptr<sycl::queue> qexec;   //queue to execute the graph, this doesn't support immediate command list
	bool m_synchronize_when_capture_done = false;
};

}
