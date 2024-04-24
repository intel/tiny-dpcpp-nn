/** @file   sycl_graph.cpp
 *  @author Yantao Zhang (yantao.zhang@intel.com)
 *  @brief  Implementation of SYCL graph 
 *  @todo   Update the executable graph if needed.
 */

#include "SyclGraph.h"

namespace tinydpcppnn{

inline std::deque<SyclGraph*>& current_captures() {
    static thread_local std::deque<SyclGraph*> s_current_captures;
    return s_current_captures;
}

inline SyclGraph* current_capture() {
    return current_captures().empty() ? nullptr : current_captures().front();
}

SyclGraph::~SyclGraph() {
    reset();
}
SyclGraph::SyclGraph() {
    graph = nullptr;
    exeGraph = nullptr;
    qexec = nullptr;
}

void SyclGraph::reset(){
    if(graph){
        graph.reset();
        graph = nullptr;
    }
    if(exeGraph){
        exeGraph.reset();
        exeGraph = nullptr;
    }
    if(qexec){
        qexec.reset();
        qexec = nullptr;
    }
}

ScopeGuard SyclGraph::capture_guard(sycl::queue * stream) {
    if (stream == nullptr ) {
        return ScopeGuard{};
    }

    // If the caller is being recorded, stop recording.
    sycl_ext::queue_state capture_status = stream->ext_oneapi_get_state();
    if (capture_status == sycl_ext::queue_state::recording) {
        return ScopeGuard{};
    }

    //start recording
    if(!graph)
        graph = std::make_shared<graph_t>(stream->get_context(), stream->get_device());
    graph->begin_recording(* stream);
    current_captures().push_back(this);
    
    //register the callback function to be called when this object goes out of scope
    return ScopeGuard{[this, stream]() {
        // Stop recording
        graph->end_recording(*stream);
        if (current_captures().back() != this) {
            throw std::runtime_error{"SyclGraph: must end captures in reverse order of creation."};
        }
        current_captures().pop_back();

        if (m_synchronize_when_capture_done) {
            stream->wait();
            m_synchronize_when_capture_done = false;
        }

        //Todo update finalized graph if needed.
        
        if(!exeGraph)
            exeGraph = std::make_shared<exegraph_t>( graph->finalize());
        
        //
        if(!qexec)
            qexec = std::make_shared<sycl::queue>(stream->get_context(), stream->get_device(),
            sycl::property_list(sycl::ext::intel::property::queue::no_immediate_command_list()));
        qexec->ext_oneapi_graph(*exeGraph).wait();
    }};
}

void SyclGraph::schedule_synchronize() {
    m_synchronize_when_capture_done = true;
}

}