/** @file   ScopeGuard.h
 */
#pragma once

namespace tinydpcppnn{

/**
 * This class registers a callback function, 
 * which will be called when the object of this class gets destructed.
*/
class ScopeGuard {
public:
	ScopeGuard() = default;
	ScopeGuard(const std::function<void()>& callback) : m_callback{callback} {}
	ScopeGuard(std::function<void()>&& callback) : m_callback{std::move(callback)} {}
	ScopeGuard& operator=(const ScopeGuard& other) = delete;
	ScopeGuard(const ScopeGuard& other) = delete;
	ScopeGuard& operator=(ScopeGuard&& other) { std::swap(m_callback, other.m_callback); return *this; }
	ScopeGuard(ScopeGuard&& other) { *this = std::move(other); }
	~ScopeGuard() { if (m_callback) { m_callback(); } }

	void disarm() {
		m_callback = {};
	}
private:
	std::function<void()> m_callback;
};

}


