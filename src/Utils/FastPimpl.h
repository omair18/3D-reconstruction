/**
 * @file FastPimpl.h.
 *
 * Declares the stacktrace dumper class
 */

#ifndef FAST_PIMPL_H
#define FAST_PIMPL_H

#include <type_traits>
#include <utility>

/**
 * @namespace Utils
 *
 * @brief Namespace of libutils library
 */
namespace Utils
{

template<class T, std::size_t Size, std::size_t Alignment>
class FastPimpl
{
public:
    template<class...Args>
    explicit FastPimpl(Args &&...args)
    {
        new (Ptr()) T(std::forward<Args>(args)...);
    }

    FastPimpl &operator=(FastPimpl &&fastPimpl) noexcept
    {
        *Ptr() = std::move(*fastPimpl);
        return *this;
    }

    FastPimpl &operator=(const FastPimpl &fastPimpl)
    {
        Ptr()->~T();
        new (Ptr()) T(fastPimpl);
        return *this;
    }

    T* operator->() noexcept
    {
        return Ptr();
    }

    const T* operator->() const noexcept
    {
        return Ptr();
    }

    T& operator*() noexcept
    {
        return *Ptr();
    }

    const T& operator*() const noexcept
    {
        return *Ptr();
    }

    ~FastPimpl() noexcept
    {
        validate<sizeof(T), alignof(T)>();
        Ptr()->~T();
    }

    T* Ptr()
    {
        return reinterpret_cast<T*>(&m_data);
    }

private:

    std::aligned_storage<Size, Alignment> m_data;

    template<std::size_t ActualSize, std::size_t ActualAlignment>
    static void validate() noexcept
    {
        static_assert(Size == ActualSize, "Size and sizeof(T) mismatch");
        static_assert(Alignment == ActualAlignment, "Alignment and alignof(T) mismatch");
    }

};
}

#endif //FAST_PIMPL_H
