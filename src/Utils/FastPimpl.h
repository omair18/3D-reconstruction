/**
 * @file FastPimpl.h.
 *
 * Declares the fast "Pointer to IMPLementation" class
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

/**
 * @class FastPimpl
 *
 * @brief
 *
 * @tparam T Type of implementation
 * @tparam Size Size of type T in bytes
 * @tparam Alignment Alignment of type T in bytes
 */
template<class T, std::size_t Size, std::size_t Alignment>
class FastPimpl
{
public:

    /**
     *
     * @tparam Args
     * @param args
     */
    template<class...Args>
    explicit FastPimpl(Args &&...args)
    {
        new (Ptr()) T(std::forward<Args>(args)...);
    }

    /**
     *
     * @param fastPimpl
     * @return
     */
    FastPimpl &operator=(FastPimpl &&fastPimpl) noexcept
    {
        *Ptr() = std::move(*fastPimpl);
        return *this;
    }

    /**
     *
     * @param fastPimpl
     * @return
     */
    FastPimpl &operator=(const FastPimpl &fastPimpl)
    {
        Ptr()->~T();
        new (Ptr()) T(fastPimpl);
        return *this;
    }

    /**
     *
     * @return
     */
    T* operator->() noexcept
    {
        return Ptr();
    }

    /**
     *
     * @return
     */
    const T* operator->() const noexcept
    {
        return Ptr();
    }

    /**
     *
     * @return
     */
    T& operator*() noexcept
    {
        return *Ptr();
    }

    /**
     *
     * @return
     */
    const T& operator*() const noexcept
    {
        return *Ptr();
    }

    /**
     *
     */
    ~FastPimpl() noexcept
    {
        validate<sizeof(T), alignof(T)>();
        Ptr()->~T();
    }

    /**
     *
     * @return
     */
    T* Ptr()
    {
        return reinterpret_cast<T*>(&m_data);
    }

private:

    ///
    std::aligned_storage<Size, Alignment> m_data;

    /**
     *
     * @tparam ActualSize
     * @tparam ActualAlignment
     */
    template<std::size_t ActualSize, std::size_t ActualAlignment>
    static void validate() noexcept
    {
        static_assert(Size == ActualSize, "Size and sizeof(T) mismatch");
        static_assert(Alignment == ActualAlignment, "Alignment and alignof(T) mismatch");
    }

};
}

#endif //FAST_PIMPL_H
