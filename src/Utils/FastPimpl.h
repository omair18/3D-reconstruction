/**
 * @file FastPimpl.h.
 *
 * @brief Declares the fast "Pointer to IMPLementation" class
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
 * @brief A template class for Fast PIMPL. It is used to avoid dynamic memory allocation and make the object of
 * class T cache-friendly.
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
     * @brief This constructor uses placement-new to create T-class object at data_.
     *
     * @tparam Args - variadic template param of arguments of T-class constructor
     * @param args - Arguments of T-class constructor
     */
    template<class...Args>
    explicit FastPimpl(Args &&...args)
    {
        new (Ptr()) T(std::forward<Args>(args)...);
    }

    /**
     * @brief Operator = for R-value param. Moves data from other object to current.
     *
     * @param fastPimpl - R-value reference for other FastPimpl object
     * @return L-value reference to current FastPimpl object.
     */
    FastPimpl &operator=(FastPimpl&& fastPimpl) noexcept
    {
        *Ptr() = std::move(*fastPimpl);
        return *this;
    }

    /**
     * @brief Operator = for L-value param. Destroys current data and creates a copy of other object's data.
     *
     * @param fastPimpl - L-value reference for other FastPimpl object
     * @return L-value reference to current FastPimpl object.
     */
    FastPimpl &operator=(const FastPimpl &fastPimpl)
    {
        Ptr()->~T();
        new (Ptr()) T(fastPimpl);
        return *this;
    }

    /**
     * @brief Provides access to data_ as T-class pointer.
     *
     * @return Pointer to T-class object at data_.
     */
    T* operator->() noexcept
    {
        return Ptr();
    }

    /**
     * @brief Provides access to data_ as T-class constant pointer.
     *
     * @return Constant pointer to T-class object at data_.
     */
    const T* operator->() const noexcept
    {
        return Ptr();
    }

    /**
     * @brief Provides access to data_ as T-class object.
     *
     * @return L-value reference to object from data_.
     */
    T& operator*() noexcept
    {
        return *Ptr();
    }

    /**
     * @brief Provides access to data_ as T-class constant object.
     *
     * @return Constant L-value reference to object from data_.
     */
    const T& operator*() const noexcept
    {
        return *Ptr();
    }

    /**
     * @brief Destroys T-class object at data_.
     */
    ~FastPimpl() noexcept
    {
        validate<sizeof(T), alignof(T)>();
        Ptr()->~T();
    }

    /**
     * @brief Provides access to data_ as T-class pointer.
     *
     * @return Pointer to T-class object at data_.
     */
    T* Ptr()
    {
        return reinterpret_cast<T*>(&data_);
    }

private:

    /// Aligned memory segment for placing data.
    std::aligned_storage<Size, Alignment> data_;

    /**
     * @brief Compares Size and Alignment template params with real size and alignment of T-class at compile time.
     * If there is a value mismatch, there will be a compilation error. Error information will contain correct values
     * of size and alignment.
     *
     * @tparam ActualSize - Real size of T-class object
     * @tparam ActualAlignment - Real alignment of T-class object
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
