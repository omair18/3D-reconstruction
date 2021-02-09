#include <iostream>
#include <tuple>


int main()
{
    std::apply([](auto&&... args)
    {
        ((std::cout << args), ...);
    },
    std::make_tuple("This is ", "Sparta!!!"));
    return 0;
}
