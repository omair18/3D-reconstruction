#include <boost/json.hpp>
#include <iostream>


int main()
{
    boost::json::value a = {
            {"a", 1233},
            {"b", {{
                "123", 123
            }}}
    };

    {
        boost::json::value test = a.at("b");
        test = {{"123", "abc"}};
        std::cout << test << std::endl;
    }

    a.as_object()["28"] = 27;
    std::cout << a << std::endl;
    return 0;
}