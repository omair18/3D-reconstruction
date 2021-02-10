#include <boost/json.hpp>
#include <iostream>


int main()
{
    boost::json::value a = {
            {"a", 1233},
            {"b", {1, 2, 3} }
    };

    std::vector<int> b;

    auto& arr = a.at("b").as_array();

    b.reserve(arr.size());

    for(auto& val : arr)
    {
        b.push_back(val.as_int64());
    }
    a.at("b").as_object();

    std::cout << a << std::endl;
    return 0;
}