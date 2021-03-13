#include <boost/json.hpp>
#include <iostream>


int main()
{
    boost::json::value t1, t2;
    t1.emplace_object();
    t2.emplace_object();

    t1.as_object().emplace("1", 1);
    t2.as_object().emplace("1", 2);

    std::cout << (t1 == t2) << std::endl;

    return 0;
}