#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>


int main(int argc, char** argv)
{
    boost::program_options::options_description optionsDescription("Usage");
    boost::program_options::command_line_parser commandLineParser(argc, argv);



    commandLineParser.options(optionsDescription).run();
}