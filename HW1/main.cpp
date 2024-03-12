#include <iostream>
#include "DataHandler.hpp"

using namespace std;
int main()
{
    auto table = tools::readCsv("./HW1.csv");

    cout << tools::tableDetail(table) << "\n";
}